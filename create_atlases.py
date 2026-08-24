"""Create decimated, textured trunk meshes from a TreeLearn-labeled cloud.

The script intentionally processes one tree at a time. This keeps peak memory
reasonable for the multi-million-point dense cloud and makes a small pilot
possible with ``--max-trees`` before processing the whole scene.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import laspy
import numpy as np
import open3d as o3d
import pycolmap
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image


@dataclass
class Camera:
    name: str
    path: Path
    width: int
    height: int
    colmap_image: object


def make_logger(quiet: bool) -> Callable[[str], None]:
    return (lambda message: None) if quiet else print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segmented_cloud", type=Path)
    parser.add_argument("colmap_model", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=float, default=5.0,
                        help="Height in world units retained above each tree base (default: 5 m).")
    parser.add_argument("--vertical-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--voxel-size", type=float, default=0.03,
                        help="Point-cloud voxel size before reconstruction (default: 0.03).")
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--target-triangles", type=int, default=12000)
    parser.add_argument("--atlas-size", type=int, default=2048)
    parser.add_argument("--min-points", type=int, default=500)
    parser.add_argument("--max-trees", type=int)
    parser.add_argument("--depth-tolerance", type=float, default=0.03,
                        help="Extra world-distance tolerance for ray visibility checks.")
    parser.add_argument("--views-per-face", type=int, default=6,
                        help="Maximum source views blended for each face (default: 6).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for texel projection and image sampling (default: cuda).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output; errors and the manifest are still produced.")
    return parser.parse_args()


def load_cameras(model_path: Path, image_root: Path,
                 log: Callable[[str], None]) -> list[Camera]:
    log(f"Loading COLMAP reconstruction: {model_path}")
    reconstruction = pycolmap.Reconstruction(str(model_path))
    log(f"  Found {len(reconstruction.images)} images and {len(reconstruction.cameras)} cameras")
    cameras: list[Camera] = []
    for colmap_image in reconstruction.images.values():
        path = image_root / colmap_image.name
        if not path.exists() or not colmap_image.has_pose:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError):
            continue
        cameras.append(Camera(colmap_image.name, path, width, height, colmap_image))
    if not cameras:
        raise RuntimeError("No posed COLMAP images could be opened")
    log(f"  Loaded {len(cameras)} posed source images from {image_root}")
    return cameras


def load_tree_points(path: Path, axis: int, height: float,
                     min_points: int, log: Callable[[str], None]) -> dict[int, np.ndarray]:
    log(f"Loading segmented point cloud: {path}")
    cloud = laspy.read(path)
    if "treeID" not in cloud.point_format.extra_dimension_names:
        raise ValueError("The segmented cloud has no treeID extra dimension")
    coordinates = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float32)
    labels = np.asarray(cloud["treeID"])
    label_ids = np.unique(labels)
    log(f"  Read {len(labels):,} points with {len(label_ids[label_ids > 0])} positive tree labels")
    trees: dict[int, np.ndarray] = {}
    discarded = 0
    for tree_id in label_ids:
        tree_id = int(tree_id)
        if tree_id <= 0:
            continue
        points = coordinates[labels == tree_id]
        base = float(points[:, axis].min())
        points = points[points[:, axis] <= base + height]
        if len(points) >= min_points:
            trees[tree_id] = points
        else:
            discarded += 1
    log(f"  Retained {len(trees)} trees after +{height:g} vertical crop; discarded {discarded} small trees")
    return trees


def build_mesh(points: np.ndarray, voxel_size: float, poisson_depth: int,
               target_triangles: int) -> tuple[np.ndarray, np.ndarray]:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if len(cloud.points) < 30:
        raise ValueError("too few points after voxel downsampling")
    radius = max(voxel_size * 3.0, 0.05)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    cloud.orient_normals_consistent_tangent_plane(min(30, len(cloud.points) - 1))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=poisson_depth, scale=1.1, linear_fit=True)
    density = np.asarray(densities)
    if len(density):
        mesh.remove_vertices_by_mask(density < np.quantile(density, 0.02))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    if target_triangles > 0 and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("surface reconstruction produced an empty mesh")
    return vertices, faces


def cylindrical_uv(vertices: np.ndarray, axis: int) -> np.ndarray:
    other = [index for index in range(3) if index != axis]
    # Unwrap around this tree's own vertical axis, not the scene origin.
    horizontal_center = np.median(vertices[:, other], axis=0)
    centered = vertices[:, other] - horizontal_center
    angle = np.arctan2(centered[:, 1], centered[:, 0])
    u = (angle / (2.0 * np.pi) + 0.5) % 1.0
    low, high = vertices[:, axis].min(), vertices[:, axis].max()
    v = (vertices[:, axis] - low) / max(high - low, 1e-6)
    return np.column_stack((u, v)).astype(np.float32)


def unwrap_face_u(face_uv: np.ndarray) -> np.ndarray:
    """Choose equivalent U coordinates with the smallest triangle span."""
    candidates = []
    for first_shift in (-1, 0, 1):
        for second_shift in (-1, 0, 1):
            for third_shift in (-1, 0, 1):
                shifts = np.array([first_shift, second_shift, third_shift], dtype=np.float32)
                values = face_uv[:, 0] + shifts
                candidates.append((float(values.max() - values.min()), values))
    _, best_u = min(candidates, key=lambda item: item[0])
    result = face_uv.copy()
    result[:, 0] = best_u
    return result


def camera_projection(camera: Camera, point: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
    projection = camera.colmap_image.project_point(point.astype(float))
    if projection is None:
        return None
    pose = camera.colmap_image.cam_from_world().matrix()
    camera_point = pose[:, :3] @ point + pose[:, 3]
    if camera_point[2] <= 0:
        return None
    return np.asarray(projection, dtype=np.float64), float(camera_point[2]), camera_point


def choose_camera(cameras: list[Camera], point: np.ndarray, scene: o3d.t.geometry.RaycastingScene,
                  depth_tolerance: float) -> tuple[Camera, np.ndarray] | None:
    best: tuple[float, Camera, np.ndarray] | None = None
    for camera in cameras:
        result = camera_projection(camera, point)
        if result is None:
            continue
        pixel, depth, camera_point = result
        width, height = camera.width, camera.height
        if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            continue
        pose = camera.colmap_image.cam_from_world().matrix()
        camera_center = -pose[:, :3].T @ pose[:, 3]
        direction = point - camera_center
        distance = float(np.linalg.norm(direction))
        if distance <= 0:
            continue
        ray = np.concatenate((camera_center, direction / distance)).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(ray[None, :]))["t_hit"].numpy()[0]
        if not np.isfinite(hit) or float(hit) > distance + depth_tolerance:
            continue
        # Prefer close, front-facing observations while avoiding extreme views.
        score = depth
        if best is None or score < best[0]:
            best = (score, camera, pixel)
    return None if best is None else (best[1], best[2])


def _camera_tensors(cameras: list[Camera], device: torch.device) -> tuple[torch.Tensor, ...]:
    poses = []
    intrinsics = []
    for camera in cameras:
        poses.append(camera.colmap_image.cam_from_world().matrix())
        params = np.asarray(camera.colmap_image.camera.params, dtype=np.float32)
        if camera.colmap_image.camera.model_name != "OPENCV_FISHEYE":
            raise ValueError(f"Unsupported COLMAP camera model: {camera.colmap_image.camera.model_name}")
        intrinsics.append(params)
    return (
        torch.as_tensor(np.asarray(poses), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(intrinsics), dtype=torch.float32, device=device),
    )


def _project_fisheye(points: torch.Tensor, poses: torch.Tensor,
                     intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project points into all cameras; output shapes are (cameras, points)."""
    camera_points = torch.einsum("cij,pj->cpi", poses[:, :, :3], points) + poses[:, :, 3][:, None, :]
    depth = camera_points[:, :, 2]
    normalized = camera_points[:, :, :2] / depth.clamp_min(1e-6)[..., None]
    radius = torch.linalg.vector_norm(normalized, dim=-1)
    theta = torch.atan(radius)
    theta2 = theta * theta
    distortion = (1 + intrinsics[:, 4, None] * theta2
                  + intrinsics[:, 5, None] * theta2 * theta2
                  + intrinsics[:, 6, None] * theta2 * theta2 * theta2
                  + intrinsics[:, 7, None] * theta2 * theta2 * theta2 * theta2)
    scale = torch.where(radius > 1e-8, theta * distortion / radius, torch.ones_like(radius))
    distorted = normalized * scale[..., None]
    pixels = distorted * intrinsics[:, None, :2] + intrinsics[:, None, 2:4]
    return pixels, depth


def _face_candidates(center: torch.Tensor, cameras: list[Camera], poses: torch.Tensor,
                     intrinsics: torch.Tensor, views_per_face: int) -> torch.Tensor:
    pixels, depth = _project_fisheye(center[None, :], poses, intrinsics)
    pixels, depth = pixels[:, 0], depth[:, 0]
    bounds = torch.tensor([[camera.width, camera.height] for camera in cameras],
                          dtype=torch.float32, device=center.device)
    valid = ((depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0)
             & (pixels[:, 0] < bounds[:, 0]) & (pixels[:, 1] < bounds[:, 1]))
    scores = torch.where(valid, depth, torch.tensor(float("inf"), device=center.device))
    return torch.argsort(scores)[:views_per_face][torch.isfinite(scores[torch.argsort(scores)[:views_per_face]])]


def _load_gpu_image(camera: Camera, cache: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if camera.name not in cache:
        with Image.open(camera.path) as image:
            pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
        cache[camera.name] = pixels.permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)
    return cache[camera.name]


def rasterize_atlas(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray,
                    cameras: list[Camera], size: int, depth_tolerance: float,
                    views_per_face: int, device_name: str) -> tuple[Image.Image, Image.Image, float, int]:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    vertex_tensor = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    poses, intrinsics = _camera_tensors(cameras, device)
    atlas = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    source_images: set[str] = set()
    gpu_images: dict[str, torch.Tensor] = {}
    centers = torch.as_tensor(vertices[faces].mean(axis=1), dtype=torch.float32, device=device)
    center_pixels, center_depths = _project_fisheye(centers, poses, intrinsics)
    image_bounds = torch.tensor(
        [[camera.width, camera.height] for camera in cameras],
        dtype=torch.float32, device=device)
    valid_centers = ((center_depths > 0)
                     & (center_pixels[:, :, 0] >= 0)
                     & (center_pixels[:, :, 1] >= 0)
                     & (center_pixels[:, :, 0] < image_bounds[:, None, 0])
                     & (center_pixels[:, :, 1] < image_bounds[:, None, 1]))
    center_scores = torch.where(valid_centers, center_depths,
                                torch.full_like(center_depths, float("inf")))
    candidate_count = min(max(1, views_per_face), len(cameras))
    candidate_scores, candidate_ids = torch.topk(
        center_scores, candidate_count, largest=False, dim=0)
    candidate_ids = candidate_ids.T.cpu().numpy()
    candidate_ids[np.isinf(candidate_scores.T.cpu().numpy())] = -1

    atlas_indices = []
    world_points_all = []
    candidates_all = []
    for face_index, face in enumerate(faces):
        face_vertices = vertices[face]
        # Unwrap each face independently so the cylindrical seam cannot stretch it.
        face_uv = unwrap_face_u(uv[face])
        triangle = face_uv * (size - 1)
        minimum = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        maximum = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), size - 1)
        if np.any(minimum > maximum):
            continue
        xs, ys = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1),
            np.arange(minimum[1], maximum[1] + 1))
        sample_uv = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-8:
            continue
        wa = ((b[1] - c[1]) * (sample_uv[:, 0] - c[0]) + (c[0] - b[0]) * (sample_uv[:, 1] - c[1])) / denominator
        wb = ((c[1] - a[1]) * (sample_uv[:, 0] - c[0]) + (a[0] - c[0]) * (sample_uv[:, 1] - c[1])) / denominator
        inside = (wa >= 0) & (wb >= 0) & (wa + wb <= 1)
        if not np.any(inside):
            continue
        sample_uv, wa, wb = sample_uv[inside], wa[inside], wb[inside]
        world_points = (wa[:, None] * face_vertices[0] + wb[:, None] * face_vertices[1]
                        + (1 - wa - wb)[:, None] * face_vertices[2])
        atlas_indices.append(sample_uv.astype(np.int64))
        world_points_all.append(world_points)
        candidates_all.append(np.broadcast_to(candidate_ids[face_index],
                                               (len(world_points), candidate_count)))
    if world_points_all:
        world_points = torch.as_tensor(np.concatenate(world_points_all), dtype=torch.float32, device=device)
        sample_indices = np.concatenate(atlas_indices)
        candidates = torch.as_tensor(np.concatenate(candidates_all), dtype=torch.long, device=device)
        blended = torch.zeros((len(world_points), 3), device=device)
        total_weight = torch.zeros(len(world_points), device=device)
        for camera_index in torch.unique(candidates[candidates >= 0]).tolist():
            selector = (candidates == camera_index).any(dim=1)
            camera = cameras[camera_index]
            pixels, depths = _project_fisheye(
                world_points[selector], poses[camera_index:camera_index + 1],
                intrinsics[camera_index:camera_index + 1])
            pixels, depths = pixels[0], depths[0]
            width, height = camera.width, camera.height
            valid = ((depths > 0) & (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0)
                     & (pixels[:, 0] < width - 1) & (pixels[:, 1] < height - 1))
            if not torch.any(valid):
                continue
            selected_indices = torch.where(selector)[0][valid]
            grid = pixels[valid].clone()
            grid[:, 0] = grid[:, 0] / (width - 1) * 2 - 1
            grid[:, 1] = grid[:, 1] / (height - 1) * 2 - 1
            sampled = F.grid_sample(
                _load_gpu_image(camera, gpu_images, device), grid[None, :, None, :],
                mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0].T
            weight = 1.0 / depths[valid].clamp_min(1e-3)
            blended[selected_indices] += sampled * weight[:, None]
            total_weight[selected_indices] += weight
            source_images.add(camera.name)
        valid = total_weight > 0
        rgb = (blended[valid] / total_weight[valid, None] * 255).byte().cpu().numpy()
        pixel_indices = sample_indices[valid.cpu().numpy()]
        pixel_indices[:, 0] %= size
        write_y, write_x = pixel_indices[:, 1], pixel_indices[:, 0]
        unwritten = mask[write_y, write_x] == 0
        atlas[write_y[unwritten], write_x[unwritten]] = rgb[unwritten]
        mask[write_y[unwritten], write_x[unwritten]] = 255
    coverage = float(np.count_nonzero(mask)) / float(size * size)
    return Image.fromarray(atlas), Image.fromarray(mask), coverage, len(source_images)


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray,
               uv: np.ndarray, atlas: Image.Image) -> None:
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=np.asarray(atlas))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh.export(path, file_type="glb")


def process_tree(tree_id: int, points: np.ndarray, cameras: list[Camera], args: argparse.Namespace,
                 meshes_dir: Path, atlases_dir: Path, log: Callable[[str], None]) -> dict:
    log(f"  Tree {tree_id}: reconstructing mesh from {len(points):,} cropped points")
    vertices, faces = build_mesh(points, args.voxel_size, args.poisson_depth, args.target_triangles)
    log(f"  Tree {tree_id}: mesh has {len(vertices):,} vertices and {len(faces):,} triangles")
    uv = cylindrical_uv(vertices, args.vertical_axis)
    log(f"  Tree {tree_id}: projecting visible image texture into {args.atlas_size}x{args.atlas_size} atlas")
    try:
        atlas, mask, coverage, observations = rasterize_atlas(
            vertices, faces, uv, cameras, args.atlas_size, args.depth_tolerance,
            args.views_per_face, args.device)
        stem = f"tree_{tree_id:06d}"
        mesh_path = meshes_dir / f"{stem}.glb"
        atlas_path = atlases_dir / f"{stem}.png"
        mask_path = atlases_dir / f"{stem}_mask.png"
        atlas.save(atlas_path)
        mask.save(mask_path)
        write_mesh(mesh_path, vertices, faces, uv, atlas)
        log(f"  Tree {tree_id}: coverage {coverage:.1%}, {observations} source images, wrote {stem}.glb")
        return {
            "tree_id": tree_id,
            "mesh": str(mesh_path),
            "atlas": str(atlas_path),
            "mask": str(mask_path),
            "center": points.mean(axis=0).round(6).tolist(),
            "bounds": {"min": points.min(axis=0).round(6).tolist(), "max": points.max(axis=0).round(6).tolist()},
            "point_count": int(len(points)),
            "mesh_face_count": int(len(faces)),
            "texture_coverage": coverage,
            "source_image_count": observations,
            "quality_flags": [] if coverage > 0.01 else ["low_texture_coverage"],
        }
    finally:
        # Release references and cached blocks before the next large tree.
        gc.collect()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    log = make_logger(args.quiet)
    output = args.output
    meshes_dir, atlases_dir = output / "meshes", output / "atlases"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    atlases_dir.mkdir(parents=True, exist_ok=True)
    log("=== Create textured trunk atlases ===")
    log(f"Output directory: {output}")
    log(f"Settings: height={args.height:g}, voxel_size={args.voxel_size:g}, "
        f"poisson_depth={args.poisson_depth}, target_triangles={args.target_triangles}, "
        f"atlas_size={args.atlas_size}, views_per_face={args.views_per_face}, device={args.device}")
    cameras = load_cameras(args.colmap_model, args.images, log)
    trees = load_tree_points(args.segmented_cloud, args.vertical_axis, args.height, args.min_points, log)
    tree_items = sorted(trees.items())[:args.max_trees] if args.max_trees else sorted(trees.items())
    if args.max_trees:
        log(f"Processing pilot subset: {len(tree_items)} of {len(trees)} trees")
    else:
        log(f"Processing all {len(tree_items)} retained trees")
    records = []
    for index, (tree_id, points) in enumerate(tree_items, 1):
        log(f"[{index}/{len(tree_items)}] treeID={tree_id}, points={len(points):,}")
        try:
            records.append(process_tree(tree_id, points, cameras, args, meshes_dir, atlases_dir, log))
        except Exception as error:
            print(f"  Tree {tree_id}: skipped: {error}", file=sys.stderr)
    manifest = {
        "coordinate_system": "COLMAP reconstruction coordinates; Z is vertical",
        "source_cloud": str(args.segmented_cloud),
        "colmap_model": str(args.colmap_model),
        "image_root": str(args.images),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "output"
        },
        "trees": records,
    }
    with (output / "atlas_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    log(f"Wrote {len(records)} textured trees to {output}")
    if len(records) != len(tree_items):
        log(f"Skipped {len(tree_items) - len(records)} trees; see messages above")


if __name__ == "__main__":
    main()