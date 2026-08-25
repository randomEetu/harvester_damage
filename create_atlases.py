"""Create cylinder-based trunk atlases from a TreeLearn-labeled cloud.

Pipeline summary:
1. Select trunk points per tree (height crop + radial branch suppression).
2. Fit a tilted cylinder to each tree.
3. Build an analytic cylinder mesh (no Poisson reconstruction).
4. Project source images onto tree points and blend per-point color.
5. Rasterize blended points into a rectangular cylinder atlas.
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
import pycolmap
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


@dataclass
class Camera:
    name: str
    path: Path
    width: int
    height: int
    center: np.ndarray
    colmap_image: object
    camera_model: object
    cam_from_world: np.ndarray


@dataclass
class TreeData:
    points: np.ndarray
    colors16: np.ndarray | None


@dataclass
class CylinderModel:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    t_min: float
    t_max: float
    e1: np.ndarray
    e2: np.ndarray
    seam_offset: float


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
    parser.add_argument("--min-points", type=int, default=500)
    parser.add_argument("--max-trees", type=int)

    parser.add_argument("--trunk-slice-height", type=float, default=0.25,
                        help="Vertical slice height for trunk radial filtering (default: 0.25 m).")
    parser.add_argument("--trunk-radius-quantile", type=float, default=0.7,
                        help="Radial quantile kept per slice to suppress branches (default: 0.7).")
    parser.add_argument("--trunk-radius-scale", type=float, default=1.2,
                        help="Multiplier on per-slice radial threshold (default: 1.2).")

    parser.add_argument("--atlas-size", type=int, default=1024)
    parser.add_argument("--inpaint-radius-px", type=int, default=18,
                        help="kNN inpainting radius in atlas pixels (default: 18).")

    parser.add_argument("--max-texture-points", type=int, default=70000,
                        help="Maximum points per tree used for camera texture projection.")
    parser.add_argument("--max-cameras-per-tree", type=int, default=24,
                        help="Nearest cameras tested for each tree.")
    parser.add_argument("--max-views-per-point", type=int, default=3,
                        help="Maximum blended camera observations per point.")
    parser.add_argument("--min-normal-dot", type=float, default=0.1,
                        help="Reject camera samples if view angle is too grazing.")
    parser.add_argument(
        "--texture-source",
        choices=("camera_cylinder", "camera_points", "point_rgb"),
        default="camera_cylinder",
        help=(
            "Texture source: project analytic cylinder texels from cameras, "
            "project observed point samples from cameras, or use LAS point RGB."
        ),
    )

    parser.add_argument("--cylinder-sides", type=int, default=96)
    parser.add_argument("--cylinder-rings", type=int, default=48)

    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def filter_trunk_points(points: np.ndarray, axis: int, slice_height: float,
                        radius_quantile: float, radius_scale: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, np.zeros(0, dtype=bool)
    other = [index for index in range(3) if index != axis]
    z = points[:, axis]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min < 1e-6:
        return points, np.ones(len(points), dtype=bool)

    bins = max(2, int(np.ceil((z_max - z_min) / max(slice_height, 1e-3))))
    edges = np.linspace(z_min, z_max, bins + 1)
    keep = np.zeros(len(points), dtype=bool)
    global_center = np.median(points[:, other], axis=0)

    for bin_index in range(bins):
        in_bin = (z >= edges[bin_index]) & (z <= edges[bin_index + 1] if bin_index == bins - 1 else z < edges[bin_index + 1])
        if not np.any(in_bin):
            continue
        slice_points = points[in_bin][:, other]
        center = np.median(slice_points, axis=0) if len(slice_points) >= 10 else global_center
        radii = np.linalg.norm(slice_points - center[None, :], axis=1)
        threshold = np.quantile(radii, np.clip(radius_quantile, 0.1, 0.98)) * max(radius_scale, 0.5)
        keep[in_bin] = radii <= threshold

    if keep.mean() < 0.2:
        keep[:] = True
    return points[keep], keep


def load_tree_points(path: Path, axis: int, height: float,
                     min_points: int, log: Callable[[str], None],
                     trunk_slice_height: float,
                     trunk_radius_quantile: float,
                     trunk_radius_scale: float) -> dict[int, TreeData]:
    log(f"Loading segmented point cloud: {path}")
    cloud = laspy.read(path)
    if "treeID" not in cloud.point_format.extra_dimension_names:
        raise ValueError("The segmented cloud has no treeID extra dimension")

    coordinates = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float32)
    has_rgb = all(name in cloud.point_format.dimension_names for name in ("red", "green", "blue"))
    colors16 = (
        np.column_stack((cloud.red, cloud.green, cloud.blue)).astype(np.float32)
        if has_rgb else None
    )
    labels = np.asarray(cloud["treeID"])
    label_ids = np.unique(labels)
    log(f"  Read {len(labels):,} points with {len(label_ids[label_ids > 0])} positive tree labels")

    trees: dict[int, TreeData] = {}
    discarded = 0
    retained_after_filter = 0

    for tree_id in label_ids:
        tree_id = int(tree_id)
        if tree_id <= 0:
            continue
        selected = labels == tree_id
        points = coordinates[selected]
        point_colors = colors16[selected] if colors16 is not None else None
        base = float(points[:, axis].min())
        trunk_mask = points[:, axis] <= base + height
        points = points[trunk_mask]
        if point_colors is not None:
            point_colors = point_colors[trunk_mask]
        points, keep = filter_trunk_points(
            points, axis, trunk_slice_height, trunk_radius_quantile, trunk_radius_scale)
        if point_colors is not None:
            point_colors = point_colors[keep]
        retained_after_filter += len(points)
        if len(points) >= min_points:
            trees[tree_id] = TreeData(points=points, colors16=point_colors)
        else:
            discarded += 1

    log(f"  Retained {len(trees)} trees after +{height:g} vertical crop; discarded {discarded} small trees")
    log(f"  Trunk filter kept {retained_after_filter:,} points across retained trees")
    return trees


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
        pose = np.asarray(colmap_image.cam_from_world().matrix(), dtype=np.float64)
        center = np.asarray(colmap_image.projection_center(), dtype=np.float32)
        camera_model = colmap_image.camera
        if camera_model is None:
            continue
        cameras.append(
            Camera(
                name=colmap_image.name,
                path=path,
                width=width,
                height=height,
                center=center,
                colmap_image=colmap_image,
                camera_model=camera_model,
                cam_from_world=pose,
            )
        )
    if not cameras:
        raise RuntimeError("No posed COLMAP images could be opened")
    log(f"  Loaded {len(cameras)} posed source images from {image_root}")
    return cameras


def orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1) + 1e-8
    e2 = np.cross(axis, e1)
    e2 /= np.linalg.norm(e2) + 1e-8
    return e1.astype(np.float32), e2.astype(np.float32)


def align_seam_offset(angles: np.ndarray, bins: int = 180) -> float:
    u = (angles / (2.0 * np.pi) + 0.5) % 1.0
    hist, edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
    seam_bin = int(np.argmin(hist))
    return float(0.5 * (edges[seam_bin] + edges[seam_bin + 1]))


def fit_cylinder(points: np.ndarray, vertical_axis: int = 2) -> CylinderModel:
    center = points.mean(axis=0)
    shifted = points - center[None, :]
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    axis = vh[0].astype(np.float32)
    axis /= np.linalg.norm(axis) + 1e-8
    # PCA axis sign is arbitrary. Keep all atlases oriented toward +vertical.
    if axis[vertical_axis] < 0:
        axis = -axis

    t = shifted @ axis
    proj = center[None, :] + t[:, None] * axis[None, :]
    radial = points - proj
    radius = float(np.median(np.linalg.norm(radial, axis=1)))

    e1, e2 = orthonormal_basis_from_axis(axis)
    x = radial @ e1
    y = radial @ e2
    angles = np.arctan2(y, x)
    seam_offset = align_seam_offset(angles)

    return CylinderModel(
        center=center.astype(np.float32),
        axis=axis,
        radius=max(radius, 1e-3),
        t_min=float(t.min()),
        t_max=float(t.max()),
        e1=e1,
        e2=e2,
        seam_offset=seam_offset,
    )


def points_to_cylinder_uv(points: np.ndarray, model: CylinderModel) -> tuple[np.ndarray, np.ndarray]:
    shifted = points - model.center[None, :]
    t = shifted @ model.axis
    radial = shifted - t[:, None] * model.axis[None, :]
    x = radial @ model.e1
    y = radial @ model.e2
    angle = np.arctan2(y, x)
    u = (angle / (2.0 * np.pi) + 0.5 - model.seam_offset) % 1.0
    v = (t - model.t_min) / max(model.t_max - model.t_min, 1e-6)
    return np.column_stack((u, v)).astype(np.float32), radial


def bilinear_sample_many(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinearly sample an RGB image at N floating-point pixel coordinates."""
    if len(xy) == 0:
        return np.empty((0, 3), dtype=np.float32)

    h, w, _ = image.shape
    x = np.clip(xy[:, 0], 0.0, max(w - 1.001, 0.0))
    y = np.clip(xy[:, 1], 0.0, max(h - 1.001, 0.0))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]

    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]
    return (
        (1.0 - wx) * (1.0 - wy) * c00
        + wx * (1.0 - wy) * c10
        + (1.0 - wx) * wy * c01
        + wx * wy * c11
    ).astype(np.float32)


def project_world_points(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project N world points with COLMAP's camera model.

    Returns Nx2 pixel coordinates in the *loaded image's* resolution and a
    validity mask. The explicit world->camera transform lets pycolmap project
    all points in one vectorized call instead of calling Image.project_point
    once per point.
    """
    points64 = np.asarray(points, dtype=np.float64)
    pose = camera.cam_from_world
    cam_points = points64 @ pose[:, :3].T + pose[:, 3][None, :]

    valid = np.isfinite(cam_points).all(axis=1) & (cam_points[:, 2] > 1e-8)
    xy = np.full((len(points64), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        projected = np.asarray(camera.camera_model.img_from_cam(cam_points[valid]), dtype=np.float64)
        if projected.ndim == 1:
            projected = projected[None, :]
        xy[valid] = projected

    model_width = max(float(camera.camera_model.width), 1.0)
    model_height = max(float(camera.camera_model.height), 1.0)
    xy[:, 0] *= camera.width / model_width
    xy[:, 1] *= camera.height / model_height

    valid &= np.isfinite(xy).all(axis=1)
    valid &= (
        (xy[:, 0] >= 0.0)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 0] < camera.width - 1)
        & (xy[:, 1] < camera.height - 1)
    )
    return xy, valid


def point_rgb_to_uint8(colors: np.ndarray) -> np.ndarray:
    """Convert LAS RGB to 0..255, supporting both 8-bit-like and 16-bit data."""
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size == 0:
        return colors.reshape(-1, 3)
    scale = 1.0 if float(np.nanmax(colors)) <= 255.0 else 257.0
    return np.clip(colors / scale, 0.0, 255.0).astype(np.float32)


def inpaint_atlas_periodic(atlas: np.ndarray, observed: np.ndarray,
                           radius_px: int) -> np.ndarray:
    """Fill nearby holes while treating the cylinder's left/right seam as periodic."""
    if radius_px <= 0 or not np.any(observed):
        return atlas

    size = atlas.shape[1]
    valid_coords = np.column_stack(np.where(observed))
    valid_colors = atlas[valid_coords[:, 0], valid_coords[:, 1]].astype(np.float32)
    empty_coords = np.column_stack(np.where(~observed))
    if len(empty_coords) == 0:
        return atlas

    # Duplicate observed samples across the horizontal wrap so pixels near u=0
    # can be filled from pixels near u=1 and vice versa.
    left = valid_coords.copy()
    left[:, 1] -= size
    right = valid_coords.copy()
    right[:, 1] += size
    tiled_coords = np.concatenate((left, valid_coords, right), axis=0)
    tiled_colors = np.concatenate((valid_colors, valid_colors, valid_colors), axis=0)

    tree = cKDTree(tiled_coords)
    k = min(4, len(tiled_coords))
    distances, nearest = tree.query(
        empty_coords, k=k, distance_upper_bound=radius_px
    )
    if distances.ndim == 1:
        distances = distances[:, None]
        nearest = nearest[:, None]

    fillable = np.isfinite(distances).any(axis=1)
    if not np.any(fillable):
        return atlas

    dst = empty_coords[fillable]
    nn_idx = nearest[fillable]
    nn_dist = distances[fillable]
    weights = 1.0 / np.clip(nn_dist, 1e-3, None)
    weights[~np.isfinite(nn_dist)] = 0.0

    safe_idx = np.clip(nn_idx, 0, len(tiled_colors) - 1)
    samples = tiled_colors[safe_idx]
    blended = (
        (samples * weights[:, :, None]).sum(axis=1)
        / np.clip(weights.sum(axis=1, keepdims=True), 1e-6, None)
    )
    atlas[dst[:, 0], dst[:, 1]] = np.clip(blended, 0, 255).astype(np.uint8)
    return atlas


def choose_candidate_cameras(tree_center: np.ndarray, cameras: list[Camera],
                             max_cameras: int) -> list[Camera]:
    ordered = sorted(cameras, key=lambda camera: float(np.linalg.norm(camera.center - tree_center)))
    return ordered[:max_cameras]


def blend_point_colors_from_cameras(points: np.ndarray, normals: np.ndarray,
                                    cameras: list[Camera], max_views_per_point: int,
                                    min_normal_dot: float) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """Blend camera colors onto observed 3D points, vectorized camera-by-camera."""
    n = len(points)
    accum = np.zeros((n, 3), dtype=np.float64)
    total_w = np.zeros(n, dtype=np.float64)
    view_count = np.zeros(n, dtype=np.uint16)
    used_camera_names: set[str] = set()

    for camera in cameras:
        active = view_count < max_views_per_point
        if not np.any(active):
            break

        active_idx = np.flatnonzero(active)
        active_points = points[active_idx]
        active_normals = normals[active_idx]

        view_vec = camera.center[None, :] - active_points
        dist = np.linalg.norm(view_vec, axis=1)
        good_dist = dist > 1e-6
        view_dir = view_vec / np.clip(dist[:, None], 1e-6, None)
        dot = np.einsum("ij,ij->i", active_normals, view_dir)
        facing = good_dist & (dot >= min_normal_dot)
        if not np.any(facing):
            continue

        facing_idx = active_idx[facing]
        xy, projected = project_world_points(camera, points[facing_idx])
        if not np.any(projected):
            continue

        sample_idx = facing_idx[projected]
        sample_xy = xy[projected]

        with Image.open(camera.path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
        rgb = bilinear_sample_many(image_array, sample_xy)

        sample_view = camera.center[None, :] - points[sample_idx]
        sample_dist = np.linalg.norm(sample_view, axis=1)
        sample_dir = sample_view / np.clip(sample_dist[:, None], 1e-6, None)
        sample_dot = np.einsum("ij,ij->i", normals[sample_idx], sample_dir)
        weight = sample_dot / np.clip(sample_dist, 1e-3, None)

        accum[sample_idx] += rgb * weight[:, None]
        total_w[sample_idx] += weight
        view_count[sample_idx] += 1
        used_camera_names.add(camera.name)

    used = total_w > 1e-8
    if not np.any(used):
        raise RuntimeError("No valid camera projections for this tree")

    colors = np.zeros((n, 3), dtype=np.float32)
    colors[used] = (accum[used] / total_w[used, None]).astype(np.float32)
    return colors, used, used_camera_names


def rasterize_point_atlas(uv: np.ndarray, rgb: np.ndarray, valid: np.ndarray,
                          size: int, inpaint_radius_px: int) -> tuple[Image.Image, Image.Image, float]:
    uv = uv[valid]
    rgb = rgb[valid]
    if len(uv) == 0:
        raise RuntimeError("No valid point colors available for atlas rasterization")

    # Horizontal coordinate is periodic. The -0.5 aligns samples with texel
    # centers used by render_cylinder_atlas_from_cameras.
    u = (uv[:, 0] % 1.0) * size - 0.5
    v = np.clip(uv[:, 1], 0.0, 1.0) * (size - 1)

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = x0 + 1
    y1 = np.minimum(y0 + 1, size - 1)
    wx = u - x0
    wy = v - y0

    accum = np.zeros((size * size, 3), dtype=np.float64)
    weight = np.zeros(size * size, dtype=np.float64)

    def splat(px: np.ndarray, py: np.ndarray, w: np.ndarray) -> None:
        px = px % size
        py = np.clip(py, 0, size - 1)
        flat = py * size + px
        np.add.at(weight, flat, w)
        np.add.at(accum[:, 0], flat, rgb[:, 0] * w)
        np.add.at(accum[:, 1], flat, rgb[:, 1] * w)
        np.add.at(accum[:, 2], flat, rgb[:, 2] * w)

    splat(x0, y0, (1 - wx) * (1 - wy))
    splat(x1, y0, wx * (1 - wy))
    splat(x0, y1, (1 - wx) * wy)
    splat(x1, y1, wx * wy)

    atlas = np.zeros((size * size, 3), dtype=np.uint8)
    observed_flat = weight > 1e-8
    atlas[observed_flat] = np.clip(
        accum[observed_flat] / weight[observed_flat, None], 0, 255
    ).astype(np.uint8)

    atlas = atlas.reshape(size, size, 3)
    observed = observed_flat.reshape(size, size)
    atlas = inpaint_atlas_periodic(atlas, observed, inpaint_radius_px)

    mask = (observed.astype(np.uint8) * 255)
    coverage = float(observed.mean())
    return Image.fromarray(atlas), Image.fromarray(mask), coverage


def render_cylinder_atlas_from_cameras(model: CylinderModel, cameras: list[Camera],
                                       size: int, max_views_per_texel: int,
                                       min_normal_dot: float,
                                       inpaint_radius_px: int) -> tuple[Image.Image, Image.Image, float, set[str]]:
    """Render the analytic cylinder into the source images and blend texel colors.

    The expensive dimension (all texels) is vectorized; Python only loops over
    candidate cameras.
    """
    u = (np.arange(size, dtype=np.float32) + 0.5) / size
    v = (np.arange(size, dtype=np.float32) + 0.5) / size
    uu, vv = np.meshgrid(u, v)

    theta = 2.0 * np.pi * ((uu + model.seam_offset) - 0.5)
    t = model.t_min + vv * (model.t_max - model.t_min)

    circle = (
        np.cos(theta)[..., None] * model.e1[None, None, :]
        + np.sin(theta)[..., None] * model.e2[None, None, :]
    )
    points = (
        model.center[None, None, :]
        + t[..., None] * model.axis[None, None, :]
        + model.radius * circle
    )

    flat_points = points.reshape(-1, 3).astype(np.float32, copy=False)
    flat_normals = circle.reshape(-1, 3).astype(np.float32, copy=False)

    n = len(flat_points)
    accum = np.zeros((n, 3), dtype=np.float64)
    total_w = np.zeros(n, dtype=np.float64)
    view_count = np.zeros(n, dtype=np.uint16)
    used_camera_names: set[str] = set()

    # VERY PRACTICAL DEBUGGING TEST
    p = points.reshape(-1, 3)
    print("CYLINDER:")
    print("min   ", p.min(axis=0))
    print("max   ", p.max(axis=0))
    print("center", p.mean(axis=0))

    centers = np.stack([c.center for c in cameras])
    print("\nCOLMAP CAMERA CENTERS:")
    print("min   ", centers.min(axis=0))
    print("max   ", centers.max(axis=0))
    print("mean  ", centers.mean(axis=0))

    for camera in cameras:
        active = view_count < max_views_per_texel
        if not np.any(active):
            break

        active_idx = np.flatnonzero(active)
        active_points = flat_points[active_idx]
        active_normals = flat_normals[active_idx]

        view_vec = camera.center[None, :] - active_points
        dist = np.linalg.norm(view_vec, axis=1)
        view_dir = view_vec / np.clip(dist[:, None], 1e-6, None)
        dot = np.einsum("ij,ij->i", active_normals, view_dir)
        facing = (dist > 1e-6) & (dot >= min_normal_dot)
        if not np.any(facing):
            continue

        facing_idx = active_idx[facing]
        xy, projected = project_world_points(camera, flat_points[facing_idx])
        if not np.any(projected):
            continue

        sample_idx = facing_idx[projected]
        sample_xy = xy[projected]

        with Image.open(camera.path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
        rgb = bilinear_sample_many(image_array, sample_xy)

        sample_view = camera.center[None, :] - flat_points[sample_idx]
        sample_dist = np.linalg.norm(sample_view, axis=1)
        sample_dir = sample_view / np.clip(sample_dist[:, None], 1e-6, None)
        sample_dot = np.einsum("ij,ij->i", flat_normals[sample_idx], sample_dir)
        weight = sample_dot / np.clip(sample_dist, 1e-3, None)

        accum[sample_idx] += rgb * weight[:, None]
        total_w[sample_idx] += weight
        view_count[sample_idx] += 1
        used_camera_names.add(camera.name)

    observed = total_w > 1e-8
    if not np.any(observed):
        raise RuntimeError("No valid camera projections for this cylinder")

    atlas_img = np.zeros((n, 3), dtype=np.uint8)
    atlas_img[observed] = np.clip(
        accum[observed] / total_w[observed, None], 0, 255
    ).astype(np.uint8)
    atlas_img = atlas_img.reshape(size, size, 3)

    observed_2d = observed.reshape(size, size)
    atlas_img = inpaint_atlas_periodic(
        atlas_img, observed_2d, inpaint_radius_px
    )
    mask_img = observed_2d.astype(np.uint8) * 255

    coverage = float(observed.mean())
    return (
        Image.fromarray(atlas_img),
        Image.fromarray(mask_img),
        coverage,
        used_camera_names,
    )


def build_cylinder_mesh(model: CylinderModel, sides: int, rings: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    uvs = []
    for ring in range(rings + 1):
        v = ring / max(rings, 1)
        t = model.t_min + v * (model.t_max - model.t_min)
        axis_point = model.center + t * model.axis
        for side in range(sides + 1):
            u = side / max(sides, 1)
            theta = 2.0 * np.pi * ((u + model.seam_offset) - 0.5)
            circle = model.radius * (np.cos(theta) * model.e1 + np.sin(theta) * model.e2)
            vertices.append(axis_point + circle)
            uvs.append([u, v])

    vertices = np.asarray(vertices, dtype=np.float32)
    uvs = np.asarray(uvs, dtype=np.float32)

    faces = []
    stride = sides + 1
    for ring in range(rings):
        row0 = ring * stride
        row1 = (ring + 1) * stride
        for side in range(sides):
            a = row0 + side
            b = row0 + side + 1
            c = row1 + side
            d = row1 + side + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return vertices, np.asarray(faces, dtype=np.int32), uvs


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray,
               uv: np.ndarray, atlas: Image.Image) -> None:
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=np.asarray(atlas))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh.export(path, file_type="glb")


def process_tree(tree_id: int, tree_data: TreeData,
                 cameras: list[Camera], args: argparse.Namespace,
                 meshes_dir: Path, atlases_dir: Path,
                 log: Callable[[str], None]) -> dict:
    points = tree_data.points
    log(f"  Tree {tree_id}: fitting tilted cylinder to {len(points):,} points")
    cylinder = fit_cylinder(points, args.vertical_axis)

    tree_center = points.mean(axis=0)
    candidates = choose_candidate_cameras(
        tree_center, cameras, args.max_cameras_per_tree
    )

    # DEBUG: project red points onto images and save them
    camera = candidates[0]

    img = np.asarray(
        Image.open(camera.path).convert("RGB")
    ).copy()

    sample_points = points[::20]

    projected = 0

    for point in sample_points:
        uv = camera.colmap_image.project_point(point.astype(np.float64))
        if uv is None:
            continue

        x = int(round(float(uv[0])))
        y = int(round(float(uv[1])))

        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            r = 3
            img[
                max(0, y-r):min(img.shape[0], y+r+1),
                max(0, x-r):min(img.shape[1], x+r+1)
            ] = [255, 0, 0]

            projected += 1

    Image.fromarray(img).save(
        args.output / f"debug_projection_tree_{tree_id}.jpg"
    )

    print(
        f"Projected {projected}/{len(sample_points)} "
        f"TreeLearn points into {camera.name}"
    )
    # DEBUG ENDS

    if args.texture_source == "camera_cylinder":
        log(
            f"  Tree {tree_id}: projecting cylinder texels from "
            f"{len(candidates)} nearest cameras"
        )
        atlas, mask, coverage, used_cameras = render_cylinder_atlas_from_cameras(
            cylinder,
            candidates,
            args.atlas_size,
            args.max_views_per_point,
            args.min_normal_dot,
            args.inpaint_radius_px,
        )
        texture_points_used = args.atlas_size * args.atlas_size

    else:
        uv, radial = points_to_cylinder_uv(points, cylinder)
        normals = radial / np.clip(
            np.linalg.norm(radial, axis=1, keepdims=True), 1e-6, None
        )

        if len(points) > args.max_texture_points:
            rng = np.random.default_rng(tree_id)
            indices = rng.choice(
                len(points), size=args.max_texture_points, replace=False
            )
        else:
            indices = np.arange(len(points))

        texture_points = points[indices]
        texture_uv = uv[indices]
        texture_normals = normals[indices]

        if args.texture_source == "point_rgb":
            if tree_data.colors16 is None:
                raise RuntimeError(
                    "texture-source=point_rgb requested, but the LAS/LAZ has no RGB fields"
                )
            point_rgb8 = point_rgb_to_uint8(tree_data.colors16[indices])
            used_mask = np.ones(len(indices), dtype=bool)
            used_cameras: set[str] = set()
            log(
                f"  Tree {tree_id}: rasterizing {len(indices):,} LAS RGB point samples"
            )
        else:
            log(
                f"  Tree {tree_id}: projecting {len(indices):,} point samples "
                f"from {len(candidates)} nearest cameras"
            )
            try:
                point_rgb8, used_mask, used_cameras = blend_point_colors_from_cameras(
                    texture_points,
                    texture_normals,
                    candidates,
                    args.max_views_per_point,
                    args.min_normal_dot,
                )
            except RuntimeError:
                if tree_data.colors16 is None:
                    raise
                log(
                    f"  Tree {tree_id}: no valid camera projections; "
                    "falling back to LAS RGB"
                )
                point_rgb8 = point_rgb_to_uint8(tree_data.colors16[indices])
                used_mask = np.ones(len(indices), dtype=bool)
                used_cameras = set()

        atlas, mask, coverage = rasterize_point_atlas(
            texture_uv,
            point_rgb8,
            used_mask,
            args.atlas_size,
            args.inpaint_radius_px,
        )
        texture_points_used = len(indices)

    mesh_vertices, mesh_faces, mesh_uv = build_cylinder_mesh(
        cylinder, args.cylinder_sides, args.cylinder_rings)

    stem = f"tree_{tree_id:06d}"
    mesh_path = meshes_dir / f"{stem}.glb"
    atlas_path = atlases_dir / f"{stem}.png"
    mask_path = atlases_dir / f"{stem}_mask.png"

    atlas.save(atlas_path)
    mask.save(mask_path)
    write_mesh(mesh_path, mesh_vertices, mesh_faces, mesh_uv, atlas)

    log(
        f"  Tree {tree_id}: coverage {coverage:.1%}, "
        f"{len(used_cameras)} source images, radius={cylinder.radius:.3f} m")

    return {
        "tree_id": tree_id,
        "mesh": str(mesh_path),
        "atlas": str(atlas_path),
        "mask": str(mask_path),
        "center": points.mean(axis=0).round(6).tolist(),
        "axis": cylinder.axis.round(6).tolist(),
        "radius": float(cylinder.radius),
        "t_range": [float(cylinder.t_min), float(cylinder.t_max)],
        "point_count": int(len(points)),
        "texture_points": int(texture_points_used),
        "mesh_face_count": int(len(mesh_faces)),
        "texture_coverage": coverage,
        "source_image_count": int(len(used_cameras)),
        "quality_flags": [] if coverage > 0.1 else ["low_texture_coverage"],
    }


def main() -> None:
    args = parse_args()
    log = make_logger(args.quiet)

    output = args.output
    meshes_dir, atlases_dir = output / "meshes", output / "atlases"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    atlases_dir.mkdir(parents=True, exist_ok=True)

    log("=== Create cylinder trunk atlases ===")
    log(f"Output directory: {output}")
    log(
        f"Settings: height={args.height:g}, atlas_size={args.atlas_size}, "
        f"texture_source={args.texture_source}, max_cameras_per_tree={args.max_cameras_per_tree}")

    trees = load_tree_points(
        args.segmented_cloud, args.vertical_axis, args.height, args.min_points, log,
        args.trunk_slice_height, args.trunk_radius_quantile, args.trunk_radius_scale)

    cameras = load_cameras(args.colmap_model, args.images, log)

    tree_items = sorted(trees.items())[:args.max_trees] if args.max_trees else sorted(trees.items())
    if args.max_trees:
        log(f"Processing pilot subset: {len(tree_items)} of {len(trees)} trees")
    else:
        log(f"Processing all {len(tree_items)} retained trees")

    records = []
    for index, (tree_id, tree_data) in enumerate(tree_items, 1):
        log(f"[{index}/{len(tree_items)}] treeID={tree_id}, points={len(tree_data.points):,}")
        try:
            records.append(process_tree(
                tree_id, tree_data,
                cameras, args, meshes_dir, atlases_dir, log))
        except Exception as error:  # noqa: BLE001
            print(f"  Tree {tree_id}: skipped: {error}", file=sys.stderr)
        finally:
            gc.collect()

    manifest = {
        "coordinate_system": f"COLMAP reconstruction coordinates; axis {args.vertical_axis} treated as vertical",
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