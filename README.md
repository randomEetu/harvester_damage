# harvester_damage

## Create textured trunk atlases

The A-stage pipeline reads the propagated labels, reconstructs one decimated
mesh per tree, projects texture from the posed COLMAP images, and writes one
atlas and mask per tree together with `atlas_manifest.json`.

Run a small pilot first:

```bash
conda activate harvester_env

python create_atlases.py \
	data/segmented.laz \
	data/forest/sparse_final \
	data/forest/images \
	output/atlases \
	--max-trees 5
```

The default crop retains points from each tree's lowest point through 5 m in
the +Z direction. Useful controls include `--height`, `--voxel-size`,
`--poisson-depth`, `--target-triangles`, `--atlas-size`, and `--min-points`.
Progress messages are enabled by default; pass `--quiet` when an orchestrating
`main.py` should suppress them.

Texture modes:

- `--texture-source point_rgb` (default): builds cylindrical atlases directly
	from the RGB values stored in `segmented.laz`. This is faster and currently
	produces more stable bark atlases.
- `--texture-source camera`: projects textures from COLMAP images using
	fisheye camera models and GPU blending.

Point-RGB mode knobs:

- `--trunk-slice-height`, `--trunk-radius-quantile`, `--trunk-radius-scale`:
	suppress branches/foliage before meshing and texturing.
- `--inpaint-radius-px`: fills empty atlas texels from nearby observed texels.
	The atlas mask still marks originally observed texels.

Camera mode knobs:

- `--views-per-face` and `--device` (CUDA by default).

Outputs are written under the selected output directory:

```text
meshes/tree_000001.glb
atlases/tree_000001.png
atlases/tree_000001_mask.png
atlas_manifest.json
```