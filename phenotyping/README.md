# Phenotyping Scripts

This directory keeps the leaf phenotyping pipeline that was used to build the final ground-truth tables and geometry baselines.

## 2D Scan GT Pipeline

- `phenotyping/data/measure_2d_scan_traits.py`
  Measures `length_cm`, `width_cm`, `perimeter_cm`, and an initial apex-angle estimate from scanned PDF leaves.
- `phenotyping/data/visualize_2d_scan_traits.py`
  Renders annotated PNGs for the 2D scan measurements.
- `phenotyping/data/build_final_gt_dataset.py`
  Merges scan traits, area GT, and vein-length annotations into the 5-trait GT table.
- `phenotyping/data/generate_final_apex_candidates.py`
  Produces two apex-angle candidates per scanned leaf for manual 1/2 selection.
- `phenotyping/data/finalize_leaf_traits_with_apex.py`
  Merges the manually selected apex angle into the final 6-trait GT table.

## Small RGB-D Geometry Baseline

- `phenotyping/data/final/measure_small_ply_geometry.py`
  Measures `length_cm`, `width_cm`, and `perimeter_cm` directly from small-dataset PLY point clouds and exports visualization images.

## Notes

- Large intermediate data, rendered images, and datasets are intentionally ignored by git.
- Older one-off scripts that were superseded by the files above were removed during cleanup.
- Broken historical experiment entrypoints that depended on missing modules were also removed.
