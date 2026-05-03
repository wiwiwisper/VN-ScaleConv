# VN-ScaleConv

Point-cloud regression code for leaf trait prediction, plus scripts for building leaf phenotyping labels from scanned PDFs and RGB-D PLY files.

## Environment

- Python 3.9+
- PyTorch
- PyTorch3D
- Open3D
- NumPy
- SciPy
- scikit-learn
- OpenCV
- matplotlib
- tqdm

For the PDF-based phenotyping scripts, install Poppler tools so that `pdfinfo` and `pdftoppm` are available in `PATH`.

## Repository Layout

```text
.
├── data_utils/
├── models/
├── phenotyping/
├── train_reg_vndgcnn_chidu.py
├── test_reg.py
├── train_reg_multitrait.py
└── test_reg_multitrait.py
```

## Dataset Placement

### 1. RGB-D point cloud dataset

Put the RGB-D PLY dataset under `data/`.

Expected structure:

```text
data/
├── small/
│   ├── area_ground_truth.txt
│   ├── train/
│   │   ├── 1/
│   │   ├── 2/
│   │   └── ...
│   ├── val/
│   │   ├── 1/
│   │   ├── 2/
│   │   └── ...
│   └── test/
│       ├── 1/
│       ├── 2/
│       └── ...
└── big/
    ├── area_ground_truth.txt
    ├── train/
    ├── val/
    └── test/
```

Notes:

- `area_ground_truth.txt` is required by `data_utils/LeafDataLoader.py`.
- Each line in `area_ground_truth.txt` should be:

```text
sample_name value
```

Example:

```text
1_small_516979647 1234.56
2_small_355761178 1456.78
```

### 2. Multi-trait GT table

For multi-trait training and evaluation, place the GT file at:

```text
phenotyping/data/final/gt_leaf_traits_6params_90.txt
```

Expected header:

```text
leaf_id    pdf_name    length_cm    width_cm    perimeter_cm    vein_length_cm    area_cm2    apex_angle_deg
```

### 3. 2D scanned leaf dataset

For the PDF phenotyping scripts, place the scanned leaves here:

```text
phenotyping/data/2dsaomiao/
```

Expected files:

```text
phenotyping/data/2dsaomiao/
├── 1.pdf
├── 2.pdf
├── 3.pdf
└── ...
```

If you build the final GT table with area and vein annotations, the default script paths are:

```text
phenotyping/data/3dgt_gt_leaf_with_lpssmooth_100_02/area_results_with_smooth.txt
phenotyping/data/yemai_length/2dsaomiao_png(1).zip
```

## Training And Testing

### 1. Single-trait regression

Train:

```bash
python train_reg_vndgcnn_chidu.py \
  --data_path data/small \
  --log_dir small_area_z \
  --rot z \
  --batch_size 64 \
  --epoch 200 \
  --learning_rate 0.01 \
  --optimizer Adam \
  --gpu 0
```

Test:

```bash
python test_reg.py \
  --data_path data/small \
  --log_dir small_area_z \
  --rot z \
  --batch_size 64 \
  --gpu 0 \
  --label_unit mm2
```

Outputs:

- checkpoints: `log/reg/<log_dir>/checkpoints/`
- training log: `log/reg/<log_dir>/logs/`
- test log: `log/reg/<log_dir>/eval.txt`

### 2. Multi-trait regression

Train all six traits:

```bash
python train_reg_multitrait.py \
  --data_path data/small \
  --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt \
  --log_dir small_multitrait_z \
  --output_root log/reg_multitrait \
  --rot z \
  --n_knn 10 \
  --batch_size 64 \
  --epoch 300 \
  --learning_rate 0.01 \
  --optimizer Adam \
  --gpu 0
```

Train selected traits only:

```bash
python train_reg_multitrait.py \
  --data_path data/small \
  --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt \
  --log_dir small_length_width_z \
  --output_root log/reg_multitrait \
  --rot z \
  --target_traits length_cm,width_cm \
  --batch_size 64 \
  --epoch 300 \
  --gpu 0
```

If you want to merge `val` and `test` and use them as one validation split:

```bash
python train_reg_multitrait.py \
  --data_path data/small \
  --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt \
  --log_dir small_multitrait_mergeval_z \
  --output_root log/reg_multitrait \
  --rot z \
  --batch_size 64 \
  --epoch 300 \
  --gpu 0 \
  --merge_val_test
```

Test:

```bash
python test_reg_multitrait.py \
  --data_path data/small \
  --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt \
  --experiment_dir log/reg_multitrait/small_multitrait_z \
  --result_tag test_z \
  --rot z \
  --batch_size 64 \
  --gpu 0
```

Test selected traits:

```bash
python test_reg_multitrait.py \
  --data_path data/small \
  --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt \
  --experiment_dir log/reg_multitrait/small_length_width_z \
  --result_tag test_z \
  --rot z \
  --target_traits length_cm,width_cm \
  --batch_size 64 \
  --gpu 0
```

Outputs:

- checkpoints: `<output_root>/<log_dir>/checkpoints/`
- training logs: `<output_root>/<log_dir>/logs/`
- evaluation tables: `<output_root>/<log_dir>/eval_outputs/`

## Phenotyping Scripts

### 1. Measure 2D scan traits from PDFs

```bash
python phenotyping/data/measure_2d_scan_traits.py \
  --input-dir phenotyping/data/2dsaomiao \
  --output-txt phenotyping/data/scan2d_leaf_traits.txt \
  --render-dpi 200
```

### 2. Generate 2D measurement visualizations

```bash
python phenotyping/data/visualize_2d_scan_traits.py \
  --input-dir phenotyping/data/2dsaomiao \
  --output-dir phenotyping/data/2dsaomiao_measure \
  --render-dpi 200
```

### 3. Build 5-trait GT table

```bash
python phenotyping/data/build_final_gt_dataset.py \
  --scan-traits-txt phenotyping/data/scan2d_leaf_traits.txt \
  --area-txt phenotyping/data/3dgt_gt_leaf_with_lpssmooth_100_02/area_results_with_smooth.txt \
  --vein-zip phenotyping/data/yemai_length/2dsaomiao_png(1).zip \
  --pdf-dir phenotyping/data/2dsaomiao \
  --output-txt phenotyping/data/final/gt_leaf_traits_5params.txt \
  --visual-dir phenotyping/data/final/visual
```

### 4. Prepare apex-angle review images

```bash
python phenotyping/data/prepare_apex_angle_review.py \
  --input-dir phenotyping/data/2dsaomiao \
  --output-dir phenotyping/data/final/apex_review \
  --output-txt phenotyping/data/final/apex_angle_review.txt \
  --render-dpi 200
```

After running this command, set the `selection` column in `phenotyping/data/final/apex_angle_review.txt`.

### 5. Finalize the 6-trait GT table

```bash
python phenotyping/data/finalize_leaf_traits_with_apex.py \
  --gt-txt phenotyping/data/final/gt_leaf_traits_5params.txt \
  --apex-choice-txt phenotyping/data/final/apex_angle_review.txt \
  --pdf-dir phenotyping/data/2dsaomiao \
  --vein-zip phenotyping/data/yemai_length/2dsaomiao_png(1).zip \
  --output-txt phenotyping/data/final/gt_leaf_traits_6params_90.txt \
  --visual-dir phenotyping/data/final/visual
```

### 6. Measure geometry from small PLY point clouds

```bash
python phenotyping/data/final/measure_small_ply_geometry.py \
  --dataset-root data/small \
  --output-dir phenotyping/data/final/small_geom_from_ply \
  --gt-txt phenotyping/data/final/gt_leaf_traits_6params_90.txt
```

## Usage Notes

- `train_reg_vndgcnn_chidu.py` is the single-target regression entry point.
- `train_reg_multitrait.py` is the multi-target regression entry point.
- `test_reg.py` reads checkpoints from `log/reg/<log_dir>/checkpoints/best_model.pth`.
- `test_reg_multitrait.py` reads checkpoints from `--experiment_dir/checkpoints/best_model.pth`.
- The data loaders cache sampled point clouds under `data/<dataset>/processed/` or `data/<dataset>/processed_multitrait/`.
- If `val/` is missing for single-trait training, the script falls back to `test/` as validation.
- For multi-trait runs with `--merge_val_test`, the loader merges `val/` and `test/` into one evaluation split.
