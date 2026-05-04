# VN-ScaleConv

## Environment

Recommended environment:

```bash
conda create -n vnscale python=3.9 -y
conda activate vnscale
```

Install PyTorch and PyTorch3D with versions matching your CUDA environment, then install the remaining packages:

```bash
pip install open3d numpy scipy scikit-learn opencv-python matplotlib tqdm
```

## Dataset

Put the RGB-D PLY dataset under `data/`.

```text
data/
├── small/
│   ├── area_ground_truth.txt
│   ├── train/
│   ├── val/
│   └── test/
└── big/
    ├── area_ground_truth.txt
    ├── train/
    ├── val/
    └── test/
```

Each split directory contains leaf folders, and each leaf folder contains `.ply` files.

`area_ground_truth.txt` format:

```text
sample_name value
```

Example:

```text
1_small_516979647 1234.56
2_small_355761178 1456.78
```

For multi-trait regression, place the GT file at:

```text
phenotyping/data/final/gt_leaf_traits_6params_90.txt
```

## Train

Single-trait regression:

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

Multi-trait regression:

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

## Test

Single-trait regression:

```bash
python test_reg.py \
  --data_path data/small \
  --log_dir small_area_z \
  --rot z \
  --batch_size 64 \
  --gpu 0 \
  --label_unit mm2
```

Multi-trait regression:

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

## Measure 3 Traits

The measurement script outputs:

- `length_cm`
- `width_cm`
- `perimeter_cm`

Run:

```bash
python phenotyping/data/final/measure_small_ply_geometry.py \
  --dataset-root data/small \
  --output-dir phenotyping/data/final/small_geom_from_ply \
  --gt-txt phenotyping/data/final/gt_leaf_traits_6params_90.txt
```

Outputs:

- result table: `phenotyping/data/final/small_geom_from_ply/small_geom_measurements.txt`
- visualizations: `phenotyping/data/final/small_geom_from_ply/visual/`
