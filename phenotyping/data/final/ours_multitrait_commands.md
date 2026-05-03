# Ours Small Single-Trait Commands

当前版本按你的最新要求整理：

- 只跑 `small` RGB-D 点云数据集
- 只用有 GT 的叶片 `1-90`
- 训练集仍然用原始 `train`
- 原始 `val + test` 合并成一个验证集
- 不做 label normalization
- 不做 6 指标联合回归
- `area_cm2` 不重跑，沿用你之前 area 的实验
- 这次只分别跑 5 个单指标：
  - `length_cm`
  - `width_cm`
  - `perimeter_cm`
  - `vein_length_cm`
  - `apex_angle_deg`

输出目录统一写到：

```bash
phenotyping/data/final/ours_small_singletrait
```

Python：

```bash
/home/asus/miniconda3/envs/vnn/bin/python
```

## Train Commands

### length_cm

```bash
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_length_mergeval_train_z --output_root phenotyping/data/final/ours_small_singletrait --rot z --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits length_cm
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_length_mergeval_train_so3 --output_root phenotyping/data/final/ours_small_singletrait --rot so3 --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits length_cm
```

### width_cm

```bash
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_width_mergeval_train_z --output_root phenotyping/data/final/ours_small_singletrait --rot z --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits width_cm
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_width_mergeval_train_so3 --output_root phenotyping/data/final/ours_small_singletrait --rot so3 --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits width_cm
```

### perimeter_cm

```bash
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_perimeter_mergeval_train_z --output_root phenotyping/data/final/ours_small_singletrait --rot z --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits perimeter_cm
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_perimeter_mergeval_train_so3 --output_root phenotyping/data/final/ours_small_singletrait --rot so3 --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits perimeter_cm
```

### vein_length_cm

```bash
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_vein_mergeval_train_z --output_root phenotyping/data/final/ours_small_singletrait --rot z --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits vein_length_cm
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_vein_mergeval_train_so3 --output_root phenotyping/data/final/ours_small_singletrait --rot so3 --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits vein_length_cm
```

### apex_angle_deg

```bash
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_apex_mergeval_train_z --output_root phenotyping/data/final/ours_small_singletrait --rot z --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits apex_angle_deg
/home/asus/miniconda3/envs/vnn/bin/python train_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --log_dir small_apex_mergeval_train_so3 --output_root phenotyping/data/final/ours_small_singletrait --rot so3 --n_knn 10 --batch_size 64 --epoch 300 --learning_rate 0.01 --optimizer Adam --gpu 0 --merge_val_test --target_traits apex_angle_deg
```

## Validation Commands

说明：

- 这里没有独立 test split
- 所有验证命令都在合并后的 `val + test` 上跑
- 三种测试条件还是保留：
  - `z/z`
  - `z/so3`
  - `so3/so3`

### length_cm

```bash
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_length_mergeval_train_z --rot z --result_tag zz_valmerge --merge_val_test --target_traits length_cm
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_length_mergeval_train_z --rot so3 --result_tag zso3_valmerge --merge_val_test --target_traits length_cm
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_length_mergeval_train_so3 --rot so3 --result_tag so3so3_valmerge --merge_val_test --target_traits length_cm
```

### width_cm

```bash
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_width_mergeval_train_z --rot z --result_tag zz_valmerge --merge_val_test --target_traits width_cm
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_width_mergeval_train_z --rot so3 --result_tag zso3_valmerge --merge_val_test --target_traits width_cm
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_width_mergeval_train_so3 --rot so3 --result_tag so3so3_valmerge --merge_val_test --target_traits width_cm
```

### perimeter_cm

```bash
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_perimeter_mergeval_train_z --rot z --result_tag zz_valmerge --merge_val_test --target_traits perimeter_cm
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_perimeter_mergeval_train_z --rot so3 --result_tag zso3_valmerge --merge_val_test --target_traits perimeter_cm
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_perimeter_mergeval_train_so3 --rot so3 --result_tag so3so3_valmerge --merge_val_test --target_traits perimeter_cm
```

### vein_length_cm

```bash
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_vein_mergeval_train_z --rot z --result_tag zz_valmerge --merge_val_test --target_traits vein_length_cm
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_vein_mergeval_train_z --rot so3 --result_tag zso3_valmerge --merge_val_test --target_traits vein_length_cm
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_vein_mergeval_train_so3 --rot so3 --result_tag so3so3_valmerge --merge_val_test --target_traits vein_length_cm
```

### apex_angle_deg

```bash
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_apex_mergeval_train_z --rot z --result_tag zz_valmerge --merge_val_test --target_traits apex_angle_deg
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_apex_mergeval_train_z --rot so3 --result_tag zso3_valmerge --merge_val_test --target_traits apex_angle_deg
/home/asus/miniconda3/envs/vnn/bin/python test_reg_multitrait.py --model vn_dgcnn_chidu_multitrait --data_path data/small --gt_txt phenotyping/data/final/gt_leaf_traits_6params_90.txt --experiment_dir phenotyping/data/final/ours_small_singletrait/small_apex_mergeval_train_so3 --rot so3 --result_tag so3so3_valmerge --merge_val_test --target_traits apex_angle_deg
```
