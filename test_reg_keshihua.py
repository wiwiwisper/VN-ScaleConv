"""
Author: Modified from test_reg.py
Date: 2025
Description: Regression test script with visualization
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))

# 导入自定义模块
from data_utils.LeafDataLoader import LeaveDataLoader

# 支持的模型列表
MODEL_REGISTRY = {
    'pointnet_reg': ('pointnet_reg', 'get_model'),
    'vn_dgcnn_reg': ('vn_dgcnn_reg', 'get_model'),
    'dgcnn_reg': ('dgcnn_reg', 'get_model'),
    'vn_dgcnn_se_reg': ('vn_dgcnn_se_reg', 'get_model'),
    'vn_dgcnn_chidu_reg': ('vn_dgcnn_chidu_reg', 'get_model'),
}

def parse_args():
    parser = argparse.ArgumentParser('PointNet Regression Test with Visualization')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size [default: 16]')
    parser.add_argument('--model', default='pointnet_reg', help='Model name [default: pointnet_reg]',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet_reg', help='Log directory name [default: pointnet_reg]')
    parser.add_argument('--rot', type=str, default='aligned', help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--feature_transform', action='store_true', default=False,
                        help='Use feature transform in PointNet [default: False]')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root folder')
    return parser.parse_args()


def apply_rotation(points, rot_mode):
    """Apply rotation augmentation to the points."""
    if rot_mode == 'z':
        angle = torch.rand(points.shape[0], device=points.device) * 360
        trot = RotateAxisAngle(angle=angle, axis="Z", degrees=True).to(points.device)
    elif rot_mode == 'so3':
        rotations = random_rotations(points.shape[0]).to(points.device)
        trot = Rotate(R=rotations).to(points.device)
    else:
        return points  # no rotation

    points = trot.transform_points(points)
    return points


def load_model_and_loss(model_name):
    import importlib

    module_name, class_name = MODEL_REGISTRY[model_name]
    model_module = importlib.import_module(f'models.{module_name}')
    model_class = getattr(model_module, class_name)

    return model_class


def test_and_visualize(model, loader, args, save_dir):
    """测试模型并生成可视化图表"""
    model.eval()
    preds_all = []
    labels_all = []

    print('Testing and collecting predictions...')
    with torch.no_grad():
        for points, label in tqdm(loader, desc='Testing'):
            points = points.float().cuda()
            label = label.float().cuda()

            # Apply rotation (if any)
            if args.rot != 'aligned':
                points = apply_rotation(points, args.rot)

            points = points.transpose(2, 1)  # [B, 3, N]

            pred, _ = model(points)

            # Save predictions and labels
            preds_all.append(pred.squeeze().cpu().numpy())
            labels_all.append(label.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    # 计算误差
    errors = preds_all - labels_all
    abs_errors = np.abs(errors)
    rel_errors = abs_errors / (labels_all + 1e-8) * 100  # 相对误差（百分比）

    # 计算评估指标
    mse = np.mean(errors ** 2)
    mae = np.mean(abs_errors)
    mape = np.mean(rel_errors)
    from sklearn.metrics import r2_score
    r2 = r2_score(labels_all, preds_all)

    print(f'\nTest Results:')
    print(f'MSE: {mse:.6f}')
    print(f'MAE: {mae:.6f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'R² Score: {r2:.6f}')

    # ========== 开始可视化 ==========
    plt.rcParams['font.size'] = 12

    # 图1: GT vs Pred 对比图（按样本索引）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 样本序列图
    ax1 = axes[0, 0]
    indices = np.arange(len(labels_all))
    ax1.scatter(indices, labels_all, c='red', alpha=0.6, s=30, label='Ground Truth', marker='o')
    ax1.scatter(indices, preds_all, c='blue', alpha=0.6, s=30, label='Prediction', marker='x')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Area Value')
    ax1.set_title('Ground Truth vs Prediction (by Sample)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 散点图 (GT vs Pred)
    ax2 = axes[0, 1]
    ax2.scatter(labels_all, preds_all, c='blue', alpha=0.5, s=30)
    # 画对角线（完美预测线）
    min_val = min(labels_all.min(), preds_all.min())
    max_val = max(labels_all.max(), preds_all.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Ground Truth')
    ax2.set_ylabel('Prediction')
    ax2.set_title(f'Prediction vs Ground Truth\nR²={r2:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # 子图3: 误差分布图（按GT值排序）
    ax3 = axes[1, 0]
    sorted_indices = np.argsort(labels_all)
    ax3.scatter(labels_all[sorted_indices], errors[sorted_indices], c='green', alpha=0.5, s=30)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Ground Truth (sorted)')
    ax3.set_ylabel('Error (Pred - GT)')
    ax3.set_title('Error Distribution by Ground Truth Value')
    ax3.grid(True, alpha=0.3)

    # 子图4: 相对误差分布
    ax4 = axes[1, 1]
    ax4.hist(rel_errors, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(x=mape, color='r', linestyle='--', linewidth=2, label=f'Mean={mape:.2f}%')
    ax4.set_xlabel('Relative Error (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Relative Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    vis_path = os.path.join(save_dir, f'visualization_{args.rot}.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f'Visualization saved to: {vis_path}')
    plt.close()

    # ========== 图2: 详细误差分析 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 绝对误差 vs GT
    ax1 = axes[0]
    ax1.scatter(labels_all, abs_errors, c='purple', alpha=0.5, s=30)
    ax1.set_xlabel('Ground Truth')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title(f'Absolute Error vs GT\nMAE={mae:.4f}')
    ax1.grid(True, alpha=0.3)

    # 相对误差 vs GT
    ax2 = axes[1]
    ax2.scatter(labels_all, rel_errors, c='brown', alpha=0.5, s=30)
    ax2.axhline(y=mape, color='r', linestyle='--', linewidth=2, label=f'Mean={mape:.2f}%')
    ax2.set_xlabel('Ground Truth')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Relative Error vs GT')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 误差直方图
    ax3 = axes[2]
    ax3.hist(errors, bins=50, color='teal', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(errors), color='orange', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(errors):.4f}')
    ax3.set_xlabel('Error (Pred - GT)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    error_path = os.path.join(save_dir, f'error_analysis_{args.rot}.png')
    plt.savefig(error_path, dpi=150, bbox_inches='tight')
    print(f'Error analysis saved to: {error_path}')
    plt.close()

    # ========== 保存数值结果到txt ==========
    results_path = os.path.join(save_dir, f'results_{args.rot}.txt')
    with open(results_path, 'w') as f:
        f.write(f'Test Results for {args.log_dir} with rotation {args.rot}\n')
        f.write(f'=' * 60 + '\n')
        f.write(f'MSE: {mse:.6f}\n')
        f.write(f'MAE: {mae:.6f}\n')
        f.write(f'MAPE: {mape:.2f}%\n')
        f.write(f'R² Score: {r2:.6f}\n')
        f.write(f'Min Error: {errors.min():.6f}\n')
        f.write(f'Max Error: {errors.max():.6f}\n')
        f.write(f'Std Error: {errors.std():.6f}\n')
        f.write(f'\nNumber of samples: {len(labels_all)}\n')
        f.write(f'GT range: [{labels_all.min():.4f}, {labels_all.max():.4f}]\n')
        f.write(f'Pred range: [{preds_all.min():.4f}, {preds_all.max():.4f}]\n')
    print(f'Results saved to: {results_path}')

    return mse, mae, mape, r2


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = f'log/reg/{args.log_dir}'
    vis_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print('PARAMETERS...')
    print(args)

    '''DATA LOADING'''
    print('Load dataset ...')
    DATA_PATH = args.data_path

    TEST_DATASET = LeaveDataLoader(root=DATA_PATH, args=args, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    model_class = load_model_and_loss(args.model)

    if args.model == 'pointnet_reg':
        model = model_class(feature_transform=args.feature_transform).cuda()
    else:
        model = model_class(args).cuda()

    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded trained model.')

    '''TESTING AND VISUALIZATION'''
    print('Start testing and visualization...')
    mse, mae, mape, r2 = test_and_visualize(model, testDataLoader, args, vis_dir)

    print(f'\n{"="*60}')
    print(f'Final Test Results:')
    print(f'MSE: {mse:.6f}')
    print(f'MAE: {mae:.6f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'R² Score: {r2:.6f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
