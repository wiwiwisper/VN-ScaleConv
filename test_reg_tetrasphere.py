"""
TetraSphere回归测试脚本 - 最终修复版
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pytorch3d.transforms import Rotate, RotateAxisAngle, random_rotations
from pathlib import Path
import logging

# 路径设置
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models1'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))

from data_utils.LeafDataLoader import LeaveDataLoader
from tetrasphere_reg import get_model, get_loss

def log_string(logger, str):
    logger.info(str)
    print(str)

def apply_rotation(points, rot_mode, device):
    """旋转增强"""
    if rot_mode == 'z':
        angle = torch.rand(points.shape[0], device=device) * 360
        trot = RotateAxisAngle(angle=angle, axis="Z", degrees=True).to(device)
    elif rot_mode == 'so3':
        R = random_rotations(points.shape[0]).to(device)
        trot = Rotate(R=R).to(device)
    else:  # 'aligned'
        return points.transpose(2, 1)  # [B, 3, N]

    return trot.transform_points(points).transpose(2, 1)

def calculate_mape(y_true, y_pred):
    """修复后的MAPE计算"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

def main():
    # 参数解析
    parser = argparse.ArgumentParser('TetraSphere Regression Testing')
    parser.add_argument('--data_path', required=True, help='数据集路径')
    parser.add_argument('--log_dir', required=True, help='训练日志目录名')
    parser.add_argument('--rot', default='aligned', choices=['aligned', 'z', 'so3'],
                      help='旋转增强方式')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_spheres', type=int, default=4)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--C_prime', type=int, default=3)
    parser.add_argument('--no_mean', action='store_true')
    parser.add_argument('--num_point', type=int, default=1024)
    args = parser.parse_args()

    # 设备设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志设置
    experiment_dir = Path('./log') / args.log_dir
    checkpoints_dir = experiment_dir / 'checkpoints'

    logger = logging.getLogger("TetraSphere-Test")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(experiment_dir / f'test_{args.log_dir}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 模型初始化
    model = get_model(
        num_spheres=args.num_spheres,
        k=args.k,
        C_prime=args.C_prime,
        no_mean=args.no_mean
    ).to(device)

    # 加载模型
    checkpoint_path = checkpoints_dir / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path)

    # 修正键名
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "steerable_layer._weight":
            new_key = "steerable_layer.weight"
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    # 数据加载
    TEST_DATASET = LeaveDataLoader(root=args.data_path, args=args, split='test')
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False)

    # 测试循环
    criterion = get_loss().to(device)
    test_loss = []
    preds, truths = [], []

    with torch.no_grad():
        for points, labels in tqdm(testDataLoader, desc="测试进度"):
            points = points.float().to(device)
            labels = labels.float().to(device)

            if args.rot != 'aligned':
                points = apply_rotation(points, args.rot, device)
            else:
                points = points.transpose(2, 1)

            pred, _ = model(points)
            loss = criterion(pred.squeeze(), labels)

            test_loss.append(loss.item())
            preds.extend(pred.cpu().numpy().flatten().tolist())  # 确保展平
            truths.extend(labels.cpu().numpy().flatten().tolist())

    # 结果计算
    truths = np.array(truths)
    preds = np.array(preds)

    # 结果打印
    print("\n=== 最终测试结果 ===")
    print(f'测试Loss: {np.mean(test_loss):.6f}')
    print(f'MSE: {mean_squared_error(truths, preds):.6f}')
    print(f'MAE: {mean_absolute_error(truths, preds):.6f}')
    print(f'MAPE: {calculate_mape(truths, preds):.2f}%')
    print(f'R²: {r2_score(truths, preds):.6f}')

if __name__ == "__main__":
    main()