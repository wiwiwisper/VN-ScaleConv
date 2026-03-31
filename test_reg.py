"""
Author: Congyue Deng
Contact: congyue@stanford.edu
Date: April 2021
Modified by: You
Date: July 2025
Description: Regression test script supporting multiple models (PointNet, VN-DGCNN, etc.)
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import logging
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
    'vn_dgcnn_reg': ('vn_dgcnn_reg', 'get_model'),  # 假设使用 get_model 接口
    'dgcnn_reg': ('dgcnn_reg', 'get_model'),  # 假设使用 get_model 接口
    'vn_dgcnn_se_reg': ('vn_dgcnn_se_reg', 'get_model'),  # 假设使用 get_model 接口
    'vn_dgcnn_chidu_reg': ('vn_dgcnn_chidu_reg', 'get_model'),  # 假设使用 get_model 接口
    'vn_dgcnn_chidu_abl1_bn_reg': ('vn_dgcnn_chidu_abl1_bn', 'get_model'),      # 消融1: conv4加VN BN
    'vn_dgcnn_chidu_abl2_2scale_reg': ('vn_dgcnn_chidu_abl2_2scale', 'get_model'),  # 消融2: 2层scale_conv
    'vn_dgcnn_chidu_abl3_6scale_reg': ('vn_dgcnn_chidu_abl3_6scale', 'get_model'),  # 消融3: 6层scale_conv
}

def parse_args():
    parser = argparse.ArgumentParser('PointNet Regression Test')
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
    parser.add_argument('--n_knn', type=int, default=10, help='Number of nearest neighbors [default: 10]')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root folder')
    parser.add_argument('--label_unit', type=str, default='mm2', choices=['mm2', 'cm2'],
                        help='Unit of ground-truth labels: mm2 (default, divides by 100/10000 to display cm²) '
                             'or cm2 (already in cm², no conversion)')
    parser.add_argument('--result_file', type=str, default=None,
                        help='Path to a txt file where test results are appended (optional)')
    return parser.parse_args()


def log_string(logger, str):
    logger.info(str)
    print(str)


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


def test(model, loader, args):
    model.eval()
    mses = []
    maes = []
    mapes = []  # <-- 新增 MAPE 存储列表
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for points, label in tqdm(loader, desc='Testing'):
            points = points.float().cuda()
            label = label.float().cuda()

            # Apply rotation (if any)
            if args.rot != 'aligned':
                points = apply_rotation(points, args.rot)

            points = points.transpose(2, 1)  # [B, 3, N]

            pred, _ = model(points)

            # Save predictions and labels for final R² computation
            preds_all.append(pred.squeeze().cpu().numpy())
            labels_all.append(label.cpu().numpy())

            # Compute metrics
            mse = torch.mean((pred.squeeze() - label) ** 2).item()
            mae = torch.mean(torch.abs(pred.squeeze() - label)).item()

            # Compute MAPE: 避免除以零的问题
            epsilon = 1e-8  # 防止除以零
            mape = torch.mean(torch.abs((label - pred.squeeze()) / (label + epsilon))) * 100  # 百分比形式
            mape = mape.item()

            mses.append(mse)
            maes.append(mae)
            mapes.append(mape)

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    # Compute final metrics
    final_mse = np.mean(mses)
    final_mae = np.mean(maes)
    final_mape = np.mean(mapes)  # <-- 新增最终 MAPE

    # R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(labels_all, preds_all)

    return final_mse, final_mae, final_mape, r2, preds_all, labels_all


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = f'log/reg/{args.log_dir}'
    os.makedirs(experiment_dir, exist_ok=True)

    '''LOGGING'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{experiment_dir}/eval.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger, 'PARAMETERS...')
    log_string(logger, args)

    '''DATA LOADING'''
    log_string(logger, 'Load dataset ...')
    DATA_PATH = args.data_path

    TEST_DATASET = LeaveDataLoader(root=DATA_PATH, args=args, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    model_class = load_model_and_loss(args.model)

    if args.model == 'pointnet_reg':
        model = model_class(feature_transform=args.feature_transform).cuda()
    else:
        model = model_class(args).cuda()  # 其他模型可能需要 args 参数

    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    log_string(logger, 'Loaded trained model.')

    '''TESTING'''
    log_string(logger, 'Start testing...')
    mse, mae, mape, r2, preds_all, labels_all = test(model, testDataLoader, args)

    # Unit conversion: mm2 labels need /100 (MAE) and /10000 (MSE) to display as cm²;
    # cm2 labels are already in cm², no conversion needed.
    if args.label_unit == 'mm2':
        scale_mse = 10000.0
        scale_mae = 100.0
    else:  # cm2
        scale_mse = 1.0
        scale_mae = 1.0

    mse_display = mse / scale_mse
    mae_display = mae / scale_mae

    log_string(logger, f'Test Results:')
    log_string(logger, f'MSE: {mse_display:.6f} cm²')
    log_string(logger, f'MAE: {mae_display:.6f} cm²')
    log_string(logger, f'MAPE: {mape:.2f}%')
    log_string(logger, f'R² Score: {r2:.6f}')

    # Write per-sample MSE to result_file if specified
    if args.result_file is not None:
        filenames = [os.path.splitext(os.path.basename(p))[0] for p in TEST_DATASET.datapath]
        rows = []
        for fname, pred, label in zip(filenames, preds_all, labels_all):
            sample_mse = float((pred - label) ** 2) / scale_mse
            rows.append((fname, float(pred) / scale_mae, float(label) / scale_mae, sample_mse))
        rows.sort(key=lambda x: x[3], reverse=True)
        with open(args.result_file, 'w') as rf:
            rf.write('filename\tpred\tlabel\tMSE\n')
            for fname, pred, label, sample_mse in rows:
                rf.write(f'{fname}\t{pred:.6f}\t{label:.6f}\t{sample_mse:.6f}\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)