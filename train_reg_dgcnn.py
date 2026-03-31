"""
Author: Your Name
Contact: your_email@example.com
Date: June 2025
Description: Training script for PointNet regression task (e.g., predicting leaf area)
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pytorch3d.transforms import Rotate, RotateAxisAngle, random_rotations  # 导入PyTorch3D变换模块

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))

# 导入自定义模块
from data_utils.LeafDataLoader import LeaveDataLoader  # 替换为你自己的数据集
MODEL_REGISTRY = {
    'pointnet_reg': ('pointnet_reg', 'get_model'),
    'vn_dgcnn_reg': ('vn_dgcnn_reg', 'get_model'),  # 假设你刚写的模型使用 get_model 接口
    'dgcnn_reg': ('dgcnn_reg', 'get_model'),  # 假设你刚写的模型使用 get_model 接口
}

def load_model_and_loss(model_name):
    import importlib

    module_name, class_name = MODEL_REGISTRY[model_name]
    model_module = importlib.import_module(f'models.{module_name}')
    model_class = getattr(model_module, class_name)
    loss_class = getattr(model_module, 'get_loss', None)

    if loss_class is None:
        raise ValueError(f"Loss function not found in {module_name}")

    return model_class, loss_class

def parse_args():
    parser = argparse.ArgumentParser('PointNet Regression Training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs [default: 200]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer [default: Adam]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet_reg', help='Log directory name [default: pointnet_reg]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: 0]')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root folder')
    parser.add_argument('--feature_transform', action='store_true', default=False,
                        help='Use feature transform in PointNet [default: False]')
    parser.add_argument('--rot', type=str, default='aligned', help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--model', type=str, default='pointnet_reg',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture to train')
    return parser.parse_args()


def log_string(logger, str):
    logger.info(str)
    print(str)


def apply_rotation(points, rot_mode):
    """Apply rotation augmentation to the points."""
    if rot_mode == 'z':
        angle = torch.rand(points.shape[0], device=points.device) * 360
        trot = RotateAxisAngle(angle=angle, axis="Z", degrees=True).to(device=points.device)
    elif rot_mode == 'so3':
        R = random_rotations(points.shape[0]).to(device=points.device)
        trot = Rotate(R=R).to(device=points.device)  # 确保整个变换对象也在 GPU 上
    else:
        trot = None

    if trot is not None:
        points = trot.transform_points(points)

    return points


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/reg/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOGGING'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger, 'PARAMETERS...')
    log_string(logger, args)

    '''DATA LOADING'''
    log_string(logger, 'Load dataset ...')
    DATA_PATH = args.data_path

    TRAIN_DATASET = LeaveDataLoader(root=DATA_PATH, args=args, split='train')
    TEST_DATASET = LeaveDataLoader(root=DATA_PATH, args=args, split='test')
    VAL_DATASET = LeaveDataLoader(root=DATA_PATH, args=args, split='val')

    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valDataLoader = DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # model = PointNetReg(feature_transform=args.feature_transform).cuda()
    # criterion = PointNetRegLoss(reg_weight=0.001).cuda()
    # 动态加载模型和损失函数
    model_class, loss_class = load_model_and_loss(args.model)

    # 初始化模型
    if args.model == 'pointnet_reg':
        model = model_class(feature_transform=args.feature_transform).cuda()
    else:
        model = model_class(args).cuda()  # 如果你的 vn_dgcnn_reg 需要 args

    # 初始化损失函数
    criterion = loss_class().cuda()


    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string(logger, 'Use pretrain model')
    except:
        log_string(logger, 'No existing model, starting training from scratch...')
        start_epoch = 0

    '''OPTIMIZER'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate * 100, momentum=0.9, weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_loss = float('inf')
    global_epoch = 0

    '''TRAINING LOOP'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string(logger, f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):')

        scheduler.step()
        loss_total = []

        for batch_id, (points, label) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points = points.float().cuda()     # shape: [B, N, 3]
            label = label.float().cuda()       # shape: [B]

            # Apply rotation augmentation
            if args.rot != 'aligned':
                points = apply_rotation(points, args.rot)

            points = points.transpose(2, 1)    # 输入要求为 [B, 3, N]

            optimizer.zero_grad()
            model.train()
            pred, trans_feat = model(points)

            loss = criterion(pred.squeeze(), label, trans_feat)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.item())

        train_loss = np.mean(loss_total)
        log_string(logger, f'Train Loss: {train_loss:.6f}')

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss_total = []
            for points, label in valDataLoader:
                points = points.float().cuda()
                label = label.float().cuda()
                points = points.transpose(2, 1)
                pred, trans_feat = model(points)
                loss = criterion(pred.squeeze(), label, trans_feat)
                val_loss_total.append(loss.item())
            val_loss = np.mean(val_loss_total)
            log_string(logger, f'Val Loss: {val_loss:.6f}')

            if val_loss < best_loss:
                best_loss = val_loss
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string(logger, f'Saving model at {savepath}')
                state = {
                    'epoch': epoch + 1,
                    'loss': val_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)