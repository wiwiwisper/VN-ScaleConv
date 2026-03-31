"""
Author: Your Name
Contact: your_email@example.com
Date: June 2025
Description: Training script for TetraSphere regression task
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
from pytorch3d.transforms import Rotate, RotateAxisAngle, random_rotations

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models1'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))

# 导入自定义模块
from data_utils.LeafDataLoader import LeaveDataLoader
from tetrasphere_reg import get_model, get_loss


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
        trot = Rotate(R=R).to(device=points.device)
    else:
        trot = None

    if trot is not None:
        points = trot.transform_points(points)

    return points


def parse_args():
    parser = argparse.ArgumentParser('TetraSphere Regression Training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs [default: 200]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help='Optimizer [default: Adam]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point number [default: 1024]')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Full name for log directory (e.g. "tetrasphere_exp1")')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: 0]')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root folder')
    parser.add_argument('--rot', type=str, default='aligned', help='Rotation augmentation [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    # TetraSphere特定参数
    parser.add_argument('--num_spheres', type=int, default=4, help='Number of spheres [default: 4]')
    parser.add_argument('--k', type=int, default=20, help='Number of neighbors [default: 20]')
    parser.add_argument('--C_prime', type=int, default=3, help='Feature dimension coefficient [default: 3]')
    parser.add_argument('--no_mean', action='store_true', help='Disable mean feature')
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    # 直接使用用户指定的log_dir作为完整路径
    experiment_dir = Path('./log') / args.log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = experiment_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    log_file = experiment_dir / f'{args.log_dir}.log'

    '''LOGGING'''
    logger = logging.getLogger("TetraSphere")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 文件日志
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_string(logger, f'Experiment directory: {experiment_dir}')
    log_string(logger, 'PARAMETERS:')
    for arg in vars(args):
        log_string(logger, f'{arg}: {getattr(args, arg)}')

    '''DATA LOADING'''
    log_string(logger, 'Loading dataset...')
    TRAIN_DATASET = LeaveDataLoader(root=args.data_path, args=args, split='train')
    TEST_DATASET = LeaveDataLoader(root=args.data_path, args=args, split='test')
    VAL_DATASET = LeaveDataLoader(root=args.data_path, args=args, split='val')

    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valDataLoader = DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL INITIALIZATION'''
    log_string(logger, 'Initializing TetraSphere model...')
    model = get_model(
        num_spheres=args.num_spheres,
        k=args.k,
        C_prime=args.C_prime,
        no_mean=args.no_mean
    ).cuda()
    criterion = get_loss().cuda()

    '''LOAD CHECKPOINT IF EXISTS'''
    checkpoint_path = checkpoints_dir / 'best_model.pth'
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string(logger, f'Loaded checkpoint from {checkpoint_path}')
    except:
        log_string(logger, 'No checkpoint found, starting from scratch...')
        start_epoch = 0

    '''OPTIMIZER'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate * 100,
                                    momentum=0.9,
                                    weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_val_loss = float('inf')

    '''TRAINING LOOP'''
    log_string(logger, 'Start training...')
    for epoch in range(start_epoch, args.epoch):
        model.train()
        epoch_loss = []

        # Training phase
        for points, labels in tqdm(trainDataLoader, desc=f'Epoch {epoch + 1}'):
            points = points.float().cuda().transpose(2, 1)  # [B, 3, N]
            labels = labels.float().cuda()  # [B]

            if args.rot != 'aligned':
                points = apply_rotation(points.transpose(2, 1), args.rot).transpose(2, 1)

            optimizer.zero_grad()
            pred, _ = model(points)  # 忽略第二个返回值
            loss = criterion(pred.squeeze(), labels)  # 只传入pred和target
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        train_loss = np.mean(epoch_loss)
        log_string(logger, f'Train Epoch {epoch + 1}: Loss {train_loss:.6f}')

        # Validation phase
        model.eval()
        val_loss = []
        with torch.no_grad():
            for points, labels in valDataLoader:
                points = points.float().cuda().transpose(2, 1)
                labels = labels.float().cuda()
                pred, _ = model(points)  # 忽略第二个返回值
                loss = criterion(pred.squeeze(), labels)  # 只传入pred和target
                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        log_string(logger, f'Validation Loss: {avg_val_loss:.6f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
            log_string(logger, f'Saved new best model with val loss {best_val_loss:.6f}')

        scheduler.step()

    log_string(logger, 'Training completed!')
    log_string(logger, f'Final best validation loss: {best_val_loss:.6f}')
    log_string(logger, f'Logs saved to: {log_file}')
    log_string(logger, f'Model saved to: {checkpoint_path}')


if __name__ == '__main__':
    args = parse_args()
    main(args)