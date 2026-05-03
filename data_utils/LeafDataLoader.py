import os
from argparse import Namespace

import numpy as np
import warnings
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import open3d as o3d
warnings.filterwarnings('ignore')

# 读取点云数据的函数
def load_point_cloud_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=float)
    return points

# 最远点采样
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class LeaveDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=True):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.split = split
        self.catfile = os.path.join(self.root, 'area_ground_truth.txt')
        self.save_dir = os.path.join(self.root, 'processed')
        os.makedirs(self.save_dir, exist_ok=True)

        # 读取 ground truth
        self.category_to_label = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                filename, label = line.strip().split()
                self.category_to_label[filename] = float(label)

        # 构建 datapath 和 labels
        self.datapath = []
        self.labels = []

        split_dir = os.path.join(self.root, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: split directory '{split_dir}' not found. Using empty dataset.")
        else:
            for class_dir in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_dir)
                if not os.path.isdir(class_path):
                    continue
                for file in os.listdir(class_path):
                    if file.endswith('.ply'):
                        fname = file[:-4]  # 去掉 .ply 后缀
                        if fname in self.category_to_label:
                            self.datapath.append(os.path.join(class_path, file))
                            self.labels.append(self.category_to_label[fname])

        print(f"Found {len(self.datapath)} samples in '{split}' set.")

        # 预处理路径
        self.save_path = os.path.join(self.save_dir, f'{split}_{self.npoints}.dat')

        if len(self.datapath) == 0:
            self.list_of_points = []
            self.list_of_areas = []
            return

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(f'Processing data and saving to {self.save_path}...')
                self.list_of_points = []
                self.list_of_areas = []
                for idx in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    file_path = self.datapath[idx]
                    area = self.labels[idx]
                    point_set = load_point_cloud_ply(file_path)
                    point_set = farthest_point_sample(point_set, self.npoints)
                    self.list_of_points.append(point_set)
                    self.list_of_areas.append(area)
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_areas], f)
            else:
                print(f'Loading processed data from {self.save_path}')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_areas = pickle.load(f)



    def __len__(self):
        return len(self.labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, areas = self.list_of_points[index], self.list_of_areas[index]
        else:
            raise NotImplementedError("Unprocessed data loading is not supported.")
        return point_set, areas

    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # point_set = load_point_cloud_ply('/mnt/sdc1/dyt2/rgbd_proj/vnn-master/data/rgbd_ply/1_big_854509890.ply')
    # print(f"Loaded point cloud shape: {point_set.shape}")  # [N, 3] ?
'''
    args = Namespace(num_point=1024)

    # 训练集
    train_dataset = LeaveDataLoader(root='../data/small', args=args, split='train', process_data=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 验证集
    val_dataset = LeaveDataLoader(root='../data/small', args=args, split='val', process_data=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # 测试集
    test_dataset = LeaveDataLoader(root='../data/small', args=args, split='test', process_data=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
'''
