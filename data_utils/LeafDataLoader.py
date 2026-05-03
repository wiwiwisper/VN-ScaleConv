import os
import pickle
import warnings

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_point_cloud_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=float)
    return points


def farthest_point_sample(point, npoint):
    num_points, _num_dims = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((num_points,)) * 1e10
    farthest = np.random.randint(0, num_points)
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
    def __init__(self, root, args, split="train", process_data=True):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.split = split
        self.catfile = os.path.join(self.root, "area_ground_truth.txt")
        self.save_dir = os.path.join(self.root, "processed")
        os.makedirs(self.save_dir, exist_ok=True)

        self.category_to_label = {}
        with open(self.catfile, "r") as handle:
            for line in handle:
                filename, label = line.strip().split()
                self.category_to_label[filename] = float(label)

        self.datapath = []
        self.labels = []

        split_dir = os.path.join(self.root, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: split directory '{split_dir}' not found. Using empty dataset.")
        else:
            for class_dir in sorted(os.listdir(split_dir), key=lambda x: int(x) if x.isdigit() else x):
                class_path = os.path.join(split_dir, class_dir)
                if not os.path.isdir(class_path):
                    continue
                for file in sorted(os.listdir(class_path)):
                    if file.endswith(".ply"):
                        fname = file[:-4]
                        if fname in self.category_to_label:
                            self.datapath.append(os.path.join(class_path, file))
                            self.labels.append(self.category_to_label[fname])

        print(f"Found {len(self.datapath)} samples in '{split}' set.")

        self.save_path = os.path.join(self.save_dir, f"{split}_{self.npoints}.dat")

        if len(self.datapath) == 0:
            self.list_of_points = []
            self.list_of_areas = []
            return

        if not self.process_data:
            raise NotImplementedError("Unprocessed data loading is not supported.")

        if not os.path.exists(self.save_path):
            print(f"Processing data and saving to {self.save_path}...")
            self.list_of_points = []
            self.list_of_areas = []
            for idx in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                file_path = self.datapath[idx]
                area = self.labels[idx]
                point_set = load_point_cloud_ply(file_path)
                point_set = farthest_point_sample(point_set, self.npoints)
                self.list_of_points.append(point_set)
                self.list_of_areas.append(area)
            with open(self.save_path, "wb") as handle:
                pickle.dump([self.list_of_points, self.list_of_areas], handle)
        else:
            print(f"Loading processed data from {self.save_path}")
            with open(self.save_path, "rb") as handle:
                self.list_of_points, self.list_of_areas = pickle.load(handle)

    def __len__(self):
        return len(self.labels)

    def _get_item(self, index):
        point_set, areas = self.list_of_points[index], self.list_of_areas[index]
        return point_set, areas

    def __getitem__(self, index):
        return self._get_item(index)
