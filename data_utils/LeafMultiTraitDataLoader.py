import os
import pickle
import warnings

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

TRAIT_NAMES = [
    "length_cm",
    "width_cm",
    "perimeter_cm",
    "vein_length_cm",
    "area_cm2",
    "apex_angle_deg",
]


def load_point_cloud_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    return points


def farthest_point_sample(point, npoint):
    n, d = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((n,)) * 1e10
    farthest = np.random.randint(0, n)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def parse_gt_table(gt_txt, max_leaf_id=90):
    gt_map = {}
    with open(gt_txt, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        expected = [
            "leaf_id",
            "pdf_name",
            "length_cm",
            "width_cm",
            "perimeter_cm",
            "vein_length_cm",
            "area_cm2",
            "apex_angle_deg",
        ]
        if header != expected:
            raise ValueError(f"Unexpected GT header in {gt_txt}: {header}")
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            leaf_id = int(parts[0])
            if leaf_id > max_leaf_id:
                continue
            values = np.asarray([float(x) for x in parts[2:]], dtype=np.float32)
            gt_map[leaf_id] = values
    return gt_map


class LeafMultiTraitDataLoader(Dataset):
    def __init__(
        self,
        root,
        args,
        gt_txt,
        split="train",
        process_data=True,
        stats_path=None,
        max_leaf_id=90,
        target_trait_names=None,
        merge_val_test=False,
    ):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.split = split
        self.stats_path = stats_path
        self.max_leaf_id = max_leaf_id
        self.merge_val_test = merge_val_test
        self.gt_map = parse_gt_table(gt_txt, max_leaf_id=max_leaf_id)
        self.target_trait_names = list(target_trait_names) if target_trait_names is not None else list(TRAIT_NAMES)
        invalid = [name for name in self.target_trait_names if name not in TRAIT_NAMES]
        if invalid:
            raise ValueError(f"Unknown target traits: {invalid}")
        self.target_indices = [TRAIT_NAMES.index(name) for name in self.target_trait_names]

        self.save_dir = os.path.join(self.root, "processed_multitrait")
        os.makedirs(self.save_dir, exist_ok=True)

        self.datapath = []
        self.labels = []
        self.sample_names = []

        split_names = [split]
        if self.merge_val_test and split == "val":
            split_names = ["val", "test"]

        split_dirs = [os.path.join(self.root, name) for name in split_names]
        if not any(os.path.isdir(p) for p in split_dirs):
            print(f"Warning: split directory '{split_dirs}' not found. Using empty dataset.")
            self.list_of_points = []
            self.list_of_labels = []
            self.list_of_sample_names = []
            return

        for split_dir in split_dirs:
            if not os.path.isdir(split_dir):
                continue
            for class_dir in sorted(os.listdir(split_dir), key=lambda x: int(x) if x.isdigit() else x):
                class_path = os.path.join(split_dir, class_dir)
                if not os.path.isdir(class_path):
                    continue
                for file in sorted(os.listdir(class_path)):
                    if not file.endswith(".ply"):
                        continue
                    leaf_id = int(file.split("_")[0])
                    if leaf_id not in self.gt_map:
                        continue
                    self.datapath.append(os.path.join(class_path, file))
                    self.labels.append(self.gt_map[leaf_id][self.target_indices])
                    self.sample_names.append(file[:-4])

        print(f"Found {len(self.datapath)} multi-trait samples in '{split}' set.")

        trait_tag = "_".join(self.target_trait_names)
        split_tag = split
        if self.merge_val_test and split == "val":
            split_tag = "valtestmerge"
        cache_name = f"{split_tag}_{self.npoints}_maxleaf{self.max_leaf_id}_{trait_tag}.pkl"
        self.save_path = os.path.join(self.save_dir, cache_name)

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(f"Processing multi-trait data and saving to {self.save_path}...")
                self.list_of_points = []
                self.list_of_labels = []
                self.list_of_sample_names = []
                for idx in range(len(self.datapath)):
                    file_path = self.datapath[idx]
                    label_vec = self.labels[idx]
                    point_set = load_point_cloud_ply(file_path)
                    point_set = farthest_point_sample(point_set, self.npoints).astype(np.float32)
                    self.list_of_points.append(point_set)
                    self.list_of_labels.append(np.asarray(label_vec, dtype=np.float32))
                    self.list_of_sample_names.append(self.sample_names[idx])
                with open(self.save_path, "wb") as handle:
                    pickle.dump([self.list_of_points, self.list_of_labels, self.list_of_sample_names], handle)
            else:
                print(f"Loading processed multi-trait data from {self.save_path}")
                with open(self.save_path, "rb") as handle:
                    self.list_of_points, self.list_of_labels, self.list_of_sample_names = pickle.load(handle)
                if len(self.list_of_labels) > 0:
                    first_label = np.asarray(self.list_of_labels[0], dtype=np.float32)
                    expected_dim = len(self.target_trait_names)
                    if first_label.ndim != 1 or first_label.shape[0] != expected_dim:
                        print(
                            f"Cache mismatch detected for {self.save_path}, rebuilding..."
                        )
                        self.list_of_points = []
                        self.list_of_labels = []
                        self.list_of_sample_names = []
                        for idx in range(len(self.datapath)):
                            file_path = self.datapath[idx]
                            label_vec = self.labels[idx]
                            point_set = load_point_cloud_ply(file_path)
                            point_set = farthest_point_sample(point_set, self.npoints).astype(np.float32)
                            self.list_of_points.append(point_set)
                            self.list_of_labels.append(np.asarray(label_vec, dtype=np.float32))
                            self.list_of_sample_names.append(self.sample_names[idx])
                        with open(self.save_path, "wb") as handle:
                            pickle.dump([self.list_of_points, self.list_of_labels, self.list_of_sample_names], handle)
        else:
            raise NotImplementedError("Unprocessed multi-trait loading is not supported.")

        self.list_of_labels = [np.asarray(x, dtype=np.float32) for x in self.list_of_labels]

    def __len__(self):
        return len(self.list_of_labels)

    def __getitem__(self, index):
        point_set = self.list_of_points[index]
        label_raw = self.list_of_labels[index]
        sample_name = self.list_of_sample_names[index]
        return point_set, label_raw, sample_name
