import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import Rotate, RotateAxisAngle, random_rotations
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "data"))

from data_utils.LeafMultiTraitDataLoader import LeafMultiTraitDataLoader, TRAIT_NAMES

MODEL_REGISTRY = {
    "vn_dgcnn_chidu_multitrait": ("vn_dgcnn_chidu_multitrait", "get_model"),
}


def parse_args():
    parser = argparse.ArgumentParser("Multi-trait leaf regression training")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gt_txt", type=str, default="phenotyping/data/final/gt_leaf_traits_6params_90.txt")
    parser.add_argument("--feature_transform", action="store_true", default=False)
    parser.add_argument("--rot", type=str, default="aligned", choices=["aligned", "z", "so3"])
    parser.add_argument("--n_knn", type=int, default=10)
    parser.add_argument("--model", type=str, default="vn_dgcnn_chidu_multitrait", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_leaf_id", type=int, default=90)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--output_root", type=str, default="phenotyping/data/final/ours_small_multitrait")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--target_traits", type=str, default="all", help="Comma-separated trait names or 'all'")
    parser.add_argument("--merge_val_test", action="store_true", default=False, help="Merge val and test as one validation split")
    return parser.parse_args()


def log_string(logger, message):
    logger.info(message)
    print(message)


def apply_rotation(points, rot_mode):
    if rot_mode == "z":
        angle = torch.rand(points.shape[0], device=points.device) * 360
        trot = RotateAxisAngle(angle=angle, axis="Z", degrees=True).to(device=points.device)
    elif rot_mode == "so3":
        r = random_rotations(points.shape[0]).to(device=points.device)
        trot = Rotate(R=r).to(device=points.device)
    else:
        trot = None
    if trot is not None:
        points = trot.transform_points(points)
    return points


def load_model_and_loss(model_name):
    import importlib

    module_name, class_name = MODEL_REGISTRY[model_name]
    model_module = importlib.import_module(f"models.{module_name}")
    model_class = getattr(model_module, class_name)
    loss_class = getattr(model_module, "get_loss")
    return model_class, loss_class


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(ROOT_DIR) / path


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    target_trait_names = TRAIT_NAMES if args.target_traits == "all" else [x.strip() for x in args.target_traits.split(",") if x.strip()]
    args.output_dim = len(target_trait_names)

    experiment_dir = resolve_project_path(args.output_root) / args.log_dir
    checkpoints_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiment_dir / "train_config.json"

    logger = logging.getLogger(f"MultiTraitTrain_{args.log_dir}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(str(logs_dir / f"{args.log_dir}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string(logger, "PARAMETERS...")
    log_string(logger, str(args))
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    log_string(logger, "Load multi-trait dataset ...")
    train_dataset = LeafMultiTraitDataLoader(
        root=args.data_path,
        args=args,
        gt_txt=args.gt_txt,
        split="train",
        max_leaf_id=args.max_leaf_id,
        target_trait_names=target_trait_names,
        merge_val_test=args.merge_val_test,
    )
    val_dataset = LeafMultiTraitDataLoader(
        root=args.data_path,
        args=args,
        gt_txt=args.gt_txt,
        split="val",
        max_leaf_id=args.max_leaf_id,
        target_trait_names=target_trait_names,
        merge_val_test=args.merge_val_test,
    )
    test_dataset = LeafMultiTraitDataLoader(
        root=args.data_path,
        args=args,
        gt_txt=args.gt_txt,
        split="test",
        max_leaf_id=args.max_leaf_id,
        target_trait_names=target_trait_names,
        merge_val_test=args.merge_val_test,
    )
    if len(val_dataset) == 0:
        log_string(logger, "No val split found, using test split as val.")
        val_dataset = test_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_class, loss_class = load_model_and_loss(args.model)
    model = model_class(args).cuda()
    criterion = loss_class().cuda()

    start_epoch = 0
    checkpoint_path = checkpoints_dir / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path))
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        log_string(logger, "Use pretrain model")
    else:
        log_string(logger, "No existing model, starting training from scratch...")

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate * 100, momentum=0.9, weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_val_loss = float("inf")

    log_string(logger, f"Trait order: {target_trait_names}")
    log_string(logger, "Start training...")

    for epoch in range(start_epoch, args.epoch):
        scheduler.step()
        model.train()
        train_losses = []
        for points, label_raw, _sample_name in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
            points = points.float().cuda()
            label_raw = label_raw.float().cuda()

            if args.rot != "aligned":
                points = apply_rotation(points, args.rot)
            points = points.transpose(2, 1)

            optimizer.zero_grad()
            pred_raw, _ = model(points)
            loss = criterion(pred_raw, label_raw, None)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        log_string(logger, f"Epoch {epoch + 1}/{args.epoch} Train Loss: {train_loss:.6f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for points, label_raw, _sample_name in val_loader:
                points = points.float().cuda()
                label_raw = label_raw.float().cuda()
                if args.rot != "aligned":
                    points = apply_rotation(points, args.rot)
                points = points.transpose(2, 1)
                pred_raw, _ = model(points)
                loss = criterion(pred_raw, label_raw, None)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        log_string(logger, f"Epoch {epoch + 1}/{args.epoch} Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "trait_names": target_trait_names,
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, str(checkpoint_path))
            log_string(logger, f"Saved best model to {checkpoint_path}")

    log_string(logger, f"End of training. Held-out test samples available: {len(test_dataset)}")


if __name__ == "__main__":
    main(parse_args())
