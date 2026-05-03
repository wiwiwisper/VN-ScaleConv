import argparse
import os
import sys
import datetime
import logging
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

from data_utils.LeafDataLoader import LeaveDataLoader

MODEL_REGISTRY = {
    "vn_dgcnn_chidu_reg": ("vn_dgcnn_chidu_reg", "get_model"),
}


def load_model_and_loss(model_name):
    import importlib

    module_name, class_name = MODEL_REGISTRY[model_name]
    model_module = importlib.import_module(f"models.{module_name}")
    model_class = getattr(model_module, class_name)
    loss_class = getattr(model_module, "get_loss", None)

    if loss_class is None:
        raise ValueError(f"Loss function not found in {module_name}")

    return model_class, loss_class


def parse_args():
    parser = argparse.ArgumentParser("Regression training")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--log_dir", type=str, default="vn_dgcnn_chidu_reg")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--rot", type=str, default="aligned", choices=["aligned", "z", "so3"])
    parser.add_argument("--n_knn", type=int, default=10)
    parser.add_argument("--model", type=str, default="vn_dgcnn_chidu_reg", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def log_string(logger, message):
    logger.info(message)
    print(message)


def apply_rotation(points, rot_mode):
    if rot_mode == "z":
        angle = torch.rand(points.shape[0], device=points.device) * 360
        trot = RotateAxisAngle(angle=angle, axis="Z", degrees=True).to(device=points.device)
    elif rot_mode == "so3":
        rotations = random_rotations(points.shape[0]).to(device=points.device)
        trot = Rotate(R=rotations).to(device=points.device)
    else:
        trot = None

    if trot is not None:
        points = trot.transform_points(points)

    return points


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    experiment_dir = Path("./log/reg/")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath("logs/")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, args.log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger, "PARAMETERS...")
    log_string(logger, args)

    log_string(logger, "Load dataset ...")

    train_dataset = LeaveDataLoader(root=args.data_path, args=args, split="train")
    test_dataset = LeaveDataLoader(root=args.data_path, args=args, split="test")
    val_dataset = LeaveDataLoader(root=args.data_path, args=args, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if len(val_dataset) == 0:
        log_string(logger, "No val split found, using test set as val.")
        val_loader = test_loader
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_class, loss_class = load_model_and_loss(args.model)
    model = model_class(args).cuda()
    criterion = loss_class().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + "/checkpoints/best_model.pth")
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        log_string(logger, "Use pretrain model")
    except Exception:
        log_string(logger, "No existing model, starting training from scratch...")
        start_epoch = 0

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate * 100, momentum=0.9, weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_loss = float('inf')
    global_epoch = 0

    logger.info("Start training...")
    for epoch in range(start_epoch, args.epoch):
        log_string(logger, f"Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):")

        scheduler.step()
        loss_total = []

        for _batch_id, (points, label) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            points = points.float().cuda()
            label = label.float().cuda()

            if args.rot != "aligned":
                points = apply_rotation(points, args.rot)

            points = points.transpose(2, 1)

            optimizer.zero_grad()
            model.train()
            pred, trans_feat = model(points)

            loss = criterion(pred.squeeze(), label, trans_feat)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.item())

        train_loss = np.mean(loss_total)
        log_string(logger, f"Train Loss: {train_loss:.6f}")

        with torch.no_grad():
            model.eval()
            val_loss_total = []
            for points, label in val_loader:
                points = points.float().cuda()
                label = label.float().cuda()
                if args.rot != "aligned":
                    points = apply_rotation(points, args.rot)
                points = points.transpose(2, 1)
                pred, trans_feat = model(points)
                loss = criterion(pred.squeeze(), label, trans_feat)
                val_loss_total.append(loss.item())
            val_loss = np.mean(val_loss_total)
            log_string(logger, f"Val Loss: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                savepath = str(checkpoints_dir) + "/best_model.pth"
                log_string(logger, f"Saving model at {savepath}")
                state = {
                    "epoch": epoch + 1,
                    "loss": val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)

        global_epoch += 1

    logger.info("End of training...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
