import argparse
import os
import sys
import logging

import numpy as np
import torch
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "data"))

from data_utils.LeafDataLoader import LeaveDataLoader

MODEL_REGISTRY = {
    "vn_dgcnn_chidu_reg": ("vn_dgcnn_chidu_reg", "get_model"),
}


def parse_args():
    parser = argparse.ArgumentParser("Regression test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", default="vn_dgcnn_chidu_reg", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--log_dir", type=str, default="vn_dgcnn_chidu_reg")
    parser.add_argument("--rot", type=str, default="aligned", choices=["aligned", "z", "so3"])
    parser.add_argument("--n_knn", type=int, default=10)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--label_unit", type=str, default="mm2", choices=["mm2", "cm2"])
    parser.add_argument("--result_file", type=str, default=None)
    return parser.parse_args()


def log_string(logger, message):
    logger.info(message)
    print(message)


def apply_rotation(points, rot_mode):
    if rot_mode == "z":
        angle = torch.rand(points.shape[0], device=points.device) * 360
        trot = RotateAxisAngle(angle=angle, axis="Z", degrees=True).to(points.device)
    elif rot_mode == "so3":
        rotations = random_rotations(points.shape[0]).to(points.device)
        trot = Rotate(R=rotations).to(points.device)
    else:
        return points

    points = trot.transform_points(points)
    return points


def load_model_and_loss(model_name):
    import importlib

    module_name, class_name = MODEL_REGISTRY[model_name]
    model_module = importlib.import_module(f"models.{module_name}")
    model_class = getattr(model_module, class_name)

    return model_class


def test(model, loader, args):
    model.eval()
    mses = []
    maes = []
    mapes = []
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for points, label in tqdm(loader, desc="Testing"):
            points = points.float().cuda()
            label = label.float().cuda()

            if args.rot != "aligned":
                points = apply_rotation(points, args.rot)

            points = points.transpose(2, 1)

            pred, _ = model(points)

            preds_all.append(pred.squeeze().cpu().numpy())
            labels_all.append(label.cpu().numpy())

            mse = torch.mean((pred.squeeze() - label) ** 2).item()
            mae = torch.mean(torch.abs(pred.squeeze() - label)).item()

            epsilon = 1e-8
            mape = torch.mean(torch.abs((label - pred.squeeze()) / (label + epsilon))) * 100
            mape = mape.item()

            mses.append(mse)
            maes.append(mae)
            mapes.append(mape)

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    final_mse = np.mean(mses)
    final_mae = np.mean(maes)
    final_mape = np.mean(mapes)

    from sklearn.metrics import r2_score

    r2 = r2_score(labels_all, preds_all)

    return final_mse, final_mae, final_mape, r2, preds_all, labels_all


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = f"log/reg/{args.log_dir}"
    os.makedirs(experiment_dir, exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(f"{experiment_dir}/eval.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger, "PARAMETERS...")
    log_string(logger, args)

    log_string(logger, "Load dataset ...")

    test_dataset = LeaveDataLoader(root=args.data_path, args=args, split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model_class = load_model_and_loss(args.model)
    model = model_class(args).cuda()

    checkpoint_path = os.path.join(experiment_dir, "checkpoints", "best_model.pth")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    log_string(logger, "Loaded trained model.")

    log_string(logger, "Start testing...")
    mse, mae, mape, r2, preds_all, labels_all = test(model, test_loader, args)

    if args.label_unit == "mm2":
        scale_mse = 10000.0
        scale_mae = 100.0
    else:
        scale_mse = 1.0
        scale_mae = 1.0

    mse_display = mse / scale_mse
    mae_display = mae / scale_mae

    log_string(logger, "Test Results:")
    log_string(logger, f"MSE: {mse_display:.6f} cm²")
    log_string(logger, f"MAE: {mae_display:.6f} cm²")
    log_string(logger, f"MAPE: {mape:.2f}%")
    log_string(logger, f"R² Score: {r2:.6f}")

    if args.result_file is not None:
        filenames = [os.path.splitext(os.path.basename(p))[0] for p in test_dataset.datapath]
        rows = []
        for fname, pred, label in zip(filenames, preds_all, labels_all):
            sample_mse = float((pred - label) ** 2) / scale_mse
            rows.append((fname, float(pred) / scale_mae, float(label) / scale_mae, sample_mse))
        rows.sort(key=lambda x: x[3], reverse=True)
        with open(args.result_file, "w") as rf:
            rf.write("filename\tpred\tlabel\tMSE\n")
            for fname, pred, label, sample_mse in rows:
                rf.write(f"{fname}\t{pred:.6f}\t{label:.6f}\t{sample_mse:.6f}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
