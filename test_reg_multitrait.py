import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import Rotate, RotateAxisAngle, random_rotations
from sklearn.metrics import r2_score
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
    parser = argparse.ArgumentParser("Multi-trait leaf regression test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--rot", type=str, default="aligned", choices=["aligned", "z", "so3"])
    parser.add_argument("--n_knn", type=int, default=10)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gt_txt", type=str, default="phenotyping/data/final/gt_leaf_traits_6params_90.txt")
    parser.add_argument("--max_leaf_id", type=int, default=90)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--model", type=str, default="vn_dgcnn_chidu_multitrait", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--result_tag", type=str, required=True)
    parser.add_argument("--target_traits", type=str, default="all", help="Comma-separated trait names or 'all'")
    parser.add_argument("--merge_val_test", action="store_true", default=False, help="Evaluate on merged val+test instead of test only")
    return parser.parse_args()


def apply_rotation(points, rot_mode):
    if rot_mode == "z":
        angle = torch.rand(points.shape[0], device=points.device) * 360
        trot = RotateAxisAngle(angle=angle, axis="Z", degrees=True).to(points.device)
    elif rot_mode == "so3":
        rotations = random_rotations(points.shape[0]).to(points.device)
        trot = Rotate(R=rotations).to(points.device)
    else:
        return points
    return trot.transform_points(points)


def load_model(model_name):
    import importlib

    module_name, class_name = MODEL_REGISTRY[model_name]
    model_module = importlib.import_module(f"models.{module_name}")
    return getattr(model_module, class_name)


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(ROOT_DIR) / path


def compute_metrics(labels, preds, trait_names):
    metrics = []
    for i, trait in enumerate(trait_names):
        gt = labels[:, i]
        pd = preds[:, i]
        mse = float(np.mean((pd - gt) ** 2))
        mae = float(np.mean(np.abs(pd - gt)))
        mape = float(np.mean(np.abs((gt - pd) / (gt + 1e-8))) * 100.0)
        r2 = float(r2_score(gt, pd))
        metrics.append(
            {
                "trait": trait,
                "mse": mse,
                "mae": mae,
                "mape_percent": mape,
                "r2": r2,
            }
        )
    metrics.append(
        {
            "trait": "mean",
            "mse": float(np.mean([m["mse"] for m in metrics])),
            "mae": float(np.mean([m["mae"] for m in metrics])),
            "mape_percent": float(np.mean([m["mape_percent"] for m in metrics])),
            "r2": float(np.mean([m["r2"] for m in metrics])),
        }
    )
    return metrics


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    target_trait_names = TRAIT_NAMES if args.target_traits == "all" else [x.strip() for x in args.target_traits.split(",") if x.strip()]
    args.output_dim = len(target_trait_names)

    experiment_dir = resolve_project_path(args.experiment_dir)
    checkpoint_path = experiment_dir / "checkpoints" / "best_model.pth"
    eval_dir = experiment_dir / "eval_outputs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = LeafMultiTraitDataLoader(
        root=args.data_path,
        args=args,
        gt_txt=args.gt_txt,
        split="val" if args.merge_val_test else "test",
        max_leaf_id=args.max_leaf_id,
        target_trait_names=target_trait_names,
        merge_val_test=args.merge_val_test,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_class = load_model(args.model)
    model = model_class(args).cuda()

    checkpoint = torch.load(str(checkpoint_path))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds_raw = []
    labels_raw = []
    filenames = []

    with torch.no_grad():
        for points, label_raw, sample_name in tqdm(test_loader, desc="Testing"):
            points = points.float().cuda()
            if args.rot != "aligned":
                points = apply_rotation(points, args.rot)
            points = points.transpose(2, 1)

            pred_raw, _ = model(points)
            pred_raw = pred_raw.detach().cpu().numpy()

            preds_raw.append(pred_raw)
            labels_raw.append(label_raw.numpy())
            filenames.extend(list(sample_name))

    preds_raw = np.concatenate(preds_raw, axis=0)
    labels_raw = np.concatenate(labels_raw, axis=0)
    metrics = compute_metrics(labels_raw, preds_raw, target_trait_names)

    metrics_path = eval_dir / f"{args.result_tag}_metrics.tsv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["trait", "mse", "mae", "mape_percent", "r2"])
        for row in metrics:
            writer.writerow([row["trait"], f"{row['mse']:.6f}", f"{row['mae']:.6f}", f"{row['mape_percent']:.6f}", f"{row['r2']:.6f}"])

    preds_path = eval_dir / f"{args.result_tag}_predictions.tsv"
    with open(preds_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        header = ["sample_name"]
        for trait in target_trait_names:
            header += [f"gt_{trait}", f"pred_{trait}"]
        writer.writerow(header)
        for idx, name in enumerate(filenames):
            row = [name]
            for t in range(len(target_trait_names)):
                row += [f"{labels_raw[idx, t]:.6f}", f"{preds_raw[idx, t]:.6f}"]
            writer.writerow(row)

    summary = {
        "experiment_dir": str(experiment_dir),
        "test_rotation": args.rot,
        "num_test_samples": int(len(filenames)),
        "metrics_file": str(metrics_path),
        "predictions_file": str(preds_path),
        "trait_names": target_trait_names,
    }
    summary_path = eval_dir / f"{args.result_tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main(parse_args())
