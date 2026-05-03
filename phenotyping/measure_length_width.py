from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from visualizer.plyfile import PlyData


@dataclass
class LeafTraitRow:
    ply_name: str
    length_obb_m: float
    width_obb_m: float
    apex_angle_deg: float
    perimeter_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure leaf traits from single-leaf PLY point clouds.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data/big", "data/small"],
        help="Dataset roots to scan recursively for .ply files.",
    )
    parser.add_argument(
        "--output-dir",
        default="phenotyping",
        help="Directory for TXT outputs.",
    )
    parser.add_argument(
        "--long-side-pixels",
        type=int,
        default=512,
        help="Raster width for 2D contour extraction.",
    )
    return parser.parse_args()


def load_points(ply_path: str) -> np.ndarray:
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)


def pca_project(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1.0
    return centered @ eigenvectors


def iter_ply_files(dataset_root: str) -> Iterable[str]:
    for root, _, files in os.walk(dataset_root):
        for filename in sorted(files):
            if filename.endswith(".ply"):
                yield os.path.join(root, filename)


def smooth_contour(contour_xy: np.ndarray, window: int = 9) -> np.ndarray:
    if len(contour_xy) < window or window < 3:
        return contour_xy
    pad = window // 2
    extended = np.vstack([contour_xy[-pad:], contour_xy, contour_xy[:pad]])
    kernel = np.ones(window, dtype=np.float64) / window
    smooth_x = np.convolve(extended[:, 0], kernel, mode="valid")
    smooth_y = np.convolve(extended[:, 1], kernel, mode="valid")
    return np.column_stack([smooth_x, smooth_y])


def contour_perimeter(contour_xy: np.ndarray) -> float:
    closed = np.vstack([contour_xy, contour_xy[:1]])
    diffs = np.diff(closed, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def rasterize_contour(points_xy: np.ndarray, long_side_pixels: int) -> np.ndarray:
    min_xy = points_xy.min(axis=0)
    max_xy = points_xy.max(axis=0)
    extent = np.maximum(max_xy - min_xy, 1e-8)
    scale = float(long_side_pixels) / float(extent.max())
    pad = 10

    pixel_xy = np.round((points_xy - min_xy) * scale).astype(np.int32) + pad
    image_w = int(pixel_xy[:, 0].max()) + pad + 1
    image_h = int(pixel_xy[:, 1].max()) + pad + 1
    image = np.zeros((image_h, image_w), dtype=np.uint8)
    image[pixel_xy[:, 1], pixel_xy[:, 0]] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Failed to extract a 2D contour from projected points.")

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    contour_xy = (contour - pad) / scale + min_xy
    return smooth_contour(contour_xy, window=9)


def end_width(points_xy: np.ndarray, x_threshold: float, use_max_end: bool) -> float:
    if use_max_end:
        slab = points_xy[:, 0] >= x_threshold
    else:
        slab = points_xy[:, 0] <= x_threshold
    if slab.sum() < 5:
        return float("inf")
    y_values = points_xy[slab, 1]
    return float(y_values.max() - y_values.min())


def select_tip_end(points_xy: np.ndarray) -> bool:
    min_x = float(points_xy[:, 0].min())
    max_x = float(points_xy[:, 0].max())
    length = max_x - min_x
    slab = max(0.06 * length, 1e-6)
    min_width = end_width(points_xy, min_x + slab, use_max_end=False)
    max_width = end_width(points_xy, max_x - slab, use_max_end=True)
    return max_width < min_width


def select_tip_index(contour_xy: np.ndarray, tip_is_max_x: bool) -> int:
    x_values = contour_xy[:, 0]
    x_extreme = float(x_values.max()) if tip_is_max_x else float(x_values.min())
    x_span = float(x_values.max() - x_values.min())
    tolerance = max(0.015 * x_span, 1e-6)
    if tip_is_max_x:
        candidate_idx = np.flatnonzero(x_values >= x_extreme - tolerance)
    else:
        candidate_idx = np.flatnonzero(x_values <= x_extreme + tolerance)
    y_target = float(np.median(contour_xy[candidate_idx, 1]))
    local_idx = int(np.argmin(np.abs(contour_xy[candidate_idx, 1] - y_target)))
    return int(candidate_idx[local_idx])


def collect_branch_points(contour_xy: np.ndarray, tip_idx: int, direction: int, target_length: float) -> np.ndarray:
    n_points = len(contour_xy)
    indices = [tip_idx]
    total_length = 0.0
    current_idx = tip_idx
    while total_length < target_length:
        next_idx = (current_idx + direction) % n_points
        step = float(np.linalg.norm(contour_xy[next_idx] - contour_xy[current_idx]))
        total_length += step
        indices.append(next_idx)
        current_idx = next_idx
        if current_idx == tip_idx:
            break
    return contour_xy[np.asarray(indices, dtype=np.int32)]


def branch_direction(branch_xy: np.ndarray, tip_xy: np.ndarray) -> np.ndarray:
    centered = branch_xy - branch_xy.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    outward = branch_xy.mean(axis=0) - tip_xy
    if float(np.dot(direction, outward)) < 0.0:
        direction = -direction
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Degenerate branch direction.")
    return direction / norm


def apex_angle_deg(points_xy: np.ndarray, contour_xy: np.ndarray) -> float:
    tip_is_max_x = select_tip_end(points_xy)
    tip_idx = select_tip_index(contour_xy, tip_is_max_x)
    tip_xy = contour_xy[tip_idx]
    perimeter = contour_perimeter(contour_xy)
    x_span = float(points_xy[:, 0].max() - points_xy[:, 0].min())
    branch_len = min(max(0.05 * perimeter, 0.08 * x_span), 0.18 * perimeter)

    left_branch = collect_branch_points(contour_xy, tip_idx, direction=-1, target_length=branch_len)
    right_branch = collect_branch_points(contour_xy, tip_idx, direction=1, target_length=branch_len)

    left_dir = branch_direction(left_branch, tip_xy)
    right_dir = branch_direction(right_branch, tip_xy)
    dot = float(np.clip(np.dot(left_dir, right_dir), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))
    return min(angle, 180.0 - angle)


def measure_file(ply_path: str, long_side_pixels: int) -> LeafTraitRow:
    points = load_points(ply_path)
    projected = pca_project(points)

    extents = projected.max(axis=0) - projected.min(axis=0)
    sorted_extents = np.sort(extents)[::-1]
    length = float(sorted_extents[0])
    width = float(sorted_extents[1])

    points_xy = projected[:, :2]
    contour_xy = rasterize_contour(points_xy, long_side_pixels=long_side_pixels)
    perimeter = contour_perimeter(contour_xy)
    apex_angle = apex_angle_deg(points_xy, contour_xy)

    return LeafTraitRow(
        ply_name=os.path.basename(ply_path),
        length_obb_m=length,
        width_obb_m=width,
        apex_angle_deg=apex_angle,
        perimeter_m=perimeter,
    )


def write_txt(rows: list[LeafTraitRow], output_path: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: row.ply_name)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("ply_name\tlength_cm\twidth_cm\tapex_angle_deg\tperimeter_cm\n")
        for row in sorted_rows:
            handle.write(
                f"{row.ply_name}\t{row.length_obb_m * 100.0:.4f}\t{row.width_obb_m * 100.0:.4f}\t"
                f"{row.apex_angle_deg:.4f}\t{row.perimeter_m * 100.0:.4f}\n"
            )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_root in args.datasets:
        dataset_root = os.path.normpath(dataset_root)
        dataset_name = os.path.basename(dataset_root)
        rows = [measure_file(ply_path, long_side_pixels=args.long_side_pixels) for ply_path in iter_ply_files(dataset_root)]
        txt_path = os.path.join(args.output_dir, f"{dataset_name}_length_width_cm_pca.txt")
        write_txt(rows, txt_path)


if __name__ == "__main__":
    main()
