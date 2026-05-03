from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


PAGE_RE = re.compile(r"Page size:\s*([0-9.]+)\s+x\s+([0-9.]+)\s+pts")
PT_TO_CM = 2.54 / 72.0


@dataclass
class ScanTraitRow:
    pdf_name: str
    length_cm: float
    width_cm: float
    apex_angle_deg: float
    perimeter_cm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure leaf traits from 2D scanned PDF leaves.")
    parser.add_argument(
        "--input-dir",
        default="phenotyping/data/2dsaomiao",
        help="Directory containing single-leaf scan PDFs.",
    )
    parser.add_argument(
        "--output-txt",
        default="phenotyping/data/scan2d_leaf_traits.txt",
        help="Output TXT path.",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=200,
        help="Rasterization DPI for PDF rendering.",
    )
    return parser.parse_args()


def iter_pdfs(input_dir: str) -> Iterable[str]:
    for filename in sorted(os.listdir(input_dir), key=lambda x: int(os.path.splitext(x)[0])):
        if filename.lower().endswith(".pdf"):
            yield os.path.join(input_dir, filename)


def pdf_page_size_cm(pdf_path: str) -> tuple[float, float]:
    result = subprocess.run(
        ["pdfinfo", pdf_path],
        check=True,
        capture_output=True,
        text=True,
    )
    match = PAGE_RE.search(result.stdout)
    if match is None:
        raise ValueError(f"Failed to parse page size from pdfinfo: {pdf_path}")
    width_pts = float(match.group(1))
    height_pts = float(match.group(2))
    return width_pts * PT_TO_CM, height_pts * PT_TO_CM


def render_pdf(pdf_path: str, output_png: str, dpi: int) -> None:
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf_path, os.path.splitext(output_png)[0]],
        check=True,
    )


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


def pca_project(points_xy: np.ndarray) -> np.ndarray:
    _, _, projected = pca_frame(points_xy)
    return projected


def pca_frame(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points_xy.mean(axis=0)
    centered = points_xy - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    projected = centered @ eigenvectors
    return center, eigenvectors, projected


def extract_leaf_contour(img: np.ndarray, scale_x_cm: float, scale_y_cm: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Remove page border and scan artifacts near the margins.
    crop_y0 = int(h * 0.05)
    crop_y1 = int(h * 0.95)
    crop_x0 = int(w * 0.05)
    crop_x1 = int(w * 0.95)
    crop = gray[crop_y0:crop_y1, crop_x0:crop_x1]

    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n_labels <= 1:
        raise ValueError("No foreground component found in scanned PDF.")

    components = []
    for idx in range(1, n_labels):
        x, y, comp_w, comp_h, area = stats[idx]
        if area < 500:
            continue
        components.append((int(area), idx, x, y, comp_w, comp_h))
    if not components:
        raise ValueError("No valid leaf component found in scanned PDF.")

    _, best_idx, _, _, _, _ = max(components, key=lambda item: item[0])
    leaf_mask = np.where(labels == best_idx, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    contour[:, 0] += crop_x0
    contour[:, 1] += crop_y0

    contour_cm = np.column_stack([contour[:, 0] * scale_x_cm, contour[:, 1] * scale_y_cm])
    contour_cm = smooth_contour(contour_cm, window=11)

    ys, xs = np.where(leaf_mask > 0)
    pixel_points_cm = np.column_stack([(xs + crop_x0) * scale_x_cm, (ys + crop_y0) * scale_y_cm]).astype(np.float64)
    return pixel_points_cm, contour_cm


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


def apex_geometry_for_tip(
    points_xy: np.ndarray, contour_xy: np.ndarray, tip_is_max_x: bool
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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
    angle = min(angle, 180.0 - angle)

    ray_len = max(0.15 * x_span, 0.8)
    left_ray = np.vstack([tip_xy, tip_xy + left_dir * ray_len])
    right_ray = np.vstack([tip_xy, tip_xy + right_dir * ray_len])
    return angle, tip_xy, left_ray, right_ray


def apex_angle_deg(points_xy: np.ndarray, contour_xy: np.ndarray) -> float:
    tip_is_max_x = select_tip_end(points_xy)
    angle, _, _, _ = apex_geometry_for_tip(points_xy, contour_xy, tip_is_max_x)
    return angle


def measure_pdf(pdf_path: str, dpi: int) -> ScanTraitRow:
    with tempfile.TemporaryDirectory(prefix="scan2d_") as tmp_dir:
        png_path = os.path.join(tmp_dir, "page.png")
        render_pdf(pdf_path, png_path, dpi=dpi)
        img = cv2.imread(png_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read rendered PNG for {pdf_path}")

        page_w_cm, page_h_cm = pdf_page_size_cm(pdf_path)
        img_h, img_w = img.shape[:2]

        option1 = (page_w_cm, page_h_cm)
        option2 = (page_h_cm, page_w_cm)
        img_ratio = float(img_w) / float(img_h)
        opt1_ratio = option1[0] / option1[1]
        opt2_ratio = option2[0] / option2[1]
        page_x_cm, page_y_cm = option1 if abs(opt1_ratio - img_ratio) <= abs(opt2_ratio - img_ratio) else option2

        scale_x_cm = page_x_cm / float(img_w)
        scale_y_cm = page_y_cm / float(img_h)

        mask_points_cm, contour_cm = extract_leaf_contour(img, scale_x_cm=scale_x_cm, scale_y_cm=scale_y_cm)
        center, eigenvectors, projected = pca_frame(mask_points_cm)
        contour_projected = (contour_cm - center) @ eigenvectors
        extents = projected.max(axis=0) - projected.min(axis=0)
        sorted_extents = np.sort(extents)[::-1]

        return ScanTraitRow(
            pdf_name=os.path.basename(pdf_path),
            length_cm=float(sorted_extents[0]),
            width_cm=float(sorted_extents[1]),
            apex_angle_deg=apex_angle_deg(projected, contour_projected),
            perimeter_cm=contour_perimeter(contour_cm),
        )


def write_txt(rows: list[ScanTraitRow], output_txt: str) -> None:
    rows = sorted(rows, key=lambda row: int(os.path.splitext(row.pdf_name)[0]))
    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write("pdf_name\tlength_cm\twidth_cm\tapex_angle_deg\tperimeter_cm\n")
        for row in rows:
            handle.write(
                f"{row.pdf_name}\t{row.length_cm:.4f}\t{row.width_cm:.4f}\t{row.apex_angle_deg:.4f}\t{row.perimeter_cm:.4f}\n"
            )


def main() -> None:
    args = parse_args()
    rows = [measure_pdf(pdf_path, dpi=args.render_dpi) for pdf_path in iter_pdfs(args.input_dir)]
    write_txt(rows, args.output_txt)


if __name__ == "__main__":
    main()
