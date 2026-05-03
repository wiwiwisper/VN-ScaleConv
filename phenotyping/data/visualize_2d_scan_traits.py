from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for path in [SCRIPT_DIR, REPO_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from measure_2d_scan_traits import (
    apex_angle_deg,
    branch_direction,
    collect_branch_points,
    contour_perimeter,
    extract_leaf_contour,
    iter_pdfs,
    pca_frame,
    pdf_page_size_cm,
    render_pdf,
    select_tip_end,
    select_tip_index,
)


@dataclass
class VisualTraitRow:
    pdf_name: str
    length_cm: float
    width_cm: float
    apex_angle_deg: float
    perimeter_cm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create annotated visualizations for 2D scanned leaf traits.")
    parser.add_argument(
        "--input-dir",
        default="phenotyping/data/2dsaomiao",
        help="Directory containing scanned leaf PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default="phenotyping/data/2dsaomiao_measure",
        help="Directory for annotated output images.",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=200,
        help="Rasterization DPI for PDF rendering.",
    )
    return parser.parse_args()


def resolve_page_axes_cm(page_w_cm: float, page_h_cm: float, img_w: int, img_h: int) -> tuple[float, float]:
    option1 = (page_w_cm, page_h_cm)
    option2 = (page_h_cm, page_w_cm)
    img_ratio = float(img_w) / float(img_h)
    opt1_ratio = option1[0] / option1[1]
    opt2_ratio = option2[0] / option2[1]
    return option1 if abs(opt1_ratio - img_ratio) <= abs(opt2_ratio - img_ratio) else option2


def extract_leaf_contour_with_pixels(
    img: np.ndarray, scale_x_cm: float, scale_y_cm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask_points_cm, contour_cm = extract_leaf_contour(img, scale_x_cm=scale_x_cm, scale_y_cm=scale_y_cm)
    contour_px = np.column_stack([contour_cm[:, 0] / scale_x_cm, contour_cm[:, 1] / scale_y_cm]).astype(np.float64)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
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
    ys, xs = np.where(leaf_mask > 0)
    mask_points_px = np.column_stack([xs + crop_x0, ys + crop_y0]).astype(np.float64)
    return mask_points_px, mask_points_cm, contour_px, contour_cm


def cm_to_px(points_cm: np.ndarray, scale_x_cm: float, scale_y_cm: float) -> np.ndarray:
    pts = np.asarray(points_cm, dtype=np.float64)
    x = pts[:, 0] / scale_x_cm
    y = pts[:, 1] / scale_y_cm
    return np.column_stack([x, y])


def obb_geometry(mask_points_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center, eigenvectors, projected = pca_frame(mask_points_cm)
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extents = maxs - mins
    major_idx = int(np.argmax(extents))
    minor_idx = 1 - major_idx

    corners_proj = np.array(
        [
            [mins[0], mins[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]],
            [mins[0], maxs[1]],
        ],
        dtype=np.float64,
    )
    corners_cm = corners_proj @ eigenvectors.T + center

    center_proj = (mins + maxs) / 2.0
    length_line_proj = np.vstack(
        [
            center_proj - np.eye(2)[major_idx] * extents[major_idx] / 2.0,
            center_proj + np.eye(2)[major_idx] * extents[major_idx] / 2.0,
        ]
    )
    width_line_proj = np.vstack(
        [
            center_proj - np.eye(2)[minor_idx] * extents[minor_idx] / 2.0,
            center_proj + np.eye(2)[minor_idx] * extents[minor_idx] / 2.0,
        ]
    )
    length_line_cm = length_line_proj @ eigenvectors.T + center
    width_line_cm = width_line_proj @ eigenvectors.T + center
    return center, eigenvectors, projected, corners_cm, np.vstack([length_line_cm, width_line_cm])


def tip_geometry(mask_points_cm: np.ndarray, contour_cm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    center, eigenvectors, projected = pca_frame(mask_points_cm)
    contour_projected = (contour_cm - center) @ eigenvectors
    tip_is_max_x = select_tip_end(projected)
    tip_idx = select_tip_index(contour_projected, tip_is_max_x)
    tip_proj = contour_projected[tip_idx]
    perimeter = contour_perimeter(contour_projected)
    x_span = float(projected[:, 0].max() - projected[:, 0].min())
    branch_len = min(max(0.05 * perimeter, 0.08 * x_span), 0.18 * perimeter)

    left_branch = collect_branch_points(contour_projected, tip_idx, direction=-1, target_length=branch_len)
    right_branch = collect_branch_points(contour_projected, tip_idx, direction=1, target_length=branch_len)
    left_dir = branch_direction(left_branch, tip_proj)
    right_dir = branch_direction(right_branch, tip_proj)

    angle = apex_angle_deg(projected, contour_projected)
    ray_len = max(0.15 * x_span, 0.8)
    left_ray_proj = np.vstack([tip_proj, tip_proj + left_dir * ray_len])
    right_ray_proj = np.vstack([tip_proj, tip_proj + right_dir * ray_len])
    left_ray_cm = left_ray_proj @ eigenvectors.T + center
    right_ray_cm = right_ray_proj @ eigenvectors.T + center
    tip_cm = tip_proj @ eigenvectors.T + center
    return angle, tip_cm, left_ray_cm, right_ray_cm


def draw_text_block(image: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x0, y0 = origin
    line_h = 28
    box_w = max(420, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for line in lines) + 24)
    box_h = line_h * len(lines) + 18
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    for idx, line in enumerate(lines):
        y = y0 + 30 + idx * line_h
        cv2.putText(image, line, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)


def crop_with_padding(image: np.ndarray, points_px: np.ndarray, pad: int = 60) -> tuple[np.ndarray, tuple[int, int]]:
    pts = np.asarray(points_px, dtype=np.float64)
    x0 = max(int(np.floor(pts[:, 0].min())) - pad, 0)
    y0 = max(int(np.floor(pts[:, 1].min())) - pad, 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + pad, image.shape[1] - 1)
    y1 = min(int(np.ceil(pts[:, 1].max())) + pad, image.shape[0] - 1)
    cropped = image[y0 : y1 + 1, x0 : x1 + 1].copy()
    return cropped, (x0, y0)


def shift_points(points_px: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float64).copy()
    pts[:, 0] -= offset[0]
    pts[:, 1] -= offset[1]
    return pts


def annotate_pdf(pdf_path: str, output_path: str, dpi: int) -> VisualTraitRow:
    with tempfile.TemporaryDirectory(prefix="scan2d_vis_") as tmp_dir:
        png_path = os.path.join(tmp_dir, "page.png")
        render_pdf(pdf_path, png_path, dpi=dpi)
        image = cv2.imread(png_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read rendered PNG for {pdf_path}")

        page_w_cm, page_h_cm = pdf_page_size_cm(pdf_path)
        img_h, img_w = image.shape[:2]
        page_x_cm, page_y_cm = resolve_page_axes_cm(page_w_cm, page_h_cm, img_w, img_h)
        scale_x_cm = page_x_cm / float(img_w)
        scale_y_cm = page_y_cm / float(img_h)

        _mask_points_px, mask_points_cm, contour_px, contour_cm = extract_leaf_contour_with_pixels(
            image, scale_x_cm=scale_x_cm, scale_y_cm=scale_y_cm
        )

        _, _, projected, corners_cm, axis_lines_cm = obb_geometry(mask_points_cm)
        extents = projected.max(axis=0) - projected.min(axis=0)
        sorted_extents = np.sort(extents)[::-1]
        length_cm = float(sorted_extents[0])
        width_cm = float(sorted_extents[1])
        perimeter_cm = contour_perimeter(contour_cm)
        apex_angle, tip_cm, left_ray_cm, right_ray_cm = tip_geometry(mask_points_cm, contour_cm)

        corners_px = cm_to_px(corners_cm, scale_x_cm, scale_y_cm)
        axis_lines_px = cm_to_px(axis_lines_cm, scale_x_cm, scale_y_cm)
        tip_px = cm_to_px(tip_cm.reshape(1, 2), scale_x_cm, scale_y_cm)[0]
        left_ray_px = cm_to_px(left_ray_cm, scale_x_cm, scale_y_cm)
        right_ray_px = cm_to_px(right_ray_cm, scale_x_cm, scale_y_cm)

        all_points_px = np.vstack([contour_px, corners_px, left_ray_px, right_ray_px, tip_px.reshape(1, 2)])
        canvas, offset = crop_with_padding(image, all_points_px, pad=70)

        contour_draw = np.round(shift_points(contour_px, offset)).astype(np.int32).reshape(-1, 1, 2)
        corners_draw = np.round(shift_points(corners_px, offset)).astype(np.int32)
        axis_draw = np.round(shift_points(axis_lines_px, offset)).astype(np.int32)
        tip_draw = np.round(shift_points(tip_px.reshape(1, 2), offset)).astype(np.int32)[0]
        left_ray_draw = np.round(shift_points(left_ray_px, offset)).astype(np.int32)
        right_ray_draw = np.round(shift_points(right_ray_px, offset)).astype(np.int32)

        cv2.drawContours(canvas, [contour_draw], -1, (30, 144, 255), 3)
        cv2.polylines(canvas, [corners_draw.reshape(-1, 1, 2)], True, (0, 180, 0), 3)
        cv2.line(canvas, tuple(axis_draw[0]), tuple(axis_draw[1]), (0, 0, 255), 3)
        cv2.line(canvas, tuple(axis_draw[2]), tuple(axis_draw[3]), (180, 0, 180), 3)
        cv2.line(canvas, tuple(left_ray_draw[0]), tuple(left_ray_draw[1]), (255, 140, 0), 3)
        cv2.line(canvas, tuple(right_ray_draw[0]), tuple(right_ray_draw[1]), (255, 140, 0), 3)
        cv2.circle(canvas, tuple(tip_draw), 10, (255, 140, 0), -1)

        text_lines = [
            os.path.basename(pdf_path),
            f"length_cm = {length_cm:.4f}",
            f"width_cm = {width_cm:.4f}",
            f"apex_angle_deg = {apex_angle:.4f}",
            f"perimeter_cm = {perimeter_cm:.4f}",
        ]
        draw_text_block(canvas, text_lines, origin=(18, 18))
        cv2.imwrite(output_path, canvas)

        return VisualTraitRow(
            pdf_name=os.path.basename(pdf_path),
            length_cm=length_cm,
            width_cm=width_cm,
            apex_angle_deg=float(apex_angle),
            perimeter_cm=perimeter_cm,
        )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for pdf_path in iter_pdfs(args.input_dir):
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        annotate_pdf(pdf_path, os.path.join(args.output_dir, f"{stem}.png"), dpi=args.render_dpi)


if __name__ == "__main__":
    main()
