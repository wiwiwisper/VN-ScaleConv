from __future__ import annotations

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for path in [SCRIPT_DIR, REPO_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from measure_2d_scan_traits import (
    apex_geometry_for_tip,
    contour_perimeter,
    extract_leaf_contour,
    iter_pdfs,
    pca_frame,
    pdf_page_size_cm,
    render_pdf,
)
from visualize_2d_scan_traits import crop_with_padding, draw_text_block, resolve_page_axes_cm, shift_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare apex-angle review images.")
    parser.add_argument(
        "--input-dir",
        default="phenotyping/data/2dsaomiao",
        help="Directory containing scanned leaf PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        default="phenotyping/data/final/apex_review",
        help="Directory for review images.",
    )
    parser.add_argument(
        "--output-txt",
        default="phenotyping/data/final/apex_angle_review.txt",
        help="Output TXT with review rows.",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=200,
        help="Rasterization DPI for PDF rendering.",
    )
    return parser.parse_args()


def cm_to_px(points_cm: np.ndarray, scale_x_cm: float, scale_y_cm: float) -> np.ndarray:
    points_cm = np.asarray(points_cm, dtype=np.float64)
    return np.column_stack([points_cm[:, 0] / scale_x_cm, points_cm[:, 1] / scale_y_cm])


def obb_geometry(mask_points_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center, eigenvectors, projected = pca_frame(mask_points_cm)
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extents = maxs - mins
    major_idx = int(np.argmax(extents))
    minor_idx = 1 - major_idx

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
    return center, eigenvectors, projected, np.vstack([length_line_cm, width_line_cm])


def measure_review_row(pdf_path: str, dpi: int) -> tuple[dict[str, float | str], np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="final_apex_review_") as tmp_dir:
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

        mask_points_cm, contour_cm = extract_leaf_contour(image, scale_x_cm=scale_x_cm, scale_y_cm=scale_y_cm)
        center, eigenvectors, projected, axis_lines_cm = obb_geometry(mask_points_cm)
        contour_projected = (contour_cm - center) @ eigenvectors
        extents = projected.max(axis=0) - projected.min(axis=0)
        sorted_extents = np.sort(extents)[::-1]
        perimeter_cm = contour_perimeter(contour_cm)

        angle_1, tip_1_proj, left_1_proj, right_1_proj = apex_geometry_for_tip(projected, contour_projected, tip_is_max_x=False)
        angle_2, tip_2_proj, left_2_proj, right_2_proj = apex_geometry_for_tip(projected, contour_projected, tip_is_max_x=True)

        contour_px = cm_to_px(contour_cm, scale_x_cm, scale_y_cm)
        axis_lines_px = cm_to_px(axis_lines_cm, scale_x_cm, scale_y_cm)
        tip_1_px = cm_to_px((tip_1_proj @ eigenvectors.T + center).reshape(1, 2), scale_x_cm, scale_y_cm)[0]
        tip_2_px = cm_to_px((tip_2_proj @ eigenvectors.T + center).reshape(1, 2), scale_x_cm, scale_y_cm)[0]
        left_1_px = cm_to_px(left_1_proj @ eigenvectors.T + center, scale_x_cm, scale_y_cm)
        right_1_px = cm_to_px(right_1_proj @ eigenvectors.T + center, scale_x_cm, scale_y_cm)
        left_2_px = cm_to_px(left_2_proj @ eigenvectors.T + center, scale_x_cm, scale_y_cm)
        right_2_px = cm_to_px(right_2_proj @ eigenvectors.T + center, scale_x_cm, scale_y_cm)

        all_points_px = np.vstack(
            [
                contour_px,
                axis_lines_px,
                tip_1_px.reshape(1, 2),
                tip_2_px.reshape(1, 2),
                left_1_px,
                right_1_px,
                left_2_px,
                right_2_px,
            ]
        )
        canvas, offset = crop_with_padding(image, all_points_px, pad=70)

        contour_draw = np.round(shift_points(contour_px, offset)).astype(np.int32).reshape(-1, 1, 2)
        axis_draw = np.round(shift_points(axis_lines_px, offset)).astype(np.int32)
        tip_1_draw = np.round(shift_points(tip_1_px.reshape(1, 2), offset)).astype(np.int32)[0]
        tip_2_draw = np.round(shift_points(tip_2_px.reshape(1, 2), offset)).astype(np.int32)[0]
        left_1_draw = np.round(shift_points(left_1_px, offset)).astype(np.int32)
        right_1_draw = np.round(shift_points(right_1_px, offset)).astype(np.int32)
        left_2_draw = np.round(shift_points(left_2_px, offset)).astype(np.int32)
        right_2_draw = np.round(shift_points(right_2_px, offset)).astype(np.int32)

        cv2.drawContours(canvas, [contour_draw], -1, (40, 120, 255), 3)
        cv2.line(canvas, tuple(axis_draw[0]), tuple(axis_draw[1]), (0, 0, 255), 3)
        cv2.line(canvas, tuple(axis_draw[2]), tuple(axis_draw[3]), (170, 0, 170), 3)

        color_1 = (255, 140, 0)
        color_2 = (255, 0, 180)
        cv2.line(canvas, tuple(left_1_draw[0]), tuple(left_1_draw[1]), color_1, 3)
        cv2.line(canvas, tuple(right_1_draw[0]), tuple(right_1_draw[1]), color_1, 3)
        cv2.line(canvas, tuple(left_2_draw[0]), tuple(left_2_draw[1]), color_2, 3)
        cv2.line(canvas, tuple(right_2_draw[0]), tuple(right_2_draw[1]), color_2, 3)

        cv2.circle(canvas, tuple(tip_1_draw), 9, color_1, -1)
        cv2.circle(canvas, tuple(tip_2_draw), 9, color_2, -1)
        cv2.putText(canvas, f"1 ({angle_1:.2f})", (tip_1_draw[0] + 10, tip_1_draw[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color_1, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"2 ({angle_2:.2f})", (tip_2_draw[0] + 10, tip_2_draw[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color_2, 3, cv2.LINE_AA)

        text_lines = [
            os.path.basename(pdf_path),
            f"length_cm = {sorted_extents[0]:.3f}",
            f"width_cm = {sorted_extents[1]:.3f}",
            f"perimeter_cm = {perimeter_cm:.3f}",
            f"angle_1_deg = {angle_1:.3f}",
            f"angle_2_deg = {angle_2:.3f}",
        ]
        draw_text_block(canvas, text_lines, origin=(18, 18))

        summary = {
            "pdf_name": os.path.basename(pdf_path),
            "length_cm": float(sorted_extents[0]),
            "width_cm": float(sorted_extents[1]),
            "perimeter_cm": float(perimeter_cm),
            "angle_1_deg": float(angle_1),
            "angle_2_deg": float(angle_2),
        }
        return summary, canvas


def write_review_txt(rows: list[dict[str, float | str]], output_txt: str) -> None:
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write("pdf_name\tlength_cm\twidth_cm\tperimeter_cm\tangle_1_deg\tangle_2_deg\tselection\n")
        for row in rows:
            handle.write(
                f"{row['pdf_name']}\t{row['length_cm']:.4f}\t{row['width_cm']:.4f}\t{row['perimeter_cm']:.4f}\t"
                f"{row['angle_1_deg']:.4f}\t{row['angle_2_deg']:.4f}\t\n"
            )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for pdf_path in iter_pdfs(args.input_dir):
        summary, canvas = measure_review_row(pdf_path, dpi=args.render_dpi)
        leaf_id = os.path.splitext(os.path.basename(pdf_path))[0]
        cv2.imwrite(os.path.join(args.output_dir, f"{leaf_id}.png"), canvas)
        rows.append(summary)

    rows.sort(key=lambda row: int(os.path.splitext(str(row["pdf_name"]))[0]))
    write_review_txt(rows, args.output_txt)


if __name__ == "__main__":
    main()
