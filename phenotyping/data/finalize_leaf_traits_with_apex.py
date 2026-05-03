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

from build_final_gt_dataset import (
    FinalLeafRow,
    cm_to_px,
    obb_from_mask,
    pdf_scale_cm,
    read_vein_lengths_cm,
)
from measure_2d_scan_traits import (
    apex_geometry_for_tip,
    extract_leaf_contour,
    pca_frame,
    render_pdf,
)
from visualize_2d_scan_traits import crop_with_padding, shift_points


@dataclass
class FinalLeafRowWithApex:
    leaf_id: int
    pdf_name: str
    length_cm: float
    width_cm: float
    perimeter_cm: float
    vein_length_cm: float
    area_cm2: float
    apex_angle_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize 90-leaf GT table and visualizations with selected apex angle.")
    parser.add_argument(
        "--gt-txt",
        default="phenotyping/data/final/gt_leaf_traits_5params.txt",
        help="Existing 5-parameter GT TXT.",
    )
    parser.add_argument(
        "--apex-choice-txt",
        default="phenotyping/data/final/apex_angle_review.txt",
        help="TXT with apex selection values.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="phenotyping/data/2dsaomiao",
        help="Directory containing scanned leaf PDFs.",
    )
    parser.add_argument(
        "--vein-zip",
        default="phenotyping/data/yemai_length/2dsaomiao_png(1).zip",
        help="ZIP containing vein polyline annotations.",
    )
    parser.add_argument(
        "--output-txt",
        default="phenotyping/data/final/gt_leaf_traits_6params_90.txt",
        help="Final 90-leaf GT TXT with apex angle.",
    )
    parser.add_argument(
        "--visual-dir",
        default="phenotyping/data/final/visual",
        help="Directory for final 90-leaf visualization PNGs.",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=200,
        help="Rasterization DPI for PDF rendering.",
    )
    return parser.parse_args()


def read_gt_rows(path: str) -> dict[int, FinalLeafRow]:
    rows: dict[int, FinalLeafRow] = {}
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        expected = ["leaf_id", "pdf_name", "length_cm", "width_cm", "perimeter_cm", "vein_length_cm", "area_cm2"]
        if header != expected:
            raise ValueError(f"Unexpected header in {path}: {header}")
        for line in handle:
            if not line.strip():
                continue
            leaf_id, pdf_name, length_cm, width_cm, perimeter_cm, vein_length_cm, area_cm2 = line.strip().split("\t")
            idx = int(leaf_id)
            rows[idx] = FinalLeafRow(
                leaf_id=idx,
                pdf_name=pdf_name,
                length_cm=float(length_cm),
                width_cm=float(width_cm),
                perimeter_cm=float(perimeter_cm),
                vein_length_cm=float(vein_length_cm),
                area_cm2=float(area_cm2),
            )
    return rows


def read_apex_choices(path: str, max_leaf_id: int = 90) -> dict[int, tuple[float, float, str]]:
    rows: dict[int, tuple[float, float, str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        expected = ["pdf_name", "length_cm", "width_cm", "perimeter_cm", "angle_1_deg", "angle_2_deg", "selection"]
        if header != expected:
            raise ValueError(f"Unexpected header in {path}: {header}")
        for line in handle:
            if not line.strip():
                continue
            pdf_name, _length_cm, _width_cm, _perimeter_cm, angle_1_deg, angle_2_deg, choice = line.rstrip("\n").split("\t")
            leaf_id = int(os.path.splitext(pdf_name)[0])
            if leaf_id > max_leaf_id:
                continue
            choice = choice.strip()
            if choice not in {"1", "2"}:
                raise ValueError(f"Leaf {leaf_id} missing valid chosen tip in {path}")
            rows[leaf_id] = (float(angle_1_deg), float(angle_2_deg), choice)
    if len(rows) != max_leaf_id:
        missing = sorted(set(range(1, max_leaf_id + 1)) - set(rows))
        raise ValueError(f"Missing apex choices for leaves: {missing}")
    return rows


def build_final_rows(gt_rows: dict[int, FinalLeafRow], apex_choices: dict[int, tuple[float, float, str]]) -> list[FinalLeafRowWithApex]:
    rows: list[FinalLeafRowWithApex] = []
    for leaf_id in range(1, 91):
        base = gt_rows[leaf_id]
        angle_1_deg, angle_2_deg, choice = apex_choices[leaf_id]
        apex_angle_deg = angle_1_deg if choice == "1" else angle_2_deg
        rows.append(
            FinalLeafRowWithApex(
                leaf_id=base.leaf_id,
                pdf_name=base.pdf_name,
                length_cm=base.length_cm,
                width_cm=base.width_cm,
                perimeter_cm=base.perimeter_cm,
                vein_length_cm=base.vein_length_cm,
                area_cm2=base.area_cm2,
                apex_angle_deg=apex_angle_deg,
            )
        )
    return rows


def write_txt(rows: list[FinalLeafRowWithApex], output_txt: str) -> None:
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write("leaf_id\tpdf_name\tlength_cm\twidth_cm\tperimeter_cm\tvein_length_cm\tarea_cm2\tapex_angle_deg\n")
        for row in rows:
            handle.write(
                f"{row.leaf_id}\t{row.pdf_name}\t{row.length_cm:.4f}\t{row.width_cm:.4f}\t{row.perimeter_cm:.4f}\t"
                f"{row.vein_length_cm:.4f}\t{row.area_cm2:.4f}\t{row.apex_angle_deg:.4f}\n"
            )


def apex_geometry_for_choice(mask_points_cm: np.ndarray, contour_cm: np.ndarray, choose_tip: str):
    center, eigenvectors, projected = pca_frame(mask_points_cm)
    contour_projected = (contour_cm - center) @ eigenvectors
    tip_is_max_x = choose_tip == "2"
    angle_deg, tip_proj, left_ray_proj, right_ray_proj = apex_geometry_for_tip(projected, contour_projected, tip_is_max_x=tip_is_max_x)
    tip_cm = tip_proj @ eigenvectors.T + center
    left_ray_cm = left_ray_proj @ eigenvectors.T + center
    right_ray_cm = right_ray_proj @ eigenvectors.T + center
    return angle_deg, tip_cm, left_ray_cm, right_ray_cm


def obb_corners_from_mask(mask_points_cm: np.ndarray) -> np.ndarray:
    center, eigenvectors, projected = pca_frame(mask_points_cm)
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    corners_proj = np.array(
        [
            [mins[0], mins[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]],
            [mins[0], maxs[1]],
        ],
        dtype=np.float64,
    )
    return corners_proj @ eigenvectors.T + center


def draw_small_text_block(image: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x0, y0 = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.36
    thickness = 1
    line_h = 16
    padding = 8
    box_w = max(170, max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + padding * 2)
    box_h = line_h * len(lines) + padding * 2 + 2
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)
    for idx, line in enumerate(lines):
        y = y0 + padding + 10 + idx * line_h
        cv2.putText(image, line, (x0 + padding, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def annotate_leaf(
    pdf_path: str,
    output_path: str,
    row: FinalLeafRowWithApex,
    choose_tip: str,
    vein_info: dict[str, np.ndarray | int],
    dpi: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="final_apex_vis_") as tmp_dir:
        png_path = os.path.join(tmp_dir, "page.png")
        render_pdf(pdf_path, png_path, dpi=dpi)
        image = cv2.imread(png_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read rendered PNG for {pdf_path}")

        scale_x_cm, scale_y_cm = pdf_scale_cm(pdf_path, image.shape[1], image.shape[0])
        mask_points_cm, contour_cm = extract_leaf_contour(image, scale_x_cm=scale_x_cm, scale_y_cm=scale_y_cm)
        contour_px = cm_to_px(contour_cm, scale_x_cm, scale_y_cm)
        length_line_cm, width_line_cm = obb_from_mask(mask_points_cm)
        obb_corners_cm = obb_corners_from_mask(mask_points_cm)
        length_line_px = cm_to_px(length_line_cm, scale_x_cm, scale_y_cm)
        width_line_px = cm_to_px(width_line_cm, scale_x_cm, scale_y_cm)
        obb_corners_px = cm_to_px(obb_corners_cm, scale_x_cm, scale_y_cm)

        _angle_deg, tip_cm, left_ray_cm, right_ray_cm = apex_geometry_for_choice(mask_points_cm, contour_cm, choose_tip)
        tip_px = cm_to_px(np.asarray(tip_cm).reshape(1, 2), scale_x_cm, scale_y_cm)[0]
        left_ray_px = cm_to_px(left_ray_cm, scale_x_cm, scale_y_cm)
        right_ray_px = cm_to_px(right_ray_cm, scale_x_cm, scale_y_cm)

        source_vein_px = np.asarray(vein_info["points_px"], dtype=np.float64)
        src_w = int(vein_info["image_width"])
        src_h = int(vein_info["image_height"])
        vein_polyline_px = source_vein_px.copy()
        vein_polyline_px[:, 0] *= float(image.shape[1]) / float(src_w)
        vein_polyline_px[:, 1] *= float(image.shape[0]) / float(src_h)

        overlay = image.copy()
        contour_fill = np.round(contour_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [contour_fill], color=(220, 245, 220))
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0, image)

        all_points_px = np.vstack(
            [contour_px, obb_corners_px, length_line_px, width_line_px, vein_polyline_px, left_ray_px, right_ray_px, tip_px.reshape(1, 2)]
        )
        canvas, offset = crop_with_padding(image, all_points_px, pad=60)

        contour_draw = np.round(shift_points(contour_px, offset)).astype(np.int32).reshape(-1, 1, 2)
        obb_draw = np.round(shift_points(obb_corners_px, offset)).astype(np.int32).reshape(-1, 1, 2)
        length_draw = np.round(shift_points(length_line_px, offset)).astype(np.int32)
        width_draw = np.round(shift_points(width_line_px, offset)).astype(np.int32)
        vein_draw = np.round(shift_points(vein_polyline_px, offset)).astype(np.int32).reshape(-1, 1, 2)
        left_ray_draw = np.round(shift_points(left_ray_px, offset)).astype(np.int32)
        right_ray_draw = np.round(shift_points(right_ray_px, offset)).astype(np.int32)
        tip_draw = np.round(shift_points(tip_px.reshape(1, 2), offset)).astype(np.int32)[0]

        cv2.drawContours(canvas, [contour_draw], -1, (45, 120, 235), 3)
        cv2.polylines(canvas, [obb_draw], True, (0, 170, 120), 3)
        cv2.line(canvas, tuple(length_draw[0]), tuple(length_draw[1]), (20, 20, 220), 3)
        cv2.line(canvas, tuple(width_draw[0]), tuple(width_draw[1]), (150, 30, 150), 3)
        cv2.polylines(canvas, [vein_draw], False, (20, 145, 40), 3)
        cv2.line(canvas, tuple(left_ray_draw[0]), tuple(left_ray_draw[1]), (255, 140, 0), 3)
        cv2.line(canvas, tuple(right_ray_draw[0]), tuple(right_ray_draw[1]), (255, 140, 0), 3)
        cv2.circle(canvas, tuple(tip_draw), 6, (255, 180, 0), -1)

        text_lines = [
            row.pdf_name,
            f"L = {row.length_cm:.2f} cm",
            f"W = {row.width_cm:.2f} cm",
            f"P = {row.perimeter_cm:.2f} cm",
            f"Vein = {row.vein_length_cm:.2f} cm",
            f"Area = {row.area_cm2:.2f} cm2",
            f"Apex = {row.apex_angle_deg:.2f} deg",
        ]
        draw_small_text_block(canvas, text_lines, origin=(10, 10))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)


def main() -> None:
    args = parse_args()
    gt_rows = read_gt_rows(args.gt_txt)
    apex_choices = read_apex_choices(args.apex_choice_txt, max_leaf_id=90)
    rows = build_final_rows(gt_rows, apex_choices)
    write_txt(rows, args.output_txt)

    _vein_lengths_cm, vein_meta = read_vein_lengths_cm(args.vein_zip, args.pdf_dir)
    os.makedirs(args.visual_dir, exist_ok=True)

    for row in rows:
        pdf_path = os.path.join(args.pdf_dir, row.pdf_name)
        _, _, choice = apex_choices[row.leaf_id]
        output_path = os.path.join(args.visual_dir, f"{row.leaf_id}.png")
        annotate_leaf(pdf_path, output_path, row, choice, vein_meta[row.leaf_id], dpi=args.render_dpi)


if __name__ == "__main__":
    main()
