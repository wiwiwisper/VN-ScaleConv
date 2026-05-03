from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for path in [SCRIPT_DIR, REPO_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from measure_2d_scan_traits import (
    contour_perimeter,
    extract_leaf_contour,
    iter_pdfs,
    pca_frame,
    pdf_page_size_cm,
    render_pdf,
)
from visualize_2d_scan_traits import crop_with_padding, draw_text_block, resolve_page_axes_cm, shift_points


@dataclass
class FinalLeafRow:
    leaf_id: int
    pdf_name: str
    length_cm: float
    width_cm: float
    perimeter_cm: float
    vein_length_cm: float
    area_cm2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final 5-trait GT dataset for 100 scanned leaves.")
    parser.add_argument(
        "--scan-traits-txt",
        default="phenotyping/data/scan2d_leaf_traits.txt",
        help="Existing 2D scan trait TXT containing length/width/perimeter.",
    )
    parser.add_argument(
        "--area-txt",
        default="phenotyping/data/3dgt_gt_leaf_with_lpssmooth_100_02/area_results_with_smooth.txt",
        help="Area GT file from the main experiment.",
    )
    parser.add_argument(
        "--vein-zip",
        default="phenotyping/data/yemai_length/2dsaomiao_png(1).zip",
        help="ZIP containing per-leaf vein polyline annotations.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="phenotyping/data/2dsaomiao",
        help="Directory containing original scan PDFs.",
    )
    parser.add_argument(
        "--output-txt",
        default="phenotyping/data/final/gt_leaf_traits_5params.txt",
        help="Final merged TXT output.",
    )
    parser.add_argument(
        "--visual-dir",
        default="phenotyping/data/final/visual",
        help="Directory for per-leaf visualization PNGs.",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=200,
        help="Rasterization DPI for visualization rendering.",
    )
    return parser.parse_args()


def read_scan_traits(path: str) -> dict[int, dict[str, float | str]]:
    rows: dict[int, dict[str, float | str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        expected = ["pdf_name", "length_cm", "width_cm", "apex_angle_deg", "perimeter_cm"]
        if header != expected:
            raise ValueError(f"Unexpected scan trait header in {path}: {header}")
        for line in handle:
            if not line.strip():
                continue
            pdf_name, length_cm, width_cm, _apex_angle, perimeter_cm = line.strip().split("\t")
            leaf_id = int(os.path.splitext(pdf_name)[0])
            rows[leaf_id] = {
                "pdf_name": pdf_name,
                "length_cm": float(length_cm),
                "width_cm": float(width_cm),
                "perimeter_cm": float(perimeter_cm),
            }
    return rows


def read_area_gt_cm2(path: str) -> dict[int, float]:
    area_cm2: dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            stem, value = line.split(":")
            leaf_id = int(os.path.splitext(stem.strip())[0])
            area_raw = float(value.strip())
            area_cm2[leaf_id] = area_raw / 100.0
    return area_cm2


def pdf_scale_cm(pdf_path: str, img_w: int, img_h: int) -> tuple[float, float]:
    page_w_cm, page_h_cm = pdf_page_size_cm(pdf_path)
    page_x_cm, page_y_cm = resolve_page_axes_cm(page_w_cm, page_h_cm, img_w, img_h)
    return page_x_cm / float(img_w), page_y_cm / float(img_h)


def read_vein_lengths_cm(
    vein_zip_path: str, pdf_dir: str
) -> tuple[dict[int, float], dict[int, dict[str, np.ndarray | int]]]:
    lengths_cm: dict[int, float] = {}
    vein_meta: dict[int, dict[str, np.ndarray | int]] = {}
    with zipfile.ZipFile(vein_zip_path, "r") as archive:
        json_names = sorted(
            [name for name in archive.namelist() if name.endswith(".json")],
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        for json_name in json_names:
            leaf_id = int(os.path.splitext(os.path.basename(json_name))[0])
            data = json.loads(archive.read(json_name))
            shapes = data.get("shapes", [])
            if not shapes:
                raise ValueError(f"No shapes found in vein annotation: {json_name}")
            points_px = np.asarray(shapes[0]["points"], dtype=np.float64)
            if len(points_px) < 2:
                raise ValueError(f"Too few vein points in {json_name}")

            pdf_path = os.path.join(pdf_dir, f"{leaf_id}.pdf")
            scale_x_cm, scale_y_cm = pdf_scale_cm(pdf_path, int(data["imageWidth"]), int(data["imageHeight"]))
            points_cm = np.column_stack([points_px[:, 0] * scale_x_cm, points_px[:, 1] * scale_y_cm])
            length_cm = float(np.linalg.norm(np.diff(points_cm, axis=0), axis=1).sum())

            lengths_cm[leaf_id] = length_cm
            vein_meta[leaf_id] = {
                "points_px": points_px,
                "image_width": int(data["imageWidth"]),
                "image_height": int(data["imageHeight"]),
            }
    return lengths_cm, vein_meta


def build_rows(
    scan_traits: dict[int, dict[str, float | str]], area_gt_cm2: dict[int, float], vein_lengths_cm: dict[int, float]
) -> list[FinalLeafRow]:
    leaf_ids = sorted(scan_traits.keys())
    rows: list[FinalLeafRow] = []
    for leaf_id in leaf_ids:
        if leaf_id not in area_gt_cm2:
            raise KeyError(f"Missing area GT for leaf {leaf_id}")
        if leaf_id not in vein_lengths_cm:
            raise KeyError(f"Missing vein length for leaf {leaf_id}")
        row = scan_traits[leaf_id]
        rows.append(
            FinalLeafRow(
                leaf_id=leaf_id,
                pdf_name=str(row["pdf_name"]),
                length_cm=float(row["length_cm"]),
                width_cm=float(row["width_cm"]),
                perimeter_cm=float(row["perimeter_cm"]),
                vein_length_cm=float(vein_lengths_cm[leaf_id]),
                area_cm2=float(area_gt_cm2[leaf_id]),
            )
        )
    return rows


def write_txt(rows: list[FinalLeafRow], output_txt: str) -> None:
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write("leaf_id\tpdf_name\tlength_cm\twidth_cm\tperimeter_cm\tvein_length_cm\tarea_cm2\n")
        for row in rows:
            handle.write(
                f"{row.leaf_id}\t{row.pdf_name}\t{row.length_cm:.4f}\t{row.width_cm:.4f}\t"
                f"{row.perimeter_cm:.4f}\t{row.vein_length_cm:.4f}\t{row.area_cm2:.4f}\n"
            )


def obb_from_mask(mask_points_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center, eigenvectors, projected = pca_frame(mask_points_cm)
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extents = maxs - mins
    major_idx = int(np.argmax(extents))
    minor_idx = 1 - major_idx

    center_proj = (mins + maxs) / 2.0
    major = np.eye(2)[major_idx] * extents[major_idx] / 2.0
    minor = np.eye(2)[minor_idx] * extents[minor_idx] / 2.0

    length_line_proj = np.vstack([center_proj - major, center_proj + major])
    width_line_proj = np.vstack([center_proj - minor, center_proj + minor])
    length_line_cm = length_line_proj @ eigenvectors.T + center
    width_line_cm = width_line_proj @ eigenvectors.T + center
    return length_line_cm, width_line_cm


def cm_to_px(points_cm: np.ndarray, scale_x_cm: float, scale_y_cm: float) -> np.ndarray:
    pts = np.asarray(points_cm, dtype=np.float64)
    return np.column_stack([pts[:, 0] / scale_x_cm, pts[:, 1] / scale_y_cm])


def annotate_leaf(
    pdf_path: str,
    output_path: str,
    row: FinalLeafRow,
    vein_info: dict[str, np.ndarray | int],
    dpi: int,
) -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="final_gt_vis_") as tmp_dir:
        png_path = os.path.join(tmp_dir, "page.png")
        render_pdf(pdf_path, png_path, dpi=dpi)
        image = cv2.imread(png_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read rendered PNG for {pdf_path}")

        scale_x_cm, scale_y_cm = pdf_scale_cm(pdf_path, image.shape[1], image.shape[0])
        mask_points_cm, contour_cm = extract_leaf_contour(image, scale_x_cm=scale_x_cm, scale_y_cm=scale_y_cm)
        contour_px = cm_to_px(contour_cm, scale_x_cm, scale_y_cm)
        length_line_cm, width_line_cm = obb_from_mask(mask_points_cm)
        length_line_px = cm_to_px(length_line_cm, scale_x_cm, scale_y_cm)
        width_line_px = cm_to_px(width_line_cm, scale_x_cm, scale_y_cm)
        source_vein_px = np.asarray(vein_info["points_px"], dtype=np.float64)
        src_w = int(vein_info["image_width"])
        src_h = int(vein_info["image_height"])
        vein_polyline_px = source_vein_px.copy()
        vein_polyline_px[:, 0] *= float(image.shape[1]) / float(src_w)
        vein_polyline_px[:, 1] *= float(image.shape[0]) / float(src_h)

        overlay = image.copy()
        contour_fill = np.round(contour_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [contour_fill], color=(220, 245, 220))
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0, image)

        all_points_px = np.vstack([contour_px, length_line_px, width_line_px, vein_polyline_px])
        canvas, offset = crop_with_padding(image, all_points_px, pad=70)

        contour_draw = np.round(shift_points(contour_px, offset)).astype(np.int32).reshape(-1, 1, 2)
        length_draw = np.round(shift_points(length_line_px, offset)).astype(np.int32)
        width_draw = np.round(shift_points(width_line_px, offset)).astype(np.int32)
        vein_draw = np.round(shift_points(vein_polyline_px, offset)).astype(np.int32).reshape(-1, 1, 2)

        cv2.drawContours(canvas, [contour_draw], -1, (40, 120, 255), 3)
        cv2.line(canvas, tuple(length_draw[0]), tuple(length_draw[1]), (0, 0, 255), 3)
        cv2.line(canvas, tuple(width_draw[0]), tuple(width_draw[1]), (170, 0, 170), 3)
        cv2.polylines(canvas, [vein_draw], False, (0, 160, 0), 3)

        for point in [length_draw[0], length_draw[1], width_draw[0], width_draw[1], vein_draw[0, 0], vein_draw[-1, 0]]:
            cv2.circle(canvas, tuple(point), 6, (255, 255, 255), -1)
            cv2.circle(canvas, tuple(point), 6, (30, 30, 30), 2)

        text_lines = [
            row.pdf_name,
            f"length_cm = {row.length_cm:.3f}",
            f"width_cm = {row.width_cm:.3f}",
            f"perimeter_cm = {row.perimeter_cm:.3f}",
            f"vein_length_cm = {row.vein_length_cm:.3f}",
            f"area_cm2 = {row.area_cm2:.3f}",
        ]
        draw_text_block(canvas, text_lines, origin=(18, 18))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)


def main() -> None:
    args = parse_args()
    scan_traits = read_scan_traits(args.scan_traits_txt)
    area_gt_cm2 = read_area_gt_cm2(args.area_txt)
    vein_lengths_cm, vein_meta = read_vein_lengths_cm(args.vein_zip, args.pdf_dir)
    rows = build_rows(scan_traits, area_gt_cm2, vein_lengths_cm)
    write_txt(rows, args.output_txt)

    os.makedirs(args.visual_dir, exist_ok=True)
    for pdf_path in iter_pdfs(args.pdf_dir):
        leaf_id = int(os.path.splitext(os.path.basename(pdf_path))[0])
        row = rows[leaf_id - 1]
        output_path = os.path.join(args.visual_dir, f"{leaf_id}.png")
        annotate_leaf(pdf_path, output_path, row, vein_meta[leaf_id], dpi=args.render_dpi)


if __name__ == "__main__":
    main()
