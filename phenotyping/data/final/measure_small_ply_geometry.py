from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay, cKDTree


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class MeasureResult:
    ply_path: str
    leaf_id: int
    length_cm: float
    width_cm: float
    perimeter_cm: float
    gt_length_cm: float | None
    gt_width_cm: float | None
    gt_perimeter_cm: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure length/width/perimeter from small leaf PLYs.")
    parser.add_argument("--dataset-root", default="data/small", help="Root of the small RGB-D dataset.")
    parser.add_argument(
        "--output-dir",
        default="phenotyping/data/final/small_geom_from_ply",
        help="Output directory for txt and visualizations.",
    )
    parser.add_argument("--long-side-pixels", type=int, default=900, help="Raster size for contour extraction.")
    parser.add_argument("--point-size", type=float, default=1.6, help="Projected point size in visualization.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N PLY files. 0 means all.")
    parser.add_argument(
        "--gt-txt",
        default="phenotyping/data/final/gt_leaf_traits_6params_90.txt",
        help="GT table for leaf-level trait comparison.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def iter_ply_files(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("*/*/*.ply"))


def parse_gt_table(gt_path: Path) -> dict[int, dict[str, float]]:
    gt_map: dict[int, dict[str, float]] = {}
    with open(gt_path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        expected = [
            "leaf_id",
            "pdf_name",
            "length_cm",
            "width_cm",
            "perimeter_cm",
            "vein_length_cm",
            "area_cm2",
            "apex_angle_deg",
        ]
        if header != expected:
            raise ValueError(f"Unexpected GT header in {gt_path}: {header}")
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            leaf_id = int(parts[0])
            gt_map[leaf_id] = {
                "length_cm": float(parts[2]),
                "width_cm": float(parts[3]),
                "perimeter_cm": float(parts[4]),
            }
    return gt_map


def load_points(ply_path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")
    return points


def pca_project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1.0
    projected = centered @ eigenvectors
    return projected, center, eigenvectors


def smooth_contour(contour_xy: np.ndarray, window: int = 9) -> np.ndarray:
    if len(contour_xy) < window or window < 3:
        return contour_xy
    pad = window // 2
    extended = np.vstack([contour_xy[-pad:], contour_xy, contour_xy[:pad]])
    kernel = np.ones(window, dtype=np.float64) / window
    smooth_x = np.convolve(extended[:, 0], kernel, mode="valid")
    smooth_y = np.convolve(extended[:, 1], kernel, mode="valid")
    return np.column_stack([smooth_x, smooth_y])


def triangle_circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    side_a = float(np.linalg.norm(b - c))
    side_b = float(np.linalg.norm(a - c))
    side_c = float(np.linalg.norm(a - b))
    semiperimeter = 0.5 * (side_a + side_b + side_c)
    area_term = semiperimeter * (semiperimeter - side_a) * (semiperimeter - side_b) * (semiperimeter - side_c)
    if area_term <= 1e-16:
        return float("inf")
    area = float(np.sqrt(area_term))
    return side_a * side_b * side_c / (4.0 * area)


def order_boundary_edges(edges: list[tuple[int, int]]) -> list[int]:
    neighbors: dict[int, list[int]] = {}
    for i, j in edges:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    start = min(neighbors.keys())
    ordered = [start]
    prev = None
    current = start
    while True:
        candidates = neighbors[current]
        if prev is None:
            nxt = candidates[0]
        else:
            if len(candidates) == 1:
                break
            nxt = candidates[0] if candidates[1] == prev else candidates[1]
        if nxt == start:
            break
        ordered.append(nxt)
        prev, current = current, nxt
        if len(ordered) > len(edges) + 5:
            break
    return ordered


def extract_alpha_contour(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) < 20:
        raise ValueError("Too few projected points for contour extraction.")

    tree = cKDTree(points_xy)
    distances, _indices = tree.query(points_xy, k=min(8, len(points_xy)))
    neighbor_scale = float(np.median(distances[:, 1:]))
    if not np.isfinite(neighbor_scale) or neighbor_scale <= 0.0:
        raise ValueError("Invalid local point spacing.")

    delaunay = Delaunay(points_xy)
    alpha_radius = neighbor_scale * 8.0
    edge_counts: dict[tuple[int, int], int] = {}

    for simplex in delaunay.simplices:
        pa, pb, pc = points_xy[simplex]
        radius = triangle_circumradius(pa, pb, pc)
        if radius > alpha_radius:
            continue
        tri_edges = [
            tuple(sorted((int(simplex[0]), int(simplex[1])))),
            tuple(sorted((int(simplex[1]), int(simplex[2])))),
            tuple(sorted((int(simplex[0]), int(simplex[2])))),
        ]
        for edge in tri_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if len(boundary_edges) < 8:
        hull = ConvexHull(points_xy)
        contour_xy = points_xy[hull.vertices]
        return smooth_contour(contour_xy, window=9)

    ordered_idx = order_boundary_edges(boundary_edges)
    contour_xy = points_xy[np.asarray(ordered_idx, dtype=np.int32)]
    return smooth_contour(contour_xy, window=9)


def contour_perimeter(contour_xy: np.ndarray) -> float:
    closed = np.vstack([contour_xy, contour_xy[:1]])
    diffs = np.diff(closed, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def measure_obb(points_xy: np.ndarray) -> tuple[float, float, np.ndarray]:
    center = points_xy.mean(axis=0)
    centered = points_xy - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    aligned = centered @ basis

    min_xy = aligned.min(axis=0)
    max_xy = aligned.max(axis=0)
    length = float(max_xy[0] - min_xy[0])
    width = float(max_xy[1] - min_xy[1])

    box_local = np.array(
        [
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ],
        dtype=np.float64,
    )
    box_world = box_local @ basis.T + center
    return length, width, box_world


def draw_dimension_lines(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = []
    for i in range(4):
        p0 = box[i]
        p1 = box[(i + 1) % 4]
        edges.append((float(np.linalg.norm(p1 - p0)), p0, p1))
    edges.sort(key=lambda item: item[0], reverse=True)
    long_mid = np.vstack([edges[0][1], edges[0][2]])
    short_mid = np.vstack([edges[2][1], edges[2][2]])
    return long_mid, short_mid


def save_visualization(
    ply_path: Path,
    projected: np.ndarray,
    contour_xy: np.ndarray,
    box: np.ndarray,
    length_cm: float,
    width_cm: float,
    perimeter_cm: float,
    out_path: Path,
    point_size: float,
    gt_traits: dict[str, float] | None,
) -> None:
    points_xy = projected[:, :2]
    long_seg, short_seg = draw_dimension_lines(box)

    fig, ax2 = plt.subplots(figsize=(7, 7), dpi=180)

    ax2.scatter(points_xy[:, 0], points_xy[:, 1], color="#7f8c8d", s=point_size, alpha=0.45, linewidths=0)
    ax2.plot(contour_xy[:, 0], contour_xy[:, 1], color="#1f77b4", linewidth=1.6, label="Contour")
    closed_box = np.vstack([box, box[:1]])
    ax2.plot(closed_box[:, 0], closed_box[:, 1], color="#d62728", linewidth=1.6, label="OBB")
    ax2.plot(long_seg[:, 0], long_seg[:, 1], color="#ff7f0e", linewidth=2.2, label="Length edge")
    ax2.plot(short_seg[:, 0], short_seg[:, 1], color="#2ca02c", linewidth=2.2, label="Width edge")

    ax2.text(
        0.02,
        0.98,
        (
            f"geom length = {length_cm:.2f} cm\n"
            f"geom width = {width_cm:.2f} cm\n"
            f"geom perimeter = {perimeter_cm:.2f} cm"
        ),
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )
    if gt_traits is not None:
        ax2.text(
            0.02,
            0.20,
            (
                f"GT length = {gt_traits['length_cm']:.2f} cm\n"
                f"GT width = {gt_traits['width_cm']:.2f} cm\n"
                f"GT perimeter = {gt_traits['perimeter_cm']:.2f} cm"
            ),
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )
    ax2.set_title("2D projection with measurements", fontsize=10)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.9)

    fig.suptitle(ply_path.name, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def measure_one(
    ply_path: Path,
    output_visual_dir: Path,
    long_side_pixels: int,
    point_size: float,
    gt_map: dict[int, dict[str, float]],
) -> MeasureResult:
    points = load_points(ply_path)
    projected, _center, _basis = pca_project(points)
    points_xy = projected[:, :2]

    contour_xy = extract_alpha_contour(points_xy)
    length_m, width_m, box = measure_obb(contour_xy)
    perimeter_m = contour_perimeter(contour_xy)
    leaf_id = int(ply_path.stem.split("_")[0])
    gt_traits = gt_map.get(leaf_id)

    result = MeasureResult(
        ply_path=str(ply_path.relative_to(REPO_ROOT)),
        leaf_id=leaf_id,
        length_cm=length_m * 100.0,
        width_cm=width_m * 100.0,
        perimeter_cm=perimeter_m * 100.0,
        gt_length_cm=None if gt_traits is None else gt_traits["length_cm"],
        gt_width_cm=None if gt_traits is None else gt_traits["width_cm"],
        gt_perimeter_cm=None if gt_traits is None else gt_traits["perimeter_cm"],
    )
    out_png = output_visual_dir / f"{ply_path.stem}.png"
    save_visualization(
        ply_path=ply_path,
        projected=projected,
        contour_xy=contour_xy,
        box=box,
        length_cm=result.length_cm,
        width_cm=result.width_cm,
        perimeter_cm=result.perimeter_cm,
        out_path=out_png,
        point_size=point_size,
        gt_traits=gt_traits,
    )
    return result


def write_results(results: list[MeasureResult], output_txt: Path) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write(
            "ply_path\tleaf_id\tgeom_length_cm\tgeom_width_cm\tgeom_perimeter_cm\t"
            "gt_length_cm\tgt_width_cm\tgt_perimeter_cm\n"
        )
        for row in results:
            def fmt(value: float | None) -> str:
                return "" if value is None else f"{value:.4f}"

            handle.write(
                f"{row.ply_path}\t{row.leaf_id}\t{row.length_cm:.4f}\t{row.width_cm:.4f}\t"
                f"{row.perimeter_cm:.4f}\t{fmt(row.gt_length_cm)}\t"
                f"{fmt(row.gt_width_cm)}\t{fmt(row.gt_perimeter_cm)}\n"
            )


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    gt_path = resolve_path(args.gt_txt)
    visual_dir = output_dir / "visual"
    output_txt = output_dir / "small_geom_measurements.txt"
    gt_map = parse_gt_table(gt_path)

    ply_files = iter_ply_files(dataset_root)
    if args.limit > 0:
        ply_files = ply_files[: args.limit]

    results: list[MeasureResult] = []
    for idx, ply_path in enumerate(ply_files, start=1):
        print(f"[{idx}/{len(ply_files)}] {ply_path}")
        try:
            result = measure_one(
                ply_path=ply_path,
                output_visual_dir=visual_dir,
                long_side_pixels=args.long_side_pixels,
                point_size=args.point_size,
                gt_map=gt_map,
            )
            results.append(result)
        except Exception as exc:
            print(f"Failed on {ply_path}: {exc}")

    write_results(results, output_txt)
    print(f"Saved {len(results)} measurements to {output_txt}")
    print(f"Saved visualizations to {visual_dir}")


if __name__ == "__main__":
    main()
