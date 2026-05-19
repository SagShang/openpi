"""Plot initial blue cube positions from base camera first frames.

The detector uses only image pixel values: it thresholds blue pixels in HSV and
BGR dominance space, then takes the largest plausible connected component as the
cube in each episode's first base camera frame.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Detection:
    episode: str
    video_path: Path
    detected: bool
    x: float | None = None
    y: float | None = None
    bbox_x: int | None = None
    bbox_y: int | None = None
    bbox_w: int | None = None
    bbox_h: int | None = None
    area: int | None = None
    method: str | None = None
    reason: str | None = None


def read_first_frame(video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def blue_masks(frame_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(frame_bgr)
    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)

    hsv_blue = cv2.inRange(
        hsv,
        np.array([85, 35, 25], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8),
    )
    blue_dominant = (
        (b16 > 55)
        & (b16 > g16 + 12)
        & (b16 > r16 + 18)
    ).astype(np.uint8) * 255

    stricter_dominant = (
        (b16 > 65)
        & (b16 > g16 + 20)
        & (b16 > r16 + 25)
    ).astype(np.uint8) * 255

    return [
        ("hsv_and_bgr_dominance", cv2.bitwise_and(hsv_blue, blue_dominant)),
        ("bgr_dominance", stricter_dominant),
    ]


def detect_blue_cube(frame_bgr: np.ndarray) -> Detection | None:
    candidates: list[tuple[int, str, int, int, int, int, np.ndarray]] = []
    kernel = np.ones((3, 3), dtype=np.uint8)

    for method, mask in blue_masks(frame_bgr):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        for component_idx in range(1, component_count):
            bbox_x, bbox_y, bbox_w, bbox_h, area = stats[component_idx]
            if area < 35 or area > 5000:
                continue
            if bbox_w < 5 or bbox_h < 5:
                continue
            aspect = bbox_w / max(bbox_h, 1)
            if not 0.25 <= aspect <= 4.0:
                continue
            candidates.append((int(area), method, int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h), centroids[component_idx]))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    area, method, bbox_x, bbox_y, bbox_w, bbox_h, centroid = candidates[0]
    return Detection(
        episode="",
        video_path=Path(),
        detected=True,
        x=float(centroid[0]),
        y=float(centroid[1]),
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        area=area,
        method=method,
    )


def write_csv(path: Path, detections: list[Detection]) -> None:
    fields = [
        "episode",
        "video_path",
        "detected",
        "x_px",
        "y_px",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "area_px",
        "method",
        "reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for det in detections:
            writer.writerow(
                {
                    "episode": det.episode,
                    "video_path": str(det.video_path),
                    "detected": int(det.detected),
                    "x_px": "" if det.x is None else f"{det.x:.3f}",
                    "y_px": "" if det.y is None else f"{det.y:.3f}",
                    "bbox_x": "" if det.bbox_x is None else det.bbox_x,
                    "bbox_y": "" if det.bbox_y is None else det.bbox_y,
                    "bbox_w": "" if det.bbox_w is None else det.bbox_w,
                    "bbox_h": "" if det.bbox_h is None else det.bbox_h,
                    "area_px": "" if det.area is None else det.area,
                    "method": "" if det.method is None else det.method,
                    "reason": "" if det.reason is None else det.reason,
                }
            )


def make_density_map(points_xy: np.ndarray, frame_shape: tuple[int, int, int], sigma: float) -> np.ndarray:
    height, width = frame_shape[:2]
    density = np.zeros((height, width), dtype=np.float32)
    for x, y in points_xy:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < width and 0 <= yi < height:
            density[yi, xi] += 1.0

    kernel_size = int(max(3, round(sigma * 8) // 2 * 2 + 1))
    density = cv2.GaussianBlur(density, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    if density.max() > 0:
        density /= density.max()
    return density


def save_overlay_heatmap(path: Path, background_bgr: np.ndarray, points_xy: np.ndarray, density: np.ndarray, title: str) -> None:
    background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    ax.imshow(background_rgb)
    heatmap = ax.imshow(density, cmap="magma", alpha=np.clip(density * 0.85, 0, 0.85), vmin=0, vmax=1)
    ax.scatter(points_xy[:, 0], points_xy[:, 1], s=18, c="cyan", edgecolors="black", linewidths=0.35, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("x pixel in base camera")
    ax.set_ylabel("y pixel in base camera")
    ax.set_xlim(0, background_bgr.shape[1] - 1)
    ax.set_ylim(background_bgr.shape[0] - 1, 0)
    fig.colorbar(heatmap, ax=ax, label="normalized smoothed position density")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plain_heatmap(path: Path, points_xy: np.ndarray, frame_shape: tuple[int, int, int], density: np.ndarray, title: str) -> None:
    height, width = frame_shape[:2]
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    heatmap = ax.imshow(density, cmap="magma", origin="upper", vmin=0, vmax=1)
    ax.scatter(points_xy[:, 0], points_xy[:, 1], s=16, c="cyan", edgecolors="black", linewidths=0.35, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x pixel in base camera")
    ax.set_ylabel("y pixel in base camera")
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    fig.colorbar(heatmap, ax=ax, label="normalized smoothed position density")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_binned_counts(path: Path, points_xy: np.ndarray, frame_shape: tuple[int, int, int], bin_size: int, title: str) -> np.ndarray:
    height, width = frame_shape[:2]
    x_edges = np.arange(0, width + bin_size, bin_size)
    y_edges = np.arange(0, height + bin_size, bin_size)
    counts, _, _ = np.histogram2d(points_xy[:, 1], points_xy[:, 0], bins=[y_edges, x_edges])

    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    image = ax.imshow(
        counts,
        cmap="viridis",
        origin="upper",
        extent=[0, x_edges[-1], y_edges[-1], 0],
        interpolation="nearest",
    )
    ax.scatter(points_xy[:, 0], points_xy[:, 1], s=14, c="white", edgecolors="black", linewidths=0.35, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("x pixel in base camera")
    ax.set_ylabel("y pixel in base camera")
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    fig.colorbar(image, ax=ax, label=f"count per {bin_size}x{bin_size} px bin")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return counts


def save_debug_grid(path: Path, detections: list[Detection], max_images: int = 120) -> None:
    thumbs: list[np.ndarray] = []
    for det in detections[:max_images]:
        frame = read_first_frame(det.video_path)
        if frame is None:
            continue
        vis = frame.copy()
        if det.detected:
            top_left = (int(det.bbox_x), int(det.bbox_y))
            bottom_right = (int(det.bbox_x + det.bbox_w), int(det.bbox_y + det.bbox_h))
            center = (int(round(det.x)), int(round(det.y)))
            cv2.rectangle(vis, top_left, bottom_right, (0, 255, 255), 2)
            cv2.circle(vis, center, 5, (0, 0, 255), -1)
            label = f"{det.episode} ({det.x:.0f},{det.y:.0f}) a={det.area}"
        else:
            label = f"{det.episode} NO DET"

        thumb = cv2.resize(vis, (256, 192), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(thumb, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    if not thumbs:
        return

    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    grid = np.full((rows * 192, cols * 256, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        grid[row * 192 : (row + 1) * 192, col * 256 : (col + 1) * 256] = thumb
    cv2.imwrite(str(path), grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/pick_and_place_cube"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sigma", type=float, default=18.0, help="Gaussian sigma in pixels for the smoothed density map.")
    parser.add_argument("--bin-size", type=int, default=40, help="Pixel size for the count-bin heatmap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir or dataset_dir / "analysis_blue_cube_initial_positions"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(dataset_dir.glob("*/base_rgb.mp4"))
    detections: list[Detection] = []
    first_detected_frame: np.ndarray | None = None

    for video_path in video_paths:
        episode = video_path.parent.name
        frame = read_first_frame(video_path)
        if frame is None:
            detections.append(Detection(episode=episode, video_path=video_path, detected=False, reason="could_not_read_first_frame"))
            continue

        detection = detect_blue_cube(frame)
        if detection is None:
            detections.append(Detection(episode=episode, video_path=video_path, detected=False, reason="no_plausible_blue_component"))
            continue

        detection.episode = episode
        detection.video_path = video_path
        detections.append(detection)
        if first_detected_frame is None:
            first_detected_frame = frame

    detected = [det for det in detections if det.detected]
    if not detected:
        raise RuntimeError(f"No blue cube detections found under {dataset_dir}")

    frame_shape = first_detected_frame.shape
    points_xy = np.array([(det.x, det.y) for det in detected], dtype=np.float32)
    density = make_density_map(points_xy, frame_shape, sigma=args.sigma)

    title = f"Blue cube initial positions: {len(detected)}/{len(detections)} detected"
    save_overlay_heatmap(output_dir / "blue_cube_initial_heatmap_overlay.png", first_detected_frame, points_xy, density, title)
    save_plain_heatmap(output_dir / "blue_cube_initial_heatmap_plain.png", points_xy, frame_shape, density, title)
    counts = save_binned_counts(
        output_dir / "blue_cube_initial_heatmap_binned_counts.png",
        points_xy,
        frame_shape,
        args.bin_size,
        f"{title}; {args.bin_size}x{args.bin_size} px bins",
    )
    save_debug_grid(output_dir / "blue_cube_initial_detection_debug_grid.jpg", detections)
    write_csv(output_dir / "blue_cube_initial_positions.csv", detections)

    print(f"Dataset: {dataset_dir}")
    print(f"Episodes: {len(detections)}")
    print(f"Detected: {len(detected)}")
    print(f"Missed: {len(detections) - len(detected)}")
    if len(detections) - len(detected):
        print("Missed episodes:")
        for det in detections:
            if not det.detected:
                print(f"  {det.episode}: {det.reason}")
    print(f"x range: {points_xy[:, 0].min():.1f} to {points_xy[:, 0].max():.1f} px")
    print(f"y range: {points_xy[:, 1].min():.1f} to {points_xy[:, 1].max():.1f} px")
    print(f"Mean position: x={points_xy[:, 0].mean():.1f}, y={points_xy[:, 1].mean():.1f} px")
    print(f"Median position: x={np.median(points_xy[:, 0]):.1f}, y={np.median(points_xy[:, 1]):.1f} px")
    print(f"Binned count max: {counts.max():.0f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
