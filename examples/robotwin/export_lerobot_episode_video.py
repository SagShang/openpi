from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image, ImageDraw
import torch
import tyro

LABEL_HEIGHT = 22
LABEL_PADDING = 6


def _to_rgb_image(frame: torch.Tensor | np.ndarray) -> Image.Image:
    array = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor/array, got shape {array.shape}")

    if array.shape[0] in (1, 3, 4):
        array = np.transpose(array[:3], (1, 2, 0))
    elif array.shape[-1] in (1, 3, 4):
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported image layout with shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.max(initial=0.0) <= 1.0 else 1.0
        array = array * scale

    array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    return Image.fromarray(array, mode="RGB")


def _with_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + LABEL_HEIGHT), color=(20, 20, 20))
    canvas.paste(image, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((LABEL_PADDING, 4), label, fill=(255, 255, 255))
    return canvas


def _compose(images: list[Image.Image], layout: Literal["horizontal", "vertical"]) -> Image.Image:
    if not images:
        raise ValueError("No images provided for composition")

    if layout == "horizontal":
        width = sum(img.width for img in images)
        height = max(img.height for img in images)
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        x = 0
        for image in images:
            canvas.paste(image, (x, 0))
            x += image.width
        return canvas

    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def _resolve_output_path(dataset_root: Path, episode_index: int, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return dataset_root / "debug_videos" / f"episode_{episode_index:06d}.mp4"


def main(
    dataset_root: Path,
    episode_index: int,
    *,
    output_path: Path | None = None,
    camera_keys: tuple[str, ...] | None = None,
    layout: Literal["horizontal", "vertical"] = "horizontal",
    max_frames: int | None = None,
    codec: str = "libx264",
    video_backend: str | None = None,
) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Expected a LeRobot dataset at {dataset_root}")

    repo_id = dataset_root.name
    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=[episode_index], video_backend=video_backend)

    selected_camera_keys = list(camera_keys) if camera_keys else list(dataset.meta.camera_keys)
    if not selected_camera_keys:
        raise ValueError(f"No camera keys found in dataset {dataset_root}")

    output_path = _resolve_output_path(dataset_root, episode_index, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_item = dataset[0]
    first_images = [_with_label(_to_rgb_image(first_item[key]), key.split(".")[-1]) for key in selected_camera_keys]
    first_frame = _compose(first_images, layout)

    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream(codec, rate=Fraction(str(dataset.meta.fps)).limit_denominator())
        stream.width = first_frame.width
        stream.height = first_frame.height
        stream.pix_fmt = "yuv420p"

        total_frames = len(dataset) if max_frames is None else min(len(dataset), max_frames)

        for frame_idx in range(total_frames):
            item = first_item if frame_idx == 0 else dataset[frame_idx]
            labeled = [_with_label(_to_rgb_image(item[key]), key.split(".")[-1]) for key in selected_camera_keys]
            canvas = _compose(labeled, layout)

            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 180, 18), fill=(0, 0, 0))
            timestamp = float(item["timestamp"]) if "timestamp" in item else frame_idx / dataset.meta.fps
            draw.text((6, 2), f"frame={frame_idx} t={timestamp:.3f}s", fill=(255, 255, 0))

            video_frame = av.VideoFrame.from_image(canvas)
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    print(f"Exported episode {episode_index} to {output_path}")
    print(f"Cameras: {selected_camera_keys}")
    print(f"Frames: {total_frames}")
    print(f"FPS: {dataset.meta.fps}")


if __name__ == "__main__":
    tyro.cli(main)
