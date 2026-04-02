from collections.abc import Mapping, Sequence
import json
import pathlib
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir: pathlib.Path, *, enabled: bool, purge_step: int | None = None):
        self._writer: SummaryWriter | None = None
        if enabled:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir), purge_step=purge_step)

    def log_config(self, config: Mapping[str, Any], *, step: int = 0) -> None:
        if self._writer is None:
            return
        self._writer.add_text("config", json.dumps(config, indent=2, sort_keys=True, default=str), global_step=step)
        self._writer.flush()

    def log_scalars(self, scalars: Mapping[str, Any], *, step: int) -> None:
        if self._writer is None:
            return

        for key, value in scalars.items():
            scalar = _to_scalar(value)
            if scalar is not None:
                self._writer.add_scalar(key, scalar, global_step=step)
        self._writer.flush()

    def log_images(self, tag: str, images: Sequence[np.ndarray], *, step: int) -> None:
        if self._writer is None or len(images) == 0:
            return

        image_batch = np.stack([_prepare_image(image) for image in images], axis=0)
        self._writer.add_images(tag, image_batch, global_step=step, dataformats="NHWC")
        self._writer.flush()

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()


def _to_scalar(value: Any) -> float | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _prepare_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an image with 3 dimensions, got shape {image.shape}.")

    if image.shape[-1] not in (1, 3) and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)

    image = image.astype(np.float32, copy=False)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if min_value >= 0.0 and max_value <= 1.0:
        return image
    if min_value >= -1.0 and max_value <= 1.0:
        return np.clip((image + 1.0) / 2.0, 0.0, 1.0)
    if min_value >= 0.0 and max_value <= 255.0:
        return np.clip(image / 255.0, 0.0, 1.0)
    if np.isclose(max_value, min_value):
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - min_value) / (max_value - min_value), 0.0, 1.0)
