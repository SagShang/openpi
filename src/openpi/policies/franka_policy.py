import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_franka_example() -> dict:
    """Creates a random input example for the Franka bimanual policy."""
    return {
        "state": np.ones((16,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "lift the pot",
    }


def _convert_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return einops.rearrange(img, "c h w -> h w c")


@dataclasses.dataclass(frozen=True)
class FrankaBimanualInputs(transforms.DataTransformFn):
    """Inputs for Franka-style bimanual datasets with three cameras and 16D state/action."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] != 16:
            raise ValueError(f"Expected 16D state, got shape {state.shape}")

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain only {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = _convert_image(in_images["cam_high"])
        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": _convert_image(in_images["cam_left_wrist"]),
            "right_wrist_0_rgb": _convert_image(in_images["cam_right_wrist"]),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 16:
                raise ValueError(f"Expected 16D actions, got shape {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaBimanualOutputs(transforms.DataTransformFn):
    """Returns the full 16D action chunk during inference."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}
