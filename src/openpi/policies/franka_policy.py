import dataclasses
from typing import Literal

import einops
import numpy as np

from openpi import transforms

ArmCount = Literal[1, 2]


def make_franka_example(*, num_arms: ArmCount = 1) -> dict:
    """Creates a random input example for the Franka policy."""
    action_dim = _action_dim(num_arms)
    images = {
        "cam_high": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    }
    if num_arms == 1:
        images["cam_wrist"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    else:
        images["cam_left_wrist"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        images["cam_right_wrist"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)

    return {
        "state": np.random.rand(action_dim).astype(np.float32),
        "images": images,
        "prompt": "do something",
    }


def _action_dim(num_arms: ArmCount) -> int:
    return 8 * num_arms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected image to have 3 dims, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected image to have 3 channels, got shape {image.shape}")
    return image


def _validate_vector(name: str, value: np.ndarray, expected_dim: int, *, exact: bool = True) -> np.ndarray:
    value = np.asarray(value)
    if exact and value.shape[-1] != expected_dim:
        raise ValueError(f"Expected {name} to have {expected_dim} dims, got shape {value.shape}")
    if not exact and value.shape[-1] < expected_dim:
        raise ValueError(f"Expected {name} to have {expected_dim} dims, got shape {value.shape}")
    return value


def _first_present(images: dict[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in images:
            return images[name]
    return None


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Inputs for the Franka policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width] or [height, width, channel].
      - Single arm: requires `cam_high`; optionally accepts `cam_wrist`.
      - Dual arm: requires `cam_high`; optionally accepts `cam_left_wrist` and `cam_right_wrist`.
    - state: [8] for single arm, [16] for dual arm.
    - actions: [action_horizon, state_dim] (training only).

    Notes:
    - Each arm uses `[joint_0..joint_6, gripper]`.
    """

    num_arms: ArmCount = 1

    def __call__(self, data: dict) -> dict:
        action_dim = _action_dim(self.num_arms)

        state = _validate_vector("state", np.asarray(data["state"], dtype=np.float32), action_dim)
        in_images = {name: _parse_image(img) for name, img in data["images"].items()}

        base_image = _first_present(in_images, "cam_high")
        if base_image is None:
            raise ValueError("Expected images to contain `cam_high`.")

        image_dict = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        if self.num_arms == 1:
            wrist_image = _first_present(in_images, "cam_wrist", "cam_left_wrist", "cam_right_wrist")
            if wrist_image is None:
                wrist_image = np.zeros_like(base_image)
                wrist_mask = np.False_
            else:
                wrist_mask = np.True_

            image_dict["left_wrist_0_rgb"] = wrist_image
            image_dict["right_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["left_wrist_0_rgb"] = wrist_mask
            image_masks["right_wrist_0_rgb"] = np.False_
        else:
            left_wrist = _first_present(in_images, "cam_left_wrist")
            right_wrist = _first_present(in_images, "cam_right_wrist")

            image_dict["left_wrist_0_rgb"] = left_wrist if left_wrist is not None else np.zeros_like(base_image)
            image_dict["right_wrist_0_rgb"] = right_wrist if right_wrist is not None else np.zeros_like(base_image)
            image_masks["left_wrist_0_rgb"] = np.True_ if left_wrist is not None else np.False_
            image_masks["right_wrist_0_rgb"] = np.True_ if right_wrist is not None else np.False_

        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = _validate_vector("actions", np.asarray(data["actions"], dtype=np.float32), action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Outputs for the Franka policy."""

    num_arms: ArmCount = 1

    def __call__(self, data: dict) -> dict:
        action_dim = _action_dim(self.num_arms)
        actions = _validate_vector("actions", np.asarray(data["actions"], dtype=np.float32), action_dim, exact=False)
        return {"actions": actions[:, :action_dim]}
