"""Convert Franka Data Studio pick-and-place episodes to LeRobot format.

Example:
uv run examples/franka/convert_pick_and_place_to_lerobot.py \
    --raw-dir /home/wentao/openpi/data/datasets/pick_and_place_origin \
    --repo-id pick_and_place_franka_10hz \
    --task "pick and place the block"
"""

from __future__ import annotations

from itertools import pairwise
import json
from pathlib import Path
import random
import shutil
from typing import Literal

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

JOINT_NAMES = (
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
    "gripper",
)

DEFAULT_TASK_PROMPTS = (
    "pick up the blue block and place it in the basket",
    "grasp the blue block and put it into the basket",
    "lift the blue block and place it inside the basket",
    "move the blue block into the basket",
    "pick up the blue block and put it in the basket",
)


def _episode_dirs(raw_dir: Path) -> list[Path]:
    episodes = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_dir():
            continue
        required = ("metadata.json", "samples.jsonl", "base_rgb.mp4", "wrist_rgb.mp4")
        if all((path / name).is_file() for name in required):
            episodes.append(path)
    if not episodes:
        raise FileNotFoundError(f"No valid episodes found in {raw_dir}")
    return episodes


def _load_samples(samples_path: Path) -> list[dict]:
    with samples_path.open() as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if len(samples) < 2:
        raise ValueError(f"Episode {samples_path.parent} has fewer than two samples")
    return samples


def _downsample_indices(num_samples: int, *, source_fps: int, target_fps: int) -> list[int]:
    if target_fps <= 0:
        raise ValueError(f"target fps must be positive, got {target_fps}")
    if source_fps <= 0:
        raise ValueError(f"source fps must be positive, got {source_fps}")
    if source_fps % target_fps != 0:
        raise ValueError(
            f"source fps ({source_fps}) must be an integer multiple of target fps ({target_fps})"
        )

    stride = source_fps // target_fps
    indices = list(range(0, num_samples, stride))
    if len(indices) < 2:
        raise ValueError(
            f"Episode has too few samples ({num_samples}) after downsampling "
            f"from {source_fps}Hz to {target_fps}Hz"
        )
    return indices


GripperMode = Literal["binary", "scaled", "raw"]


def _state_from_sample(
    sample: dict,
    *,
    gripper_mode: GripperMode,
    gripper_close_threshold: float,
    gripper_max_angle: float,
) -> np.ndarray:
    joints = np.asarray(sample["robot_state"]["position"], dtype=np.float32)
    raw_gripper = np.asarray(sample["gripper_position"], dtype=np.float32)
    if gripper_mode == "binary":
        # OpenPI Franka/DROID convention: 0.0 is open and 1.0 is closed.
        gripper = (raw_gripper >= np.float32(gripper_close_threshold)).astype(np.float32)
    elif gripper_mode == "scaled":
        gripper = np.clip(raw_gripper / np.float32(gripper_max_angle), 0.0, 1.0)
    elif gripper_mode == "raw":
        gripper = raw_gripper
    else:
        raise ValueError(f"Unsupported gripper_mode: {gripper_mode}")
    if joints.shape != (7,):
        raise ValueError(f"Expected 7 joint positions, got {joints.shape}")
    if gripper.shape != (1,):
        raise ValueError(f"Expected 1 gripper value, got {gripper.shape}")
    return np.concatenate([joints, gripper], dtype=np.float32)


def _read_rgb(cap: cv2.VideoCapture, video_path: Path, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _task_from_episode(ep_dir: Path) -> str:
    prompts_path = ep_dir / "prompts.jsonl"
    if not prompts_path.exists():
        return "pick and place the block"
    with prompts_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            prompt = json.loads(line).get("prompt", "")
            if isinstance(prompt, list):
                prompt = " ".join(str(part) for part in prompt)
            prompt = str(prompt).replace("_", " ").strip()
            return prompt or "pick and place the block"
    return "pick and place the block"


def _create_dataset(repo_id: str, fps: int, *, overwrite: bool) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [JOINT_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [JOINT_NAMES],
        },
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_wrist": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="franka",
        fps=fps,
        features=features,
        use_videos=False,
        image_writer_processes=4,
        image_writer_threads=8,
    )


def main(
    raw_dir: Path = Path("/home/wentao/openpi/data/datasets/pick_and_place_origin"),
    repo_id: str = "pick_and_place_franka_10hz",
    task: str | None = None,
    fps: int = 10,
    source_fps: int = 60,
    gripper_mode: GripperMode = "binary",
    gripper_close_threshold: float = 0.1,
    gripper_max_angle: float = 0.8,
    prompt_seed: int = 42,
    *,
    randomize_prompts: bool = True,
    overwrite: bool = True,
) -> None:
    dataset = _create_dataset(repo_id, fps, overwrite=overwrite)
    episode_dirs = _episode_dirs(raw_dir)
    rng = random.Random(prompt_seed)

    total_frames = 0
    for ep_dir in tqdm.tqdm(episode_dirs, desc="Converting episodes"):
        samples = _load_samples(ep_dir / "samples.jsonl")
        metadata = json.loads((ep_dir / "metadata.json").read_text())
        expected_count = int(metadata.get("sample_count", len(samples)))
        if expected_count != len(samples):
            raise ValueError(f"{ep_dir}: metadata sample_count={expected_count}, samples={len(samples)}")

        base_path = ep_dir / "base_rgb.mp4"
        wrist_path = ep_dir / "wrist_rgb.mp4"
        base_cap = cv2.VideoCapture(str(base_path))
        wrist_cap = cv2.VideoCapture(str(wrist_path))
        if not base_cap.isOpened() or not wrist_cap.isOpened():
            raise RuntimeError(f"{ep_dir}: failed to open camera videos")

        base_frames = int(base_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wrist_frames = int(wrist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if base_frames < len(samples) or wrist_frames < len(samples):
            raise ValueError(
                f"{ep_dir}: video frames shorter than samples "
                f"(base={base_frames}, wrist={wrist_frames}, samples={len(samples)})"
            )

        if task is not None:
            episode_task = task
        elif randomize_prompts:
            episode_task = rng.choice(DEFAULT_TASK_PROMPTS)
        else:
            episode_task = _task_from_episode(ep_dir)
        states = [
            _state_from_sample(
                sample,
                gripper_mode=gripper_mode,
                gripper_close_threshold=gripper_close_threshold,
                gripper_max_angle=gripper_max_angle,
            )
            for sample in samples
        ]
        sample_indices = _downsample_indices(len(samples), source_fps=source_fps, target_fps=fps)

        for frame_index, next_frame_index in pairwise(sample_indices):
            dataset.add_frame(
                {
                    "observation.state": states[frame_index],
                    "action": states[next_frame_index],
                    "observation.images.cam_high": _read_rgb(base_cap, base_path, frame_index),
                    "observation.images.cam_wrist": _read_rgb(wrist_cap, wrist_path, frame_index),
                    "task": episode_task,
                }
            )

        base_cap.release()
        wrist_cap.release()
        dataset.save_episode()
        total_frames += len(sample_indices) - 1

    dataset.stop_image_writer()
    print(
        f"Wrote {len(episode_dirs)} episodes / {total_frames} frames "
        f"at {fps}Hz to {HF_LEROBOT_HOME / repo_id}"
    )


if __name__ == "__main__":
    tyro.cli(main)
