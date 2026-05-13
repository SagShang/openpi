"""Convert the Franka blue-cube pick-and-place dataset to LeRobot format.

Example:
HF_LEROBOT_HOME=/data/wentao/openpi/data/datasets uv run \
  examples/franka/convert_pick_and_place_blue_cube_to_lerobot.py
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro

PROMPT = "pick up the blue cube and place it in the basket"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_frame(cap: cv2.VideoCapture, frame_index: int, video_path: Path) -> np.ndarray:
    current_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_index != frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _episode_dirs(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_dir() and (path / "samples.jsonl").is_file() and (path / "base_rgb.mp4").is_file()
    )


def _make_state(sample: dict) -> np.ndarray:
    joint_position = np.asarray(sample["robot_state"]["position"], dtype=np.float32)
    gripper_position = np.asarray(sample["gripper_position"], dtype=np.float32)
    if joint_position.shape != (7,):
        raise ValueError(f"Expected 7 joint positions, got {joint_position.shape}")
    if gripper_position.shape != (1,):
        raise ValueError(f"Expected 1 gripper position, got {gripper_position.shape}")
    return np.concatenate([joint_position, gripper_position]).astype(np.float32)


def _make_action(state: np.ndarray) -> np.ndarray:
    action = state.copy()
    action[-1] = np.float32(1.0 if state[-1] >= 0.1 else 0.0)
    return action


def main(
    raw_dir: Path = Path("data/datasets/pick_and_place_blue_cube"),
    output_dir: Path = Path("data/datasets/pick_and_place_blue_cube_lerobot"),
    fps: int = 20,
    source_fps: int | None = None,
    overwrite: bool = True,
    image_writer_threads: int = 8,
    image_writer_processes: int = 4,
) -> None:
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    repo_id = output_dir.name

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    episode_dirs = _episode_dirs(raw_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found in {raw_dir}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type="franka",
        fps=fps,
        use_videos=False,
        features={
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
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    total_written = 0
    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        samples = _load_jsonl(episode_dir / "samples.jsonl")
        metadata = json.loads((episode_dir / "metadata.json").read_text())
        episode_source_fps = float(source_fps or metadata.get("sample_hz", 60.0))
        stride = episode_source_fps / fps
        if not np.isclose(stride, round(stride)):
            raise ValueError(f"Only integer frame-stride downsampling is supported, got {episode_source_fps=} and {fps=}")
        frame_stride = int(round(stride))
        if frame_stride <= 0:
            raise ValueError(f"Invalid frame stride {frame_stride}")

        video_paths = {
            "observation.images.cam_high": episode_dir / "base_rgb.mp4",
            "observation.images.cam_wrist": episode_dir / "wrist_rgb.mp4",
        }
        caps = {key: cv2.VideoCapture(str(path)) for key, path in video_paths.items()}
        try:
            for key, cap in caps.items():
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open {video_paths[key]}")

            frame_counts = {key: int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for key, cap in caps.items()}
            max_frames = min(len(samples), *frame_counts.values())

            written_in_episode = 0
            for raw_frame_index in range(0, max_frames, frame_stride):
                sample = samples[raw_frame_index]
                state = _make_state(sample)
                dataset.add_frame(
                    {
                        "observation.images.cam_high": _read_frame(
                            caps["observation.images.cam_high"],
                            raw_frame_index,
                            video_paths["observation.images.cam_high"],
                        ),
                        "observation.images.cam_wrist": _read_frame(
                            caps["observation.images.cam_wrist"],
                            raw_frame_index,
                            video_paths["observation.images.cam_wrist"],
                        ),
                        "observation.state": state,
                        "action": _make_action(state),
                        "task": PROMPT,
                    }
                )
                written_in_episode += 1

            if written_in_episode == 0:
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            total_written += written_in_episode
        finally:
            for cap in caps.values():
                cap.release()

    if dataset.image_writer is not None:
        dataset.stop_image_writer()

    print(f"Converted {len(episode_dirs)} episodes and {total_written} frames to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
