"""Convert the local Franka blue-cube pick-and-place dataset to LeRobot format."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro

SOURCE_FPS = 60
TARGET_FPS = 20
IMAGE_SHAPE = (3, 480, 640)
STATE_DIM = 8


def _find_episodes(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_dir()
        and (path / "samples.jsonl").is_file()
        and (path / "metadata.json").is_file()
        and (path / "base_rgb.mp4").is_file()
        and (path / "wrist_rgb.mp4").is_file()
    )


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_rgb_frame(cap: cv2.VideoCapture, video_path: Path, frame_index: int) -> np.ndarray:
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _state_and_action(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    arm_position = np.asarray(sample["robot_state"]["position"], dtype=np.float32)
    gripper_position = np.asarray(sample["gripper_position"], dtype=np.float32)
    gripper_action = np.asarray([sample["gripper_action"]], dtype=np.float32)

    if arm_position.shape != (7,):
        raise ValueError(f"Expected robot_state.position shape (7,), got {arm_position.shape}")
    if gripper_position.shape != (1,):
        raise ValueError(f"Expected gripper_position shape (1,), got {gripper_position.shape}")

    state = np.concatenate([arm_position, gripper_position]).astype(np.float32)
    action = np.concatenate([arm_position, gripper_action]).astype(np.float32)
    return state, action


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    return cap


def _create_dataset(output_dir: Path, target_fps: int) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=output_dir.name,
        root=output_dir,
        robot_type="franka",
        fps=target_fps,
        use_videos=False,
        features={
            "observation.images.cam_high": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_wrist": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["action"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )


def main(
    raw_dir: Path = Path("data/datasets/pick_and_place_blue_cube"),
    output_dir: Path = Path("data/datasets/pick_and_place_blue_cube_lerobot"),
    target_fps: int = TARGET_FPS,
) -> None:
    if SOURCE_FPS % target_fps != 0:
        raise ValueError(f"target_fps={target_fps} must divide SOURCE_FPS={SOURCE_FPS}")
    frame_stride = SOURCE_FPS // target_fps

    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    episode_dirs = _find_episodes(raw_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found in {raw_dir}")

    dataset = _create_dataset(output_dir, target_fps)

    total_written = 0
    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        samples = _load_jsonl(episode_dir / "samples.jsonl")
        metadata = json.loads((episode_dir / "metadata.json").read_text())
        if float(metadata["sample_hz"]) != SOURCE_FPS:
            raise ValueError(f"{episode_dir} has sample_hz={metadata['sample_hz']}, expected {SOURCE_FPS}")
        prompt = metadata["prompt"]

        video_paths = {
            "observation.images.cam_high": episode_dir / "base_rgb.mp4",
            "observation.images.cam_wrist": episode_dir / "wrist_rgb.mp4",
        }
        caps = {key: _open_video(path) for key, path in video_paths.items()}
        try:
            frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps.values()]
            max_frames = min(len(samples), *frame_counts)
            written_in_episode = 0
            for raw_frame_index in range(0, max_frames, frame_stride):
                sample = samples[raw_frame_index]
                state, action = _state_and_action(sample)
                dataset.add_frame(
                    {
                        "observation.images.cam_high": _read_rgb_frame(
                            caps["observation.images.cam_high"],
                            video_paths["observation.images.cam_high"],
                            raw_frame_index,
                        ),
                        "observation.images.cam_wrist": _read_rgb_frame(
                            caps["observation.images.cam_wrist"],
                            video_paths["observation.images.cam_wrist"],
                            raw_frame_index,
                        ),
                        "observation.state": state,
                        "action": action,
                        "task": prompt,
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

    print(f"Converted {len(episode_dirs)} episodes and {total_written} frames to {output_dir} at {target_fps} Hz")


if __name__ == "__main__":
    tyro.cli(main)
