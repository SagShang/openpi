from __future__ import annotations

import io
import json
from pathlib import Path
import re
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = Path("/home/wentao/RoboTwin/data/lift_pot")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/datasets/lift_pot"
CAMERA_KEYS = {
    "cam_high": "observation/head_camera/rgb",
    "cam_left_wrist": "observation/left_camera/rgb",
    "cam_right_wrist": "observation/right_camera/rgb",
}


def episode_index(path: Path) -> int:
    match = re.search(r"episode(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Invalid episode filename: {path.name}")
    return int(match.group(1))


def decode_image(buffer: np.bytes_) -> np.ndarray:
    image = Image.open(io.BytesIO(bytes(buffer))).convert("RGB")
    # RoboTwin camera buffers are tagged as RGB, but their red and blue channels
    # are swapped in practice. Flip them here so the generated LeRobot dataset
    # matches the scene metadata and rendered colors.
    return np.asarray(image, dtype=np.uint8)[..., ::-1].transpose(2, 0, 1)


def sample_prompt(candidates: list[str], rng: np.random.Generator) -> str:
    if not candidates:
        raise ValueError("Prompt candidate list is empty")
    return str(rng.choice(candidates))


def load_prompt(
    raw_dir: Path,
    episode_id: int,
    rng: np.random.Generator,
    *,
    prompt_source: Literal["seen", "unseen", "instructions", "auto"] = "seen",
) -> str:
    instruction_path = raw_dir / "instructions" / f"episode{episode_id}.json"
    if not instruction_path.exists():
        return raw_dir.name.replace("_", " ")

    with instruction_path.open() as f:
        instruction = json.load(f)

    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, list) and instruction:
        return sample_prompt(instruction, rng)
    if isinstance(instruction, dict):
        if prompt_source == "auto":
            candidate_keys = ("instructions", "seen", "unseen")
        elif prompt_source == "seen":
            # RoboTwin official training keeps only seen instructions, then samples one per episode.
            candidate_keys = ("seen", "instructions")
        else:
            candidate_keys = (prompt_source,)

        for key in candidate_keys:
            values = instruction.get(key)
            if isinstance(values, list) and values:
                return sample_prompt(values, rng)
            if isinstance(values, str):
                return values

    return raw_dir.name.replace("_", " ")


def infer_action_dim(episode_path: Path) -> int:
    with h5py.File(episode_path, "r") as episode:
        return int(episode["joint_action/vector"].shape[-1])


def create_dataset(output_dir: Path, repo_id: str, fps: int, *, action_dim: int, use_videos: bool) -> LeRobotDataset:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    media_dtype = "video" if use_videos else "image"

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type="robotwin",
        fps=fps,
        use_videos=use_videos,
        features={
            "observation.images.cam_high": {
                "dtype": media_dtype,
                "shape": (3, 240, 320),
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": media_dtype,
                "shape": (3, 240, 320),
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": media_dtype,
                "shape": (3, 240, 320),
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["action"],
            },
        },
    )


def convert_episode(
    dataset: LeRobotDataset,
    episode_path: Path,
    raw_dir: Path,
    rng: np.random.Generator,
    *,
    prompt_source: Literal["seen", "unseen", "instructions", "auto"] = "seen",
) -> None:
    prompt = load_prompt(raw_dir, episode_index(episode_path), rng, prompt_source=prompt_source)

    with h5py.File(episode_path, "r") as episode:
        joint_positions = np.asarray(episode["joint_action/vector"][:], dtype=np.float32)
        next_joint_positions = np.concatenate([joint_positions[1:], joint_positions[-1:]], axis=0)

        for frame_idx in range(len(joint_positions)):
            dataset.add_frame(
                {
                    "observation.images.cam_high": decode_image(episode[CAMERA_KEYS["cam_high"]][frame_idx]),
                    "observation.images.cam_left_wrist": decode_image(
                        episode[CAMERA_KEYS["cam_left_wrist"]][frame_idx]
                    ),
                    "observation.images.cam_right_wrist": decode_image(
                        episode[CAMERA_KEYS["cam_right_wrist"]][frame_idx]
                    ),
                    "observation.state": joint_positions[frame_idx],
                    "action": next_joint_positions[frame_idx],
                    "task": prompt,
                }
            )

    dataset.save_episode()


def main(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    repo_id: str | None = None,
    fps: int = 50,
    prompt_source: Literal["seen", "unseen", "instructions", "auto"] = "seen",
    prompt_seed: int | None = 0,
    use_videos: bool = False,
) -> None:
    repo_id = repo_id or output_dir.name
    episode_paths = sorted((raw_dir / "data").glob("episode*.hdf5"), key=episode_index)
    if not episode_paths:
        raise ValueError(f"No episodes found in {raw_dir / 'data'}")

    action_dim = infer_action_dim(episode_paths[0])
    for episode_path in episode_paths[1:]:
        episode_action_dim = infer_action_dim(episode_path)
        if episode_action_dim != action_dim:
            raise ValueError(
                f"Inconsistent action dims in {raw_dir}: expected {action_dim}, got {episode_action_dim} in {episode_path}"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset(output_dir, repo_id, fps, action_dim=action_dim, use_videos=use_videos)
    rng = np.random.default_rng(prompt_seed)

    for episode_path in tqdm(episode_paths, desc="Converting RoboTwin episodes"):
        convert_episode(dataset, episode_path, raw_dir, rng, prompt_source=prompt_source)

    print(f"Saved {len(episode_paths)} episodes to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
