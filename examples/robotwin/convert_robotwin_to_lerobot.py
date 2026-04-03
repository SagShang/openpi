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
import pyarrow.parquet as pq
from tqdm import tqdm
import tyro

CAMERA_KEYS = {
    "cam_high": "observation/head_camera/rgb",
    "cam_left_wrist": "observation/left_camera/rgb",
    "cam_right_wrist": "observation/right_camera/rgb",
}


def _replace_list_with_sequence(node):
    if isinstance(node, dict):
        if node.get("_type") == "List":
            node["_type"] = "Sequence"
        for value in node.values():
            _replace_list_with_sequence(value)
    elif isinstance(node, list):
        for value in node:
            _replace_list_with_sequence(value)


def normalize_parquet_metadata(dataset_root: Path) -> int:
    patched_files = 0
    for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        metadata = dict(table.schema.metadata or {})
        huggingface_meta = metadata.get(b"huggingface")
        if huggingface_meta is None:
            continue

        info = json.loads(huggingface_meta.decode("utf-8"))
        before = json.dumps(info, sort_keys=True)
        _replace_list_with_sequence(info)
        after = json.dumps(info, sort_keys=True)
        if before == after:
            continue

        metadata[b"huggingface"] = after.encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), parquet_path)
        patched_files += 1

    return patched_files


def episode_index(path: Path) -> int:
    match = re.search(r"episode(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Invalid episode filename: {path.name}")
    return int(match.group(1))


def decode_image(buffer: np.bytes_) -> np.ndarray:
    image = Image.open(io.BytesIO(bytes(buffer))).convert("RGB")
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def infer_shapes(episode_path: Path) -> tuple[tuple[int, ...], dict[str, tuple[int, int, int]]]:
    with h5py.File(episode_path, "r") as episode:
        assert episode["joint_action/vector"].shape[0] > 0, f"{episode_path}: no frames"
        vector_shape = tuple(np.asarray(episode["joint_action/vector"][0], dtype=np.float32).shape)
        image_shapes = {}
        for camera_name, camera_key in CAMERA_KEYS.items():
            camera_frames = episode[camera_key]
            assert camera_frames.shape[0] > 0, f"{episode_path}: {camera_name} has no frames"
            image_shapes[camera_name] = decode_image(camera_frames[0]).shape
        return vector_shape, image_shapes


def load_prompt(
    raw_dir: Path,
    episode_id: int,
    rng: np.random.Generator,
    *,
    prompt_source: Literal["seen", "unseen"] = "seen",
) -> str:
    instruction_path = raw_dir / "instructions" / f"episode{episode_id}.json"
    assert instruction_path.exists(), f"Missing instruction file: {instruction_path}"

    with instruction_path.open() as f:
        instruction = json.load(f)

    return str(rng.choice(instruction[prompt_source]))


def create_dataset(
    output_dir: Path,
    repo: str,
    fps: int,
    *,
    vector_shape: tuple[int, ...],
    image_shapes: dict[str, tuple[int, int, int]],
    use_videos: bool,
) -> LeRobotDataset:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    media_dtype = "video" if use_videos else "image"

    return LeRobotDataset.create(
        repo_id=repo,
        root=output_dir,
        robot_type="robotwin",
        fps=fps,
        use_videos=use_videos,
        features={
            **{
                f"observation.images.{camera_name}": {
                    "dtype": media_dtype,
                    "shape": image_shapes[camera_name],
                    "names": ["channels", "height", "width"],
                }
                for camera_name in CAMERA_KEYS
            },
            "observation.state": {
                "dtype": "float32",
                "shape": vector_shape,
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": vector_shape,
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
    vector_shape: tuple[int, ...],
    image_shapes: dict[str, tuple[int, int, int]],
    prompt_source: Literal["seen", "unseen"] = "seen",
) -> None:
    prompt = load_prompt(raw_dir, episode_index(episode_path), rng, prompt_source=prompt_source)

    with h5py.File(episode_path, "r") as episode:
        states = np.asarray(episode["joint_action/vector"][:], dtype=np.float32)
        assert len(states) > 0, f"{episode_path}: no frames"
        actions = np.concatenate([states[1:], states[-1:]], axis=0)

        for frame_idx in range(len(states)):
            frame = {}
            for camera_name, camera_key in CAMERA_KEYS.items():
                image = decode_image(episode[camera_key][frame_idx])
                assert image.shape == image_shapes[camera_name], (
                    f"{episode_path}: {camera_name} frame {frame_idx} expected shape "
                    f"{image_shapes[camera_name]}, got {image.shape}"
                )
                frame[f"observation.images.{camera_name}"] = image

            state = states[frame_idx]
            action = actions[frame_idx]
            assert state.shape == vector_shape, (
                f"{episode_path}: state frame {frame_idx} expected shape {vector_shape}, got {state.shape}"
            )
            assert action.shape == vector_shape, (
                f"{episode_path}: action frame {frame_idx} expected shape {vector_shape}, got {action.shape}"
            )
            dataset.add_frame(
                {
                    **frame,
                    "observation.state": state,
                    "action": action,
                    "task": prompt,
                }
            )

    dataset.save_episode()


def main(
    input: Path,
    output: Path,
    *,
    repo: str | None = None,
    fps: int = 50,
    prompt_source: Literal["seen", "unseen"] = "seen",
    prompt_seed: int | None = 0,
    use_videos: bool = False,
) -> None:
    repo = repo or output.name
    episode_paths = sorted((input / "data").glob("episode*.hdf5"), key=episode_index)
    assert episode_paths, f"No episodes found in {input / 'data'}"

    vector_shape, image_shapes = infer_shapes(episode_paths[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset(
        output,
        repo,
        fps,
        vector_shape=vector_shape,
        image_shapes=image_shapes,
        use_videos=use_videos,
    )
    rng = np.random.default_rng(prompt_seed)

    for episode_path in tqdm(episode_paths, desc="Converting RoboTwin episodes"):
        convert_episode(
            dataset,
            episode_path,
            input,
            rng,
            vector_shape=vector_shape,
            image_shapes=image_shapes,
            prompt_source=prompt_source,
        )

    patched_files = normalize_parquet_metadata(output)
    if patched_files:
        print(f"Normalized Hugging Face parquet metadata in {patched_files} files")
    print(f"Saved {len(episode_paths)} episodes to {output}")


if __name__ == "__main__":
    tyro.cli(main)
