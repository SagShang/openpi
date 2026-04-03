"""RoboArena baseline policy configs."""

import dataclasses
from typing import TypeAlias

import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
from openpi.training import weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType


@dataclasses.dataclass(frozen=True)
class FlipRobotWinFrankaGripper(_transforms.DataTransformFn):
    """Flip RobotWin Franka gripper semantics to match OpenPI Franka stats.

    RobotWin stores gripper values as 1=open, 0=close, while OpenPI Franka
    expects 0=open, 1=close. This transform flips the single gripper dimension
    for each arm in both state and action tensors.
    """

    gripper_indices: tuple[int, ...] = (7, 15)

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            state = np.array(data["state"], copy=True)
            state[..., self.gripper_indices] = 1.0 - state[..., self.gripper_indices]
            data["state"] = state
        if "actions" in data:
            actions = np.array(data["actions"], copy=True)
            actions[..., self.gripper_indices] = 1.0 - actions[..., self.gripper_indices]
            data["actions"] = actions
        return data


def _duplicate_single_arm_stats(norm_stats):
    duplicated = {}
    for key, stats in norm_stats.items():
        duplicated[key] = dataclasses.replace(
            stats,
            mean=np.concatenate([stats.mean, stats.mean], axis=-1),
            std=np.concatenate([stats.std, stats.std], axis=-1),
            q01=None if stats.q01 is None else np.concatenate([stats.q01, stats.q01], axis=-1),
            q99=None if stats.q99 is None else np.concatenate([stats.q99, stats.q99], axis=-1),
        )
    return duplicated


def get_robotwin_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotAlohaDataConfig
    from openpi.training.config import LeRobotFrankaDataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    def make_robotwin_aloha_data_config(repo_id: str) -> LeRobotAlohaDataConfig:
        return LeRobotAlohaDataConfig(
            repo_id=repo_id,
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        )

    class RoboTwinFrankaDataConfig(LeRobotFrankaDataConfig):
        def create(self, assets_dirs, model_config):
            match model_config.model_type:
                case ModelType.PI05:
                    assets_dir = "gs://openpi-assets/checkpoints/pi05_base/assets"
                case ModelType.PI0:
                    assets_dir = "gs://openpi-assets/checkpoints/pi0_base/assets"
                case ModelType.PI0_FAST:
                    assets_dir = "gs://openpi-assets/checkpoints/pi0_fast_base/assets"
                case _:
                    raise ValueError(f"Unsupported model type: {model_config.model_type}")

            config = LeRobotFrankaDataConfig.create(
                dataclasses.replace(
                    self,
                    assets=AssetsConfig(
                        assets_dir=assets_dir,
                        asset_id="franka",
                    ),
                ),
                assets_dirs,
                model_config,
            )

            flip_gripper = FlipRobotWinFrankaGripper()
            return dataclasses.replace(
                config,
                norm_stats=(
                    _duplicate_single_arm_stats(config.norm_stats)
                    if config.norm_stats is not None and config.norm_stats["state"].mean.shape[-1] == 8
                    else config.norm_stats
                ),
                data_transforms=_transforms.Group(
                    inputs=(flip_gripper, *config.data_transforms.inputs),
                    outputs=(*config.data_transforms.outputs, flip_gripper),
                ),
            )

    def make_robotwin_franka_data_config(repo_id: str) -> RoboTwinFrankaDataConfig:
        return RoboTwinFrankaDataConfig(
            repo_id=repo_id,
            num_arms=2,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        )

    return [
        ###
        ### finetune config for robotwin
        ###
        # pi05_base by full
        TrainConfig(
            name="pi05_aloha_full_base",
            model=pi0_config.Pi0Config(pi05=True),
            data=make_robotwin_aloha_data_config("your_repo_id"),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=20_000,
            batch_size=64,
            fsdp_devices=1,  # refer line 359
        ),
        # pi05_base by lora
        TrainConfig(
            name="pi05_base_aloha_robotwin_lora",
            model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=make_robotwin_aloha_data_config("your_repo_id"),
            freeze_filter=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora",
                                        action_expert_variant="gemma_300m_lora").get_freeze_filter(),
            batch_size=32,  # the total batch_size not pre_gpu batch_size
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30000,
            fsdp_devices=1,
        ),
        # pi0_base by lora
        TrainConfig(
            name="pi0_base_aloha_robotwin_lora",
            model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
            data=make_robotwin_aloha_data_config("your_repo_id"),
            freeze_filter=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora",
                                        action_expert_variant="gemma_300m_lora").get_freeze_filter(),
            batch_size=32,  # the total batch_size not pre_gpu batch_size
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30000,
            fsdp_devices=1,  # refer line 359
        ),
        # pi0_fast_base by lora
        TrainConfig(
            name="pi0_fast_aloha_robotwin_lora",
            model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
            data=make_robotwin_aloha_data_config("your_repo_id"),
            freeze_filter=pi0_fast.Pi0FASTConfig(
                paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
            batch_size=32,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=30000,
            fsdp_devices=2,  # refer line 359
        ),
        # pi0_base by full
        TrainConfig(
            name="pi0_base_aloha_robotwin_full",
            model=pi0_config.Pi0Config(),
            data=make_robotwin_aloha_data_config("your_repo_id"),
            freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
            batch_size=32,  # the total batch_size not pre_gpu batch_size
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30000,
            fsdp_devices=4,  # refer line 359
        ),
        # pi0_fast_base by full
        TrainConfig(
            name="pi0_fast_aloha_robotwin_full",
            model=pi0_fast.Pi0FASTConfig(),
            data=make_robotwin_aloha_data_config("your_repo_id"),
            freeze_filter=pi0_fast.Pi0FASTConfig().get_freeze_filter(),
            batch_size=32,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=30000,
            fsdp_devices=1,  # refer line 359
        ),
        # pi05_franka by full
        TrainConfig(
            name="pi05_franka_full_base",
            model=pi0_config.Pi0Config(pi05=True),
            data=make_robotwin_franka_data_config("your_repo_id"),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=20_000,
            batch_size=64,
            fsdp_devices=1,  # refer line 359
        ),
        # pi05_franka by lora
        TrainConfig(
            name="pi05_base_franka_robotwin_lora",
            model=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=make_robotwin_franka_data_config("your_repo_id"),
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            batch_size=32,  # the total batch_size not pre_gpu batch_size
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30000,
            fsdp_devices=1,
        ),
        # pi0_franka by lora
        TrainConfig(
            name="pi0_base_franka_robotwin_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=make_robotwin_franka_data_config("your_repo_id"),
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            batch_size=32,  # the total batch_size not pre_gpu batch_size
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30000,
            fsdp_devices=1,  # refer line 359
        ),
        # pi0_fast_franka by lora
        TrainConfig(
            name="pi0_fast_franka_robotwin_lora",
            model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
            data=make_robotwin_franka_data_config("your_repo_id"),
            freeze_filter=pi0_fast.Pi0FASTConfig(
                paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
            batch_size=32,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=30000,
            fsdp_devices=2,  # refer line 359
        ),
        # pi0_franka by full
        TrainConfig(
            name="pi0_base_franka_robotwin_full",
            model=pi0_config.Pi0Config(),
            data=make_robotwin_franka_data_config("your_repo_id"),
            freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
            batch_size=32,  # the total batch_size not pre_gpu batch_size
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30000,
            fsdp_devices=4,  # refer line 359
        ),
        # pi0_fast_franka by full
        TrainConfig(
            name="pi0_fast_franka_robotwin_full",
            model=pi0_fast.Pi0FASTConfig(),
            data=make_robotwin_franka_data_config("your_repo_id"),
            freeze_filter=pi0_fast.Pi0FASTConfig().get_freeze_filter(),
            batch_size=32,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            num_train_steps=30000,
            fsdp_devices=1,  # refer line 359
        ),
    ]
