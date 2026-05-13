import dataclasses
import inspect
from typing import Literal

import numpy as np
from openpi_client import base_policy as _base_policy
import tree
from typing_extensions import override

PrefixSchedule = Literal["linear", "exp", "ones", "zeros"]


@dataclasses.dataclass(frozen=True)
class RealtimeChunkingConfig:
    """Runtime action chunk stitching parameters.

    The wrapper assumes the caller executes `execute_horizon` actions from each
    returned chunk before requesting another chunk. On each new request, the new
    policy output is blended with the previous chunk shifted by that amount.
    """

    execute_horizon: int
    inference_delay: int = 0
    prefix_attention_horizon: int | None = None
    schedule: PrefixSchedule = "exp"
    max_guidance_weight: float = 5.0
    return_horizon: int | None = None


class RealtimeChunkingPolicy(_base_policy.BasePolicy):
    """RTC-style receding-horizon wrapper for chunked policies.

    This implements the deployment-side part of real-time chunking: consecutive
    action chunks are aligned by reusing a prefix from the previous future plan
    and smoothly fading into the newly predicted chunk. It is intentionally
    model-agnostic, so it works with existing OpenPI checkpoints.
    """

    def __init__(self, policy: _base_policy.BasePolicy, config: RealtimeChunkingConfig):
        self._policy = policy
        self._config = config
        self._last_actions: np.ndarray | None = None
        self._last_model_actions: np.ndarray | None = None
        self._supports_model_rtc = _supports_model_rtc(policy)
        infer_parameters = inspect.signature(policy.infer).parameters
        self._infer_accepts_sample_kwargs = "sample_kwargs" in infer_parameters
        self._infer_accepts_return_model_actions = "return_model_actions" in infer_parameters
        self._schedule_id = _schedule_id(config.schedule)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        if _is_reset_request(obs):
            self.reset()

        sample_kwargs = {}
        use_model_rtc = self._last_model_actions is not None and self._supports_model_rtc
        if use_model_rtc:
            prev_horizon = self._last_model_actions.shape[0]
            execute_horizon = _validate_execute_horizon(self._config.execute_horizon, prev_horizon)
            inference_delay = _validate_inference_delay(self._config.inference_delay, execute_horizon)
            prefix_attention_horizon = self._config.prefix_attention_horizon
            if prefix_attention_horizon is None:
                prefix_attention_horizon = prev_horizon - execute_horizon
            prefix_attention_horizon = _validate_prefix_attention_horizon(prefix_attention_horizon, prev_horizon)
            sample_kwargs = {
                "prev_actions": _shift_actions(self._last_model_actions, execute_horizon),
                "rtc_inference_delay": inference_delay,
                "rtc_prefix_attention_horizon": prefix_attention_horizon,
                "rtc_schedule_id": self._schedule_id,
                "rtc_max_guidance_weight": self._config.max_guidance_weight,
            }

        infer_kwargs = {}
        if self._infer_accepts_sample_kwargs:
            infer_kwargs["sample_kwargs"] = sample_kwargs
        if self._infer_accepts_return_model_actions:
            infer_kwargs["return_model_actions"] = True
        result = self._policy.infer(obs, **infer_kwargs)
        actions = np.asarray(result["actions"])
        if actions.ndim < 2:
            raise ValueError(f"RTC expects chunked `actions` with shape [horizon, ...], got {actions.shape}")
        model_actions = result.pop("model_actions", None)
        if model_actions is not None:
            model_actions = np.asarray(model_actions)

        chunk_horizon = actions.shape[0]
        execute_horizon = _validate_execute_horizon(self._config.execute_horizon, chunk_horizon)
        inference_delay = _validate_inference_delay(self._config.inference_delay, execute_horizon)
        prefix_attention_horizon = self._config.prefix_attention_horizon
        if prefix_attention_horizon is None:
            prefix_attention_horizon = chunk_horizon - execute_horizon
        prefix_attention_horizon = _validate_prefix_attention_horizon(prefix_attention_horizon, chunk_horizon)

        stitched_actions = actions
        if self._last_actions is not None and not use_model_rtc:
            previous_plan = _shift_actions(self._last_actions, execute_horizon)
            weights = _prefix_weights(
                start=inference_delay,
                end=prefix_attention_horizon,
                total=chunk_horizon,
                schedule=self._config.schedule,
                dtype=actions.dtype,
            )
            stitched_actions = _blend_actions(previous_plan, actions, weights)

        self._last_actions = np.array(stitched_actions, copy=True)
        self._last_model_actions = np.array(model_actions, copy=True) if model_actions is not None else None

        return_horizon = self._config.return_horizon
        if return_horizon is not None:
            return_horizon = _validate_return_horizon(return_horizon, chunk_horizon)
            result = tree.map_structure(
                lambda value: _trim_chunk(value, chunk_horizon=chunk_horizon, return_horizon=return_horizon),
                result,
            )

        result["actions"] = stitched_actions[:return_horizon] if return_horizon is not None else stitched_actions
        result["rtc"] = {
            "enabled": True,
            "execute_horizon": execute_horizon,
            "inference_delay": inference_delay,
            "prefix_attention_horizon": prefix_attention_horizon,
            "schedule": self._config.schedule,
            "max_guidance_weight": self._config.max_guidance_weight,
            "return_horizon": return_horizon or chunk_horizon,
            "mode": "model_guidance" if use_model_rtc else "output_blend",
        }
        return result

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_actions = None
        self._last_model_actions = None


def _is_reset_request(obs: dict) -> bool:
    return bool(obs.get("reset", False) or obs.get("_reset", False) or obs.get("episode_reset", False))


def _supports_model_rtc(policy: _base_policy.BasePolicy) -> bool:
    if bool(getattr(policy, "_is_pytorch_model", False)):
        return False
    model = getattr(policy, "_model", None)
    sample_actions = getattr(model, "sample_actions", None)
    if sample_actions is None:
        return False
    return "prev_actions" in inspect.signature(sample_actions).parameters


def _schedule_id(schedule: PrefixSchedule) -> int:
    return {"linear": 0, "exp": 1, "ones": 2, "zeros": 3}[schedule]


def _validate_execute_horizon(execute_horizon: int, chunk_horizon: int) -> int:
    if execute_horizon < 1:
        raise ValueError(f"RTC execute_horizon must be >= 1, got {execute_horizon}")
    if execute_horizon > chunk_horizon:
        raise ValueError(
            f"RTC execute_horizon must be <= action chunk horizon ({chunk_horizon}), got {execute_horizon}"
        )
    return execute_horizon


def _validate_inference_delay(inference_delay: int, execute_horizon: int) -> int:
    if inference_delay < 0:
        raise ValueError(f"RTC inference_delay must be >= 0, got {inference_delay}")
    if inference_delay > execute_horizon:
        raise ValueError(
            f"RTC inference_delay must be <= execute_horizon ({execute_horizon}), got {inference_delay}"
        )
    return inference_delay


def _validate_prefix_attention_horizon(prefix_attention_horizon: int, chunk_horizon: int) -> int:
    if prefix_attention_horizon < 0:
        raise ValueError(f"RTC prefix_attention_horizon must be >= 0, got {prefix_attention_horizon}")
    if prefix_attention_horizon > chunk_horizon:
        raise ValueError(
            "RTC prefix_attention_horizon must be <= action chunk horizon "
            f"({chunk_horizon}), got {prefix_attention_horizon}"
        )
    return prefix_attention_horizon


def _validate_return_horizon(return_horizon: int, chunk_horizon: int) -> int:
    if return_horizon < 1:
        raise ValueError(f"RTC return_horizon must be >= 1, got {return_horizon}")
    if return_horizon > chunk_horizon:
        raise ValueError(f"RTC return_horizon must be <= action chunk horizon ({chunk_horizon}), got {return_horizon}")
    return return_horizon


def _shift_actions(actions: np.ndarray, execute_horizon: int) -> np.ndarray:
    """Shift a chunk so index 0 corresponds to the next control step."""

    if execute_horizon == actions.shape[0]:
        return np.repeat(actions[-1:], actions.shape[0], axis=0)
    tail = actions[execute_horizon:]
    pad = np.repeat(actions[-1:], execute_horizon, axis=0)
    return np.concatenate([tail, pad], axis=0)


def _prefix_weights(
    *,
    start: int,
    end: int,
    total: int,
    schedule: PrefixSchedule,
    dtype: np.dtype,
) -> np.ndarray:
    """Return weights for how much of the previous chunk to keep."""

    start = min(start, end)
    steps = np.arange(total, dtype=np.float32)
    if schedule == "ones":
        weights = np.ones(total, dtype=np.float32)
    elif schedule == "zeros":
        weights = (steps < start).astype(np.float32)
    elif schedule in {"linear", "exp"}:
        weights = np.clip((start - 1 - steps) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            weights = weights * np.expm1(weights) / (np.e - 1)
    else:
        raise ValueError(f"Invalid RTC prefix schedule: {schedule}")
    weights = np.where(steps >= end, 0, weights)
    return weights.astype(dtype, copy=False)


def _blend_actions(previous_plan: np.ndarray, actions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    reshape = (weights.shape[0],) + (1,) * (actions.ndim - 1)
    weights = weights.reshape(reshape)
    return weights * previous_plan + (1 - weights) * actions


def _trim_chunk(value, *, chunk_horizon: int, return_horizon: int):
    if isinstance(value, np.ndarray) and value.shape[:1] == (chunk_horizon,):
        return value[:return_horizon]
    return value
