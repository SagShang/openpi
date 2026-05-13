import numpy as np

from openpi.policies import realtime_chunking_policy


class FakeChunkPolicy:
    def __init__(self, chunks: list[np.ndarray]):
        self._chunks = chunks
        self._idx = 0
        self.reset_count = 0

    def infer(self, obs: dict) -> dict:
        del obs
        chunk = self._chunks[self._idx]
        self._idx += 1
        return {
            "actions": chunk.copy(),
            "extra": np.arange(chunk.shape[0]),
            "scalar": 1,
        }

    def reset(self) -> None:
        self.reset_count += 1


class FakeModelRtcPolicy(FakeChunkPolicy):
    _is_pytorch_model = False

    class Model:
        def sample_actions(self, rng, observation, *, prev_actions=None):
            del rng, observation, prev_actions

    def __init__(self, chunks: list[np.ndarray]):
        super().__init__(chunks)
        self._model = self.Model()
        self.sample_kwargs = []

    def infer(self, obs: dict, *, sample_kwargs=None, return_model_actions=False) -> dict:
        del obs
        self.sample_kwargs.append(sample_kwargs or {})
        result = super().infer({})
        if return_model_actions:
            result["model_actions"] = result["actions"] + 1000
        return result


def test_realtime_chunking_blends_previous_plan_prefix():
    chunks = [
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.full((6, 2), 100, dtype=np.float32),
    ]
    policy = realtime_chunking_policy.RealtimeChunkingPolicy(
        FakeChunkPolicy(chunks),
        realtime_chunking_policy.RealtimeChunkingConfig(
            execute_horizon=2,
            inference_delay=2,
            prefix_attention_horizon=4,
            schedule="linear",
        ),
    )

    first = policy.infer({})
    second = policy.infer({})

    np.testing.assert_array_equal(first["actions"], chunks[0])

    previous_plan = np.concatenate([chunks[0][2:], np.repeat(chunks[0][-1:], 2, axis=0)], axis=0)
    weights = np.array([1.0, 1.0, 2 / 3, 1 / 3, 0.0, 0.0], dtype=np.float32)[:, None]
    expected = weights * previous_plan + (1 - weights) * chunks[1]
    np.testing.assert_allclose(second["actions"], expected, rtol=1e-6)
    assert second["rtc"]["enabled"] is True
    assert second["rtc"]["mode"] == "output_blend"


def test_realtime_chunking_uses_model_guidance_when_supported():
    chunks = [
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.full((6, 2), 100, dtype=np.float32),
    ]
    inner = FakeModelRtcPolicy(chunks)
    policy = realtime_chunking_policy.RealtimeChunkingPolicy(
        inner,
        realtime_chunking_policy.RealtimeChunkingConfig(
            execute_horizon=2,
            inference_delay=1,
            prefix_attention_horizon=3,
            schedule="zeros",
            max_guidance_weight=7.0,
        ),
    )

    policy.infer({})
    second = policy.infer({})

    expected_prev_actions = np.concatenate(
        [chunks[0][2:] + 1000, np.repeat(chunks[0][-1:] + 1000, 2, axis=0)],
        axis=0,
    )
    np.testing.assert_array_equal(inner.sample_kwargs[0], {})
    np.testing.assert_array_equal(inner.sample_kwargs[1]["prev_actions"], expected_prev_actions)
    assert inner.sample_kwargs[1]["rtc_inference_delay"] == 1
    assert inner.sample_kwargs[1]["rtc_prefix_attention_horizon"] == 3
    assert inner.sample_kwargs[1]["rtc_schedule_id"] == 3
    assert inner.sample_kwargs[1]["rtc_max_guidance_weight"] == 7.0
    np.testing.assert_array_equal(second["actions"], chunks[1])
    assert second["rtc"]["mode"] == "model_guidance"


def test_realtime_chunking_can_trim_chunk_like_outputs():
    chunks = [np.arange(12, dtype=np.float32).reshape(6, 2)]
    policy = realtime_chunking_policy.RealtimeChunkingPolicy(
        FakeChunkPolicy(chunks),
        realtime_chunking_policy.RealtimeChunkingConfig(execute_horizon=2, return_horizon=2),
    )

    result = policy.infer({})

    np.testing.assert_array_equal(result["actions"], chunks[0][:2])
    np.testing.assert_array_equal(result["extra"], np.arange(2))
    assert result["scalar"] == 1


def test_realtime_chunking_resets_on_request_flag():
    chunks = [
        np.arange(8, dtype=np.float32).reshape(4, 2),
        np.full((4, 2), 100, dtype=np.float32),
    ]
    inner = FakeChunkPolicy(chunks)
    policy = realtime_chunking_policy.RealtimeChunkingPolicy(
        inner,
        realtime_chunking_policy.RealtimeChunkingConfig(execute_horizon=2),
    )

    policy.infer({})
    result = policy.infer({"reset": True})

    assert inner.reset_count == 1
    np.testing.assert_array_equal(result["actions"], chunks[1])
