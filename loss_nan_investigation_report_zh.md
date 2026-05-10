# pi05 Franka Pick-and-Place Loss NaN 排查报告

日期：2026-05-11  
仓库：`/home/wentao/openpi`  
相关配置：`pi05_franka_pick_and_place_full`、`pi05_franka_pick_and_place_lora`

## 1. 背景

目标是把 Franka 抓取蓝色方块并放入篮子的遥操作数据集转换成 OpenPI/JAX 可训练数据集，并使用 `pi05` 做 Franka 微调。

训练目标配置：

- `pi05`，action horizon `10`
- 全量微调训练 `30k` step
- 每 `1000` step 保存 checkpoint
- 使用 GPU `4,5,6,7`
- 尽量充分利用 4 张 A100 80G

排查过程中多次遇到：

```text
loss=nan
grad_norm=nan
param_norm=nan
```

本报告记录数据处理、坏基线复现、消融实验和当前结论。

## 2. 数据集状态

原始数据：

```text
/home/wentao/openpi/data/datasets/pick_and_place_origin
```

当前正式 LeRobot 数据集：

```text
/home/wentao/.cache/huggingface/lerobot/pick_and_place_franka
```

用于复现旧 NaN 的 scaled gripper 旧数据集：

```text
/home/wentao/.cache/huggingface/lerobot/pick_and_place_franka_scaled_gripper
```

数据基本信息：

- 共 `30` 个 episode
- 约 `29668` 帧
- 图像包含 `cam_high` 和 `cam_wrist`
- state/action 为 Franka `7` 关节加 `1` 维夹爪，共 `8` 维
- prompt 为英文任务描述，例如 `pick up the blue block and place it in the basket`

夹爪处理：

- 原始夹爪值是约 `0..0.8` 的角度值，不是归一化值。
- 旧复现实验使用过 scaled gripper：`raw / 0.8`。
- 当前正式数据集按二值语义处理：`raw >= 0.1 -> 1.0` 表示闭合，`raw < 0.1 -> 0.0` 表示张开。

norm stats 检查结论：

- 真实 `8` 维 state/action 没有发现 NaN/Inf。
- 真实 `8` 维没有 `0 std` 或接近 `0 std` 的维度。
- 二值夹爪显著降低初始 loss 量级，但不是单独解决 NaN 的因素。

## 3. 已排除或弱化的怀疑方向

| 怀疑方向 | 检查/实验 | 结论 |
|---|---|---|
| 数据中有 NaN/Inf | 扫描转换后数据、batch、state/action/image/token | 未发现非有限值 |
| norm stats 有 0 std | 检查真实 8 维 std 和 quantile span | 真实维度没有 0 std |
| prompt 未传入 | Franka config 显式 repack `prompt` | 已修正，但不是 NaN 主因 |
| 夹爪连续值语义错误 | 改为二值夹爪 | loss 量级改善，但单独改仍 NaN |
| `discrete_state_input` | 单独改 True/False | 单独改不能消除旧 NaN |
| schedule | 单独改 warmup/decay | 单独改不能消除旧 NaN |
| `loss_action_dim` | 曾短暂验证过 | 已按要求排除，不作为结论依据 |

说明：`loss_action_dim` 是我们自己加过的非 OpenPI 原生逻辑，后续复现和消融都不把它作为证据。当前代码里也已经查不到 `loss_action_dim`。

### 3.1 自定义 TrainConfig 的底层执行链路核查

以下记录对应当前代码状态。

训练入口和 step 逻辑：

- 训练入口为 `scripts/train.py` 的 `main(config)`。
- `main(config)` 调用 `_data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)` 创建 dataloader。
- `main(config)` 调用 `init_train_state(config, init_rng, mesh, resume=resuming)` 初始化模型、optimizer、opt state、EMA state。
- 每个训练 step 调用 `train_step(config, rng, state, batch)`。
- `train_step()` 内部执行：
  - `model = nnx.merge(state.model_def, state.params)`
  - `model.train()`
  - `chunked_loss = model.compute_loss(rng, observation, actions, train=True)`
  - `loss = jnp.mean(chunked_loss)`
  - `nnx.value_and_grad(...)` 计算梯度
  - `state.tx.update(...)` 计算 optimizer update
  - `optax.apply_updates(...)` 更新参数
  - 如果 `ema_decay is not None`，更新 EMA 参数

数据加载和 transform 顺序：

- `create_data_loader()` 先执行 `data_config = config.data.create(config.assets_dirs, config.model)`。
- 当前 Franka LeRobot 数据集走 `data_config.rlds_data_dir is None` 分支。
- 该分支进入 `create_torch_data_loader()`。
- `create_torch_data_loader()` 内部执行：
  - `create_torch_dataset(data_config, action_horizon, model_config)`
  - `transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)`
  - `TorchDataLoader(...)`
  - `DataLoaderImpl(data_config, data_loader)`
- `transform_dataset()` 中 transform 执行顺序为：

```text
*data_config.repack_transforms.inputs
*data_config.data_transforms.inputs
Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
*data_config.model_transforms.inputs
```

- `DataLoaderImpl.__iter__()` 输出：

```text
Observation.from_dict(batch), batch["actions"]
```

Franka data transform 路径：

- `pi05_franka_pick_and_place_full` 使用 `LeRobotFrankaDataConfig`。
- `pi05_franka_pick_and_place_lora` 使用 `LeRobotFrankaDataConfig`。
- `LeRobotFrankaDataConfig.create()` 构造：
  - `franka_policy.FrankaInputs(num_arms=self.num_arms, control_mode=self.control_mode)`
  - `franka_policy.FrankaOutputs(num_arms=self.num_arms, control_mode=self.control_mode)`
- 当前两个配置均设置：
  - `num_arms=1`
  - `control_mode="joint"`
  - `use_delta_actions=True`
- 当 `use_delta_actions=True` 且 `control_mode="joint"` 时，`LeRobotFrankaDataConfig.create()` 创建：

```text
delta_action_mask = make_bool_mask(7, -1)
```

- 该 mask 用于输入侧 `DeltaActions(delta_action_mask)` 和输出侧 `AbsoluteActions(delta_action_mask)`。

模型 transform 路径：

- `LeRobotFrankaDataConfig.create()` 调用 `ModelTransformFactory(default_prompt=self.default_prompt)(model_config)`。
- 对 `ModelType.PI05`，`ModelTransformFactory` 创建以下输入 transform：

```text
InjectDefaultPrompt(self.default_prompt)
ResizeImages(224, 224)
TokenizePrompt(PaligemmaTokenizer(model_config.max_token_len), discrete_state_input=model_config.discrete_state_input)
PadStatesAndActions(model_config.action_dim)
```

`pi05_franka_pick_and_place_full` 当前配置触发的字段：

- `model=pi0_config.Pi0Config(pi05=True, action_horizon=10)`。
- `Pi0Config.__post_init__()` 中：
  - 如果 `max_token_len is None`，`pi05=True` 时设置 `max_token_len=200`。
  - 如果 `discrete_state_input is None`，设置 `discrete_state_input=self.pi05`。
- 因此该配置下：

```text
max_token_len=200
discrete_state_input=True
```

- `data=LeRobotFrankaDataConfig(...)` 中没有显式设置 `use_quantile_norm`。
- `DataConfigFactory.create_base_config()` 中，如果 `use_quantile_norm is None`，设置：

```text
use_quantile_norm = model_config.model_type != ModelType.PI0
```

- `pi05` 的 `model_type` 为 `ModelType.PI05`，因此该配置下：

```text
use_quantile_norm=True
```

`pi05_franka_pick_and_place_lora` 当前配置触发的字段：

- `model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`。
- `Pi0Config.__post_init__()` 中：
  - 如果 `max_token_len is None`，`pi05=True` 时设置 `max_token_len=200`。
  - 因为该配置显式设置了 `discrete_state_input=False`，不会进入默认赋值分支。
- 因此该配置下：

```text
max_token_len=200
discrete_state_input=False
```

- `data=LeRobotFrankaDataConfig(..., use_quantile_norm=False, ...)` 显式设置 `use_quantile_norm=False`。
- 因此该配置下：

```text
use_quantile_norm=False
```

FSDP 相关路径：

- `scripts/train.py` 中执行 `mesh = sharding.make_mesh(config.fsdp_devices)`。
- `make_mesh(num_fsdp_devices)` 中：

```text
mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
```

- `data_sharding` 使用 `PartitionSpec(sharding.DATA_AXIS)`。
- `DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)`。
- `init_train_state()` 中执行：

```text
state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
```

- `fsdp_sharding()` 中，如果 `mesh.shape[FSDP_AXIS] == 1`，对应 array 使用 replicated sharding。
- 如果 `mesh.shape[FSDP_AXIS] > 1`，函数会对满足条件的矩阵或更高维 tensor 沿可被 FSDP 维度整除的最大轴进行 sharding。

Normalize 公式路径：

- 当 `use_quantile_norm=False` 时，`Normalize._normalize()` 使用：

```text
(x - mean) / (std + 1e-6)
```

- 当 `use_quantile_norm=True` 时，`Normalize._normalize_quantile()` 使用：

```text
(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

Prompt tokenize 路径：

- `TokenizePrompt.__call__()` 中，如果 `discrete_state_input=True`，读取 `data["state"]` 并传给 tokenizer。
- 如果 `discrete_state_input=False`，传给 tokenizer 的 state 为 `None`。

### 3.2 关于 gradient clipping / clip norm

当前 OpenPI 训练代码里确实有梯度裁剪，但它不是完整的数值异常保护。

JAX 训练路径中，默认 optimizer 是 `AdamW`：

```text
TrainConfig.optimizer = AdamW()
AdamW.clip_gradient_norm = 1.0
```

`src/openpi/training/optimizer.py` 中 `AdamW.create()` 会构造：

```text
optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), optax.adamw(...))
```

也就是说，默认会在 AdamW 更新前做 global norm clipping。多个 `pi05` 相关 config 也显式设置了：

```text
optimizer=_optimizer.AdamW(clip_gradient_norm=1.0)
```

PyTorch 训练入口 `scripts/train_pytorch.py` 也在 `loss.backward()` 后、`optim.step()` 前调用：

```text
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
```

因此，不能简单判断为“OpenPI 没有 clip norm”。

但这次 NaN / loss explosion 仍然可能发生，原因是：

- 当前日志里的 `grad_norm` 是裁剪前的原始梯度范数。JAX 路径记录的是 `optax.global_norm(grads)`，发生在 `state.tx.update(...)` 之前；PyTorch 的 `clip_grad_norm_` 返回值也通常是裁剪前 norm。因此 `grad_norm > 1.0` 不代表 clip 没有生效。
- Gradient clipping 只能处理“梯度仍然是有限值，但范数过大”的情况。如果 loss 或某个 gradient leaf 已经是 `NaN/Inf`，global norm 本身会变成 `NaN/Inf`，clip 不会自动把它恢复成有限值。
- Clipping 发生在 forward/backward 之后，无法保护 forward 中的输入归一化、激活、loss 计算、bf16/FSDP 数值路径等位置。如果这些位置先产生了非有限值，optimizer 里的 clipping 已经太晚。
- AdamW 的 clipping 裁的是 raw gradient，不是严格限制最终参数更新范数。AdamW 后续还会经过动量、二阶矩、自适应缩放和 weight decay；如果 optimizer state 或参数已被污染成 `NaN`，后续 step 会继续传播。
- 当前训练代码没有看到“loss/grad 非有限时跳过 step、保留旧参数和旧 optimizer state”的 guard。一旦某一步把参数或 Adam 状态写成 `NaN`，后续 checkpoint 基本不适合作为继续训练起点。

因此，这里的正确解释是：

```text
OpenPI 有 gradient clipping；
但当前问题更像是 NaN/Inf 污染或 forward/backward 数值路径不稳定；
clip norm 只能缓解有限梯度过大，不能作为 NaN 防线。
```

对日志的解读规则：

- `grad_norm > 1.0`：不等价于 clip 失效，因为记录的是裁剪前 norm。
- `grad_norm=nan` 且 `loss=nan`：说明至少在 loss/gradient 计算阶段已经出现非有限值，clip 通常无法挽救。
- 一旦出现 `loss/grad_norm/param_norm=nan`，应从干净 base checkpoint 或最后一个确认有限的 checkpoint 重启，不建议继续用污染后的 optimizer state。

## 4. 坏基线复现

为了保证排查有说服力，先复现了不含 `loss_action_dim` 的旧 LoRA NaN 配置。

坏基线配置：

- config：`pi05_franka_pick_and_place_lora`
- 数据：`pick_and_place_franka_scaled_gripper`
- 夹爪：scaled 连续夹爪
- `use_quantile_norm=True`
- `discrete_state_input=True`
- `batch_size=128`
- `fsdp_devices=1`
- seed `202`
- old schedule：`warmup_steps=1000`、`decay_steps=30000`
- `num_train_steps=30`
- `log_interval=1`

日志：

```text
logs/old_lora_no_lossdim_scaled_dsTrue_bs128_fsdp1_4gpu_seed202.log
```

结果：

```text
use_quantile_norm=True
Step 0: grad_norm=1.7991, loss=0.9299, param_norm=1803.7705
Step 1: grad_norm=nan, loss=nan, param_norm=nan
```

结论：旧 LoRA NaN 可以在不含 `loss_action_dim` 的情况下稳定复现，且发生得非常早。

## 5. 单因素消融实验

下面的消融都尽量固定坏基线，只改一个因素。

| 实验 | 改动 | 结果 | 日志 |
|---|---|---|---|
| 坏基线 | 无 | Step 1 NaN | `logs/old_lora_no_lossdim_scaled_dsTrue_bs128_fsdp1_4gpu_seed202.log` |
| schedule 消融 | 只改 `warmup_steps=10000`、`decay_steps=1000000` | Step 1 NaN | `logs/ablate_fsdp1_schedule10k1m_scaled_dsTrue_bs128_seed202.log` |
| 夹爪消融 | 只改二值夹爪 | Step 1 NaN | `logs/ablate_fsdp1_binary_gripper_dsTrue_oldsched_bs128_seed202.log` |
| discrete state 消融 | 只改 `discrete_state_input=False` | Step 1 NaN | `logs/ablate_fsdp1_scaled_dsFalse_oldsched_bs128_seed202.log` |
| FSDP 消融 | 只改 `fsdp_devices=4` | 30 step 无 NaN | `logs/ablate_scaled_dsTrue_bs128_fsdp4_seed202_recheck.log` |
| quantile norm 消融短跑 | 只改 `use_quantile_norm=False` | 30 step 无 NaN | `logs/ablate_lora_old_scaled_quantile_false_fsdp1_seed202.log` |
| quantile norm 消融续跑 | 从 step 30 继续同一配置 | Step 40 NaN | `logs/ablate_lora_old_scaled_quantile_false_fsdp1_seed202_resume_to1000_20260511_101104.log` |

关键日志片段：

```text
# 坏基线：quantile=True, fsdp=1
Step 0: grad_norm=1.7991, loss=0.9299, param_norm=1803.7705
Step 1: grad_norm=nan, loss=nan, param_norm=nan

# 只改 schedule
Step 0: grad_norm=1.7991, loss=0.9299, param_norm=1803.7705
Step 1: grad_norm=nan, loss=nan, param_norm=nan

# 只改二值夹爪
Step 0: grad_norm=0.9735, loss=0.0885, param_norm=1803.7705
Step 1: grad_norm=nan, loss=nan, param_norm=nan

# 只改 discrete_state_input=False
Step 0: grad_norm=1.4354, loss=0.7919, param_norm=1803.7705
Step 1: grad_norm=nan, loss=nan, param_norm=nan

# 只改 fsdp_devices=4，quantile 仍为 True
Step 0: grad_norm=1.7828, loss=0.9307, param_norm=1803.7703
Step 1: grad_norm=2.1360, loss=0.8063, param_norm=1803.7703
Step 29: grad_norm=1.4278, loss=0.7253, param_norm=1803.7704

# 只改 use_quantile_norm=False，fsdp 仍为 1
Step 0: grad_norm=8.4321, loss=10.0245, param_norm=1803.7705
Step 1: grad_norm=11.2500, loss=7.9483, param_norm=1803.7705
Step 29: grad_norm=6.0721, loss=8.1664, param_norm=1803.7706

# 从 step 30 checkpoint 继续同一配置
Step 30: grad_norm=5.2895, loss=9.8688, param_norm=1803.7704
Step 40: grad_norm=nan, loss=nan, param_norm=nan
```

## 6. 关于 `use_quantile_norm`

OpenPI 当前默认逻辑是：非 PI0 模型默认启用 quantile normalization。也就是说，`pi05` 默认会走：

```text
use_quantile_norm=True
```

GitHub issue #763 的核心讨论是：`pi05` 微调时，quantile norm 在小规模自定义数据集上可能带来性能变差甚至 loss 爆炸。我们这次复现也看到类似现象。

重要细节：

- 用户最后提到“尝试 `use_quantile_norm=True`”，但旧 NaN 坏基线日志里已经明确是 `use_quantile_norm=True`。
- 因此真正没有试过、且符合 #763 建议方向的是 `use_quantile_norm=False`。
- 我们只改了这一项，其它旧 LoRA NaN 配置保持一致。

为了能显式控制这个参数，代码里做了一个最小改动：

```text
src/openpi/training/config.py
```

新增 `DataConfigFactory.use_quantile_norm: bool | None = None`：

- `None`：保持 OpenPI 原默认行为，非 PI0 使用 quantile norm。
- `True/False`：允许单个 data config 显式覆盖。

然后只在 `pi05_franka_pick_and_place_lora` 中设置：

```python
use_quantile_norm=False
```

这次实验没有改 full 配置，没有改 seed、bs、fsdp、schedule、数据集和 `discrete_state_input`。

30 step 短跑结果：

```text
旧配置 use_quantile_norm=True:
Step 0: loss=0.9299
Step 1: loss=nan

只改 use_quantile_norm=False:
Step 0: loss=10.0245
Step 1: loss=7.9483
Step 29: loss=8.1664
```

继续训练结果：

```text
从 step 30 checkpoint 继续同一配置:
Step 30: loss=9.8688
Step 40: loss=nan
```

续跑日志：

```text
logs/ablate_lora_old_scaled_quantile_false_fsdp1_seed202_resume_to1000_20260511_101104.log
```

结论：在旧 LoRA NaN 配置上，`use_quantile_norm=False` 可以消除最早的 Step 1 NaN，但不能作为最终修复。它继续到 Step 40 仍然 NaN，只能说明 quantile norm 是早期 NaN 的重要放大因素之一，不是单独根因。

注意：`use_quantile_norm=False` 后 loss 数值明显更大，说明 z-score 归一化下目标尺度不同。这不是“效果更好”的证明，只是“不会立刻 NaN”的证据。真正训练质量仍需要长跑和评估。

## 7. 关于 FSDP

另一个单因素有效项是 `fsdp_devices=4`。

旧配置保持 `use_quantile_norm=True`，只把：

```text
fsdp_devices=1 -> fsdp_devices=4
```

结果从 Step 1 NaN 变为 30 step 无 NaN。

这说明旧 NaN 不是单纯由数据非有限值造成的；它和训练数值路径有关。`fsdp_devices=4` 改变了参数/优化器状态/激活的分片与计算路径，能避开旧配置的早期 NaN。

当前更准确的判断是：

```text
旧 LoRA Step 1 NaN 主要出现在 use_quantile_norm=True + fsdp_devices=1 这条路径上。
禁用 quantile norm 可以把 NaN 从 Step 1 推迟到 Step 40，但不能彻底修复。
改成 fsdp_devices=4 至少能让 30-step probe 不再 NaN，仍需要更长步数确认。
```

不能再简单说“只钉死在 fsdp”，但也不能说 `use_quantile_norm=False` 已经解决问题。最新续跑说明：`fsdp_devices=1` 这条数值路径仍然不稳定，quantile norm 只是让 NaN 更早暴露。

## 8. 关于 full training 的晚期 NaN

full 微调曾尝试：

```text
batch_size=64
fsdp_devices=4
use_quantile_norm=True
binary gripper
discrete_state_input=False
```

日志：

```text
logs/full_franka_pick_place_bs64_fsdp4_30k_20260511_002808.log
```

结果：

```text
Step 0: grad_norm=1.1834, loss=0.1081, param_norm=1802.3864
Step 700: grad_norm=0.2144, loss=0.0113, param_norm=1802.3861
Step 800: grad_norm=nan, loss=nan, param_norm=nan
```

解释：

- 当时 `log_interval=100`，所以 Step 800 是 701-800 区间的聚合指标。
- 第一次 NaN 实际发生在 701-800 之间某一步。
- 一旦某一步污染 Adam 状态和参数，后续继续训练没有意义。

这个结果说明：`fsdp_devices=4` 能解决旧 LoRA 的早期 Step 1 NaN，但不保证 full training 长跑完全不 NaN。`use_quantile_norm=True` 仍然可能是 full 长跑中的风险因素。

## 9. 当前结论

最重要结论：

```text
旧 LoRA NaN 可以稳定复现：use_quantile_norm=True + fsdp_devices=1 时 Step 1 NaN。
```

已经验证的单因素结果：

- 只改 schedule：无效，Step 1 NaN。
- 只改二值夹爪：无效，Step 1 NaN。
- 只改 `discrete_state_input=False`：无效，Step 1 NaN。
- 只改 `fsdp_devices=4`：有效，30 step 无 NaN。
- 只改 `use_quantile_norm=False`：30 step 无 NaN，但从 step 30 继续到 Step 40 NaN。

因此当前最可信的归因是：

```text
NaN 与 pi05 默认 quantile normalization 有关，但更强地指向 fsdp_devices=1 这条数值路径不稳定。
```

更具体地说：

- `use_quantile_norm=True` 是明确风险项，会让旧配置 Step 1 就 NaN。
- `use_quantile_norm=False` 不是最终解，只是把 NaN 推迟到 Step 40。
- `fsdp_devices=1` 是旧 LoRA 配置下持续不稳定的关键路径。
- `fsdp_devices=4` 可以规避早期 NaN，但 full 长跑仍可能在后面 NaN。
- 二值夹爪不是单独修复项，但仍应保留，因为它符合夹爪语义并显著改善 loss 尺度。

## 10. 建议

后续正式训练建议优先尝试：

```text
pi05 full finetune
binary gripper
action_horizon=10
batch_size=32 起步
fsdp_devices=4 起步
use_quantile_norm=False
XLA_PYTHON_CLIENT_PREALLOCATE=false
log_interval=10 或更小
```

原因：

- `bs64/fsdp4/use_quantile_norm=True` 已经在 701-800 区间出现 NaN。
- `use_quantile_norm=False/fsdp1` 已经在 Step 40 NaN，所以不建议继续使用 fsdp1。
- `use_quantile_norm=False` 仍建议保留，因为它能避免 quantile norm 的早期放大效应。
- `fsdp_devices=4` 对旧 LoRA 早期 NaN 有帮助，但也需要 full 长跑确认。

监控建议：

- 前 `1000` step 使用 `log_interval=10` 或更密，避免只看到聚合后的 Step 800 NaN。
- 增加 finite guard 和 post-clip norm 日志：至少区分“裁剪前 norm 过大但有限”和“loss/grad 已经 NaN/Inf”。理想情况下，发现非有限 loss/grad 时跳过该 step，不更新参数和 optimizer state。
- 如果出现 NaN，立即停止该 run。
- 从干净 base checkpoint 或 NaN 前 checkpoint 重新开始，不要继续用污染后的 Adam 状态。
- checkpoint 间隔保持 `1000` step 可以，但要注意 full train state checkpoint 体积大。

下一步最有价值实验：

```text
full finetune
binary gripper
batch_size=32
fsdp_devices=4
use_quantile_norm=False
seed=202
至少跑到 1000-2000 step
```

如果这个配置稳定，再根据显存和吞吐尝试 `fsdp_devices=2` 或更大 batch。

## 11. LIBERO 历史训练记录检查

为了确认仓库里之前是否训练过 LIBERO，以及是否出现过 `loss=nan`，额外检查了以下路径：

```text
logs/pi05_libero_lora_run_persistence.log
checkpoints/pi05_libero
checkpoints/pi05_libero_lora
assets/pi05_libero
assets/pi05_libero_lora
```

检查方法：

- 用 `rg` 搜索 `libero`、`loss`、`nan`、`grad_norm`、`param_norm` 等关键字。
- 用 TensorBoard `EventAccumulator` 解析 `checkpoints/pi05_libero*/**/events.out.tfevents.*` 里的 scalar。
- 对照 checkpoint 目录确认是否存在实际训练保存点。

### 11.1 发现的 LIBERO 训练目录

当前本地存在以下 LIBERO 相关训练输出：

```text
checkpoints/pi05_libero/pi05_libero_run
checkpoints/pi05_libero_lora/pi05_libero_lora_run
checkpoints/pi05_libero_lora/pi05_libero_lora_run_persistence
checkpoints/pi05_libero_lora/tmp_pi05_libero_lora_120step_diag
```

其中 `pi05_libero_lora/pi05_libero_lora_run` 有实际 checkpoint：

```text
5000
10000
15000
20000
25000
30000
```

这说明之前确实跑过较长的 `pi05_libero_lora` 训练，并保存到了 `30000` step。

### 11.2 TensorBoard loss 检查结果

`checkpoints/pi05_libero/pi05_libero_run` 的 TensorBoard event 只包含 `camera_views` 和 `config/text_summary`，没有 `loss`、`grad_norm`、`param_norm` scalar。因此该 run 不能仅凭 event 文件判断 loss 是否 NaN。

`checkpoints/pi05_libero_lora/pi05_libero_lora_run` 的 TensorBoard scalar 显示，`loss` 从记录起就是 NaN：

```text
events.out.tfevents.1775145499.a03.2623830.0
loss count=243
first=nan@0
last=nan@24200
nonfinite=243

events.out.tfevents.1775215647.a03.63352.0
loss count=3
first=nan@24100
last=nan@24300
nonfinite=3

events.out.tfevents.1775248403.a03.19212.0
loss count=59
first=nan@24100
last=nan@29900
nonfinite=59
```

同一 run 的 `grad_norm` 和 `param_norm` 也都是 NaN。这说明该较长 LIBERO LoRA run 的训练指标从 step 0 或恢复记录点开始就已经是非有限值，后续保存到 `30000` 的 checkpoint 很可能已经被 NaN 污染，不适合作为可靠的继续训练起点。

`checkpoints/pi05_libero_lora/pi05_libero_lora_run_persistence` 的 TensorBoard scalar 显示，最后一次记录是 step 0 正常、step 100 变 NaN：

```text
loss count=2
first=0.09698819369077682@0
last=nan@100
nonfinite=1
```

`checkpoints/pi05_libero_lora/tmp_pi05_libero_lora_120step_diag` 是一个短诊断 run，TensorBoard scalar 没有 NaN：

```text
loss@0  = 0.09335267543792725
loss@10 = 0.08366885781288147
loss@20 = 0.09203191846609116
loss@30 = 0.08265446871519089
loss@40 = 0.08234480023384094
```

这说明不是所有 LIBERO 启动都会立刻 NaN；至少短诊断配置在前 40 step 内是有限的。

### 11.3 文本日志证据

`logs/pi05_libero_lora_run_persistence.log` 中存在多次 LIBERO persistence 训练尝试。代表性片段：

```text
Step 0: grad_norm=0.3765, loss=0.0900, param_norm=1803.7705
Step 100: grad_norm=nan, loss=nan, param_norm=nan

Step 0: grad_norm=0.3122, loss=0.0854, param_norm=1803.7705
Step 100: grad_norm=0.3368, loss=0.0830, param_norm=1803.7704
Step 200: grad_norm=0.2688, loss=0.0695, param_norm=1803.7704

Step 0: grad_norm=nan, loss=nan, param_norm=nan

Step 0: grad_norm=0.2063, loss=0.0879, param_norm=1803.7705
Step 100: grad_norm=nan, loss=nan, param_norm=nan

Step 0: grad_norm=nan, loss=nan, param_norm=nan

Step 0: grad_norm=0.3856, loss=0.0897, param_norm=1803.7705
Step 100: grad_norm=0.2178, loss=0.7403, param_norm=1803.7704

Step 0: grad_norm=nan, loss=nan, param_norm=nan

Step 0: grad_norm=0.5144, loss=0.0970, param_norm=1803.7705
Step 100: grad_norm=nan, loss=nan, param_norm=nan
```

该日志还出现过一次数据路径错误：

```text
FileNotFoundError: [Errno 2] No such file or directory: '/home/wentaoasets/physical-intelligence/libero'
```

以及一次 CUDA symbol 警告：

```text
Could not load symbol cuFuncGetName. Error: /lib/x86_64-linux-gnu/libcuda.so.1: undefined symbol: cuFuncGetName
```

这些错误不等价于 loss NaN 的根因，但说明当时 LIBERO persistence 期间存在多轮重启、配置/环境尝试和失败记录。

### 11.4 LIBERO 小结

结论：

```text
之前确实训练过 LIBERO，尤其是 pi05_libero_lora_run，并且存在明确的 loss=nan 记录。
```

更具体地说：

- `pi05_libero_lora/pi05_libero_lora_run` 有到 `30000` step 的 checkpoint，但 TensorBoard 中 `loss`、`grad_norm`、`param_norm` 全部为 NaN。
- `pi05_libero_lora_run_persistence` 多次出现 step 0 正常、step 100 NaN，也有 step 0 直接 NaN 的记录。
- `tmp_pi05_libero_lora_120step_diag` 前 40 step 正常，说明短诊断 run 并非必然 NaN。
- `pi05_libero/pi05_libero_run` 没有 loss scalar，不能判断是否 NaN。

建议：

- 不建议直接从 `checkpoints/pi05_libero_lora/pi05_libero_lora_run` 的 `30000` checkpoint 继续训练，除非先离线检查参数和优化器状态是否存在 NaN/Inf。
- 如果要重新跑 LIBERO，建议从干净 base checkpoint 开始，并在前几百 step 使用更密的 `log_interval`。
- 若复用 `pi05_libero_lora_run_persistence` 的配置，应优先对比那次短诊断 run 和 NaN run 的差异，尤其是 batch size、FSDP、学习率 schedule、数据路径和恢复 checkpoint。

## 12. `openpi_pi05_qa_manual_zh.md` 中的相关社区线索

本节从 `openpi_pi05_qa_manual_zh.md` 中抽取与本次 `loss=nan` / loss explosion / 归一化 / 夹爪语义 / action padding 相关的条目。它们不是本地实验证据，但可作为后续排查方向。

### 12.1 Issue #814: loss explosion and not converge

来源：

```text
openpi_pi05_qa_manual_zh.md
Issue #814: loss explosion and not converge
https://github.com/Physical-Intelligence/openpi/issues/814
```

手册摘要：

- 训练过程中 loss 爆炸且不收敛。
- 用户怀疑 `norm_stats.json` 中接近 0 或为 0 的标准差会导致数值不稳定。
- 社区讨论主线认为“零/近零 std”是强相关诱因。
- 有人把 std 中的 0 值替换后训练恢复稳定。
- 但也有人替换 std、改用 quantile norm 后仍然爆炸。
- 因此该问题目前更像强相关线索，而不是单一根因。

和我们本次实验的对应关系：

- 我们检查过 Franka 当前真实 8 维 state/action，没有发现 0 std 或接近 0 std。
- 但我们确实复现了“归一化路径影响 NaN 时间点”的现象：`use_quantile_norm=True` 时旧 LoRA 在 Step 1 NaN；`use_quantile_norm=False` 时能推迟到 Step 40。
- 这和 #814 的社区观察一致：归一化统计和归一化方式很重要，但不一定是唯一根因。
- 我们当前结果进一步说明：即便关闭 quantile norm，`fsdp_devices=1` 这条路径仍然会在 Step 40 NaN。

### 12.2 Issue #763: Question about using quantiles in pi 0.5 when fine tune

来源：

```text
openpi_pi05_qa_manual_zh.md
Issue #763: Question about using quantiles in pi 0.5 when fine tune.
https://github.com/Physical-Intelligence/openpi/issues/763
```

手册摘要：

- 用户怀疑 pi0.5 微调时默认启用 `use_quantile_norm` 会显著拉低效果。
- 多位用户复现了 quantile norm 比 mean/std z-score 归一化更差的现象。
- 有人指出当某一维 `q01` 与 `q99` 非常接近时，可能引发 loss 爆炸。
- 还有人提醒：即使在 `TrainConfig` 里设成 `False`，如果 `DataConfigFactory` 仍强制开启，问题依然存在。

和我们本次实验的对应关系：

- 这个条目直接解释了为什么必须检查最终 `data_config.use_quantile_norm`，不能只看 `TrainConfig` 表面参数。
- 我们已在 `DataConfigFactory` 增加显式开关，并在日志里确认旧 LoRA 实验实际为 `use_quantile_norm=False`。
- 本地结果支持“quantile norm 是风险项”：旧配置 `use_quantile_norm=True` 在 Step 1 NaN。
- 但本地结果也说明“关掉 quantile norm 不是充分条件”：同配置继续到 Step 40 仍 NaN。

### 12.3 Issue #692: Single arm model performance is worse on pi0.5 when compared with pi0

来源：

```text
openpi_pi05_qa_manual_zh.md
Issue #692: Single arm model performance is worse on pi0.5 when compared with pi0
https://github.com/Physical-Intelligence/openpi/issues/692
```

手册摘要：

- 多个用户在单臂任务上复现 pi0.5 比 pi0 差的现象。
- 讨论中较具体的建议是重新检查归一化，尤其尝试关闭 `q01/q99` 这类统计项。
- 没有形成统一有效修复方案。

和我们本次实验的对应关系：

- 我们任务也是 Franka 单臂。
- 我们观察到 `q01/q99` 量化归一化确实影响早期 NaN。
- 但 `use_quantile_norm=False` 只推迟 NaN，说明单臂 pi0.5 的不稳定可能还与 FSDP/sharding、动作尺度、LoRA/full 配置等因素耦合。

### 12.4 Issue #850: Running time is too long for Pi05 fine tuning of LoRA

来源：

```text
openpi_pi05_qa_manual_zh.md
Issue #850: Running time is too long for Pi05 fine tuning of LoRA
https://github.com/Physical-Intelligence/openpi/issues/850
```

手册摘要：

- 用户在自建数据上做 pi0.5 LoRA 微调时训练耗时很长。
- 社区建议检查 gripper state 编码是否符合预训练假设，常见为 `0/1`。
- 有人降低 batch size 后继续训练，loss 稳定在约 `0.022`，但没有统一结论。

和我们本次实验的对应关系：

- 这支持我们把 Franka 夹爪从原始 `0..0.8` 角度值改成二值语义。
- 本地消融显示：只改二值夹爪不能单独解决 NaN，但能显著改善 loss 初值，从约 `0.93` 降到约 `0.09`。
- 因此二值夹爪应作为数据语义修正保留，但不应被当作 NaN 根因的唯一修复。

### 12.5 Issue #817: 真实设备效果差与归一化/反归一化一致性

来源：

```text
openpi_pi05_qa_manual_zh.md
Issue #817: The actual performance on the real device is very poor after fine-tuning the pi by 0.5.
https://github.com/Physical-Intelligence/openpi/issues/817
```

手册摘要：

- 用户怀疑训练阶段做了归一化，而推理阶段直接喂原始状态、直接执行原始动作，导致输入输出分布不一致。
- 线程没有明确解决方案。
- 核心怀疑点是训练与推理侧缺少一致的归一化/反归一化处理。

和我们本次实验的对应关系：

- 这个条目不直接解释训练 NaN，但提醒后续部署时必须确认推理侧使用相同 norm stats 和动作反归一化逻辑。
- 如果训练阶段改了 `use_quantile_norm` 或换了数据集 assets，推理侧也必须加载同一份 assets，否则即使训练 loss 正常，真实动作也可能异常。

### 12.6 Issue #841: 评估中 NaN 的非训练根因

来源：

```text
openpi_pi05_qa_manual_zh.md
Issue #841: Non-deterministic inference across episodes with identical inputs/noise
https://github.com/Physical-Intelligence/openpi/issues/841
```

手册摘要：

- 用户在评估时发现相同输入、相同 seed、相同噪声下，后续 episode 输出不同动作甚至 NaN。
- 最后定位到 `imageio` 录视频通过 `os.fork()` 拉起 ffmpeg，而 JAX 是多线程的，fork 会导致不安全行为和非确定性/NaN。
- 关闭视频保存后问题消失，建议离线编码视频。

和我们本次实验的对应关系：

- 本次 NaN 发生在训练日志 `loss/grad_norm/param_norm` 中，不是 rollout 录视频触发。
- 但后续做评估时应避免在 JAX 进程中 fork ffmpeg，否则可能把推理 NaN 和训练 NaN 混淆。

### 12.7 Pull Request #871: action mask / padded action 维度

来源：

```text
openpi_pi05_qa_manual_zh.md
PullRequest #871: add action mask in loss computation
https://github.com/Physical-Intelligence/openpi/pull/871
```

手册摘要：

- PR 认为 PyTorch 版 π 模型在计算 MSE loss 时把 padding 出来的动作维度也算进去了，可能拖累训练效果。
- 提议新增 `actual_action_dim` 和 `action_mask`，在 loss 中屏蔽 padded action。
- 目前没有维护者评审结论。

和我们本次实验的对应关系：

- 我们的 Franka 动作真实维度是 8，但 pi05 默认 `action_dim=32`，确实存在 padding 维度。
- 这个方向曾启发过我们短暂尝试 `loss_action_dim`，但它是非 OpenPI 原生修改，已按要求从最终证据链中剔除。
- 当前报告结论不依赖 action mask；不过该 PR 说明“padding 维参与 loss”是社区也关注的风险点，后续如果官方合并类似逻辑，可以重新评估。

### 12.8 Issue #836: FAST 预训练中的单臂/双臂 action padding

来源：

```text
openpi_pi05_qa_manual_zh.md
Issue #836: FAST model pre-training single-arm and dual-arm data
https://github.com/Physical-Intelligence/openpi/issues/836
```

手册摘要：

- 用户追问 pi0.5 FAST 预训练混合单臂/双臂数据时，padding 出来的动作维度是否会在 loss 中被 mask。
- 线程没有答复，因此 action 维度 masking 与解码恢复的细节仍未澄清。

和我们本次实验的对应关系：

- 这和 Franka 8 维动作被 pad 到 32 维的问题相关。
- 但由于没有官方结论，不能把它作为本次 NaN 根因。
- 可以作为后续等待官方 action mask 方案或自己做严格对照实验的背景资料。

### 12.9 Issue #773 / Issue #674: norm stats 的必要性与产物位置

来源：

```text
openpi_pi05_qa_manual_zh.md
Issue #773: question about computation of the norm stats
https://github.com/Physical-Intelligence/openpi/issues/773

Issue #674: cannot find the download dataset
https://github.com/Physical-Intelligence/openpi/issues/674
```

手册摘要：

- #773 关注自定义数据微调时是否必须重新计算 norm stats，尤其是 DROID/custom dataset 场景。
- #674 指出 `compute_norm_stats.py` 的直接产物是写入 config assets 目录，数据集本身位置由 config 决定。
- 两个条目都没有完整官方解决方案。

和我们本次实验的对应关系：

- 本次训练依赖 `assets/pi05_franka_pick_and_place_lora/pick_and_place_franka_scaled_gripper` 和对应 `norm_stats`。
- 实验中必须区分 HF LeRobot 数据目录和 OpenPI assets/norm_stats 目录。
- 后续任何数据重转、夹爪语义改变、`use_quantile_norm` 改变，都应重新确认 assets 中的 norm stats 和训练配置一致。

### 12.10 小结

手册中的社区线索和本地实验基本一致地指向三类风险：

- 归一化风险：`q01/q99` 过近、0/近 0 std、`use_quantile_norm` 默认开启、assets 与数据不匹配。
- 动作语义风险：夹爪应优先使用符合预训练假设的 `0/1` 语义；单臂动作被 pad 到更高维时，padding loss/mask 仍是待澄清问题。
- 数值路径风险：不同 FSDP/sharding、batch、恢复 checkpoint、评估进程 fork 等都可能改变 NaN 表现。

结合我们自己的实验，当前最务实的判断仍然是：

```text
use_quantile_norm=True 会让旧 LoRA 配置 Step 1 NaN；
use_quantile_norm=False 只能推迟到 Step 40；
fsdp_devices=1 路径不稳定；
后续应优先验证 use_quantile_norm=False + fsdp_devices=4 的 full/LoRA 长跑。
```
