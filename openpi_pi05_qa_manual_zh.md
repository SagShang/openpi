- # openpi pi0.5 / pi05 Q&A 查阅手册

  - 生成时间：2026-03-13 03:26 UTC
  - 仓库：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
  - 检索范围：GitHub Issues、Pull Requests、Discussions
  - 关键词别名：`pi0.5`、`pi 0.5`、`π0.5`、`pi05`、`Pi-0.5`、`π 0.5`
  - 原始抓取条目数：Issues 105 / PRs 9 / Discussions 1
  - 合并完全重复标题后的手册条目数：114

  ## 使用说明

  - 每条 Q&A 都保留原始 GitHub 链接，便于回看上下文。
  - `Answer` 优先摘取维护者/协作者回复；没有官方答复时，保留社区讨论中的主要信息。
  - 遇到完全相同标题的重复条目，已在手册内合并，并保留相关链接。

  ## 主题目录

  - PyTorch/JAX 转换、依赖与后端一致性（6）
  - 两阶段规划 / subtask 生成 / 高层输出（7）
  - 其他 / 杂项（16）
  - 微调/训练链路与工程故障（22）
  - 性能退化、不收敛与效果不达预期（6）
  - 推理部署、硬件适配与运行时兼容（10）
  - 数据接口、状态/动作表示与归一化（15）
  - 权重/checkpoint/数据资源的发布、下载与访问（26）
  - 模型结构、预训练与实现细节追问（6）

## PyTorch/JAX 转换、依赖与后端一致性

### [PR #634] Pi05 + PyTorch support

- 链接：[https://github.com/Physical-Intelligence/openpi/pull/634](https://github.com/Physical-Intelligence/openpi/pull/634)
- 状态：`MERGED`
- 问题（中文概述）：该 PR 旨在为 openpi 增加 Pi05 的 PyTorch 支持。
- 回答/结论（中文概述）：该 PR 已 `MERGED`，说明相关支持已并入仓库；但线程正文与评论几乎没有技术细节，暂无更完整的实现说明。

### [Issue #738] [Question] why jax use float32 while torch use bfloat16?

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/738](https://github.com/Physical-Intelligence/openpi/issues/738)
- 状态：`CLOSED`
- 问题（中文概述）：提问者想知道：为什么在 PI 0.5 里 JAX 默认看起来用 `float32`，而 PyTorch 版本更偏向 `bfloat16`。
- 回答/结论（中文概述）：暂无官方结论/以讨论为主；社区认为这更像 JAX/NumPy 与 PyTorch 的默认精度策略差异，而不是 openpi 特有设计。提问者还反馈 JAX 已混合精度，且在 4090 上 PyTorch 推理更快，但后续有人报告 PyTorch 效果可能更差。

### [Issue #745] PyTorch Conversion - Transformers replace is not installed correctly

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/745](https://github.com/Physical-Intelligence/openpi/issues/745)
- 状态：`CLOSED`
- 问题（中文概述）：用户按文档完成 JAX->PyTorch checkpoint 转换，并复制 `transformers_replace` 后，加载 `pi05_droid_pytorch` 仍报“安装不正确”。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；线程里只有另一位用户表示复现了同样问题，没有给出官方修复或可验证的解决步骤。

### [Issue #749] ModuleNotFoundError: No module named 'nnx'

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/749](https://github.com/Physical-Intelligence/openpi/issues/749)
- 状态：`CLOSED`
- 问题（中文概述）：用户在执行 `compute_norm_stats.py --config-name pi05_piper_lora_finetune` 时遇到 `ModuleNotFoundError: No module named nnx`。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该 issue 被关闭，但线程内没有维护者或社区给出可执行的排查结论。

### [Issue #781] How to convert pi05 base model to torch

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/781](https://github.com/Physical-Intelligence/openpi/issues/781)
- 状态：`OPEN`
- 问题（中文概述）：用户想把 `pi05_base` 转成 Torch，但在 `config.py` 里找不到对应的 `pi05 base` 配置名，不清楚 `--config_name` 应该填什么。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；仅有其他用户跟帖表示同问，线程中没有官方说明该使用哪个 config。

### [Issue #878] Pytorch version pi0.5 libero failing for unknown reasons.

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/878](https://github.com/Physical-Intelligence/openpi/issues/878)
- 状态：`OPEN`
- 问题（中文概述）：用户反馈 PyTorch 版的 pi0.5 LIBERO 微调模型在执行时异常失败，但没有定位到具体原因。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；评论里只有一张现象截图，没有维护者给出原因分析或修复建议。

## 两阶段规划 / subtask 生成 / 高层输出

### [Issue #647] where is the code of high-level/low-level inference procedure in pi0.5?

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/647](https://github.com/Physical-Intelligence/openpi/issues/647)
- 状态：`CLOSED`
- 问题（中文概述）：用户在论文中看到 pi0.5 先生成高层语义子任务、再生成低层动作，但在开源代码里找不到这段两阶段推理实现。
- 回答/结论（中文概述）：协作者 `kpertsch` 明确说明：当前 openpi 开源仓库只支持 action decoding，不支持论文里显式的高层 subtask 解码；不过现有实现已经能得到“可用的 policy”。社区补充说，如果要自行尝试，可单独调用 `PaliGemma.llm` 做高层文本生成。

### [Issue #664] [Feature request] Add or document “subtask prediction” stage (π0.5 two-stage inference)

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/664](https://github.com/Physical-Intelligence/openpi/issues/664)
- 状态：`OPEN`
- 问题（中文概述）：这是一个功能请求：希望官方补上或文档化 π0.5 论文中的“先预测 subtask、再预测动作”的两阶段推理。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；多位用户认为即使显式 subtask 不是推理必需，也应提供这一步以便严格复现实验、做 apples-to-apples 对比和研究高层规划的贡献。

### [Issue #679] How to generate subtasks for pi0.5?

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/679](https://github.com/Physical-Intelligence/openpi/issues/679)
- 状态：`OPEN`
- 问题（中文概述）：用户尝试直接从 `prefix_out` 解码来生成 pi0.5 的子任务文本，但输出大量乱码，想知道正确做法。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；社区倾向认为不能直接解码 `prefix_out`，而是需要类似 #701 所讨论的自回归解码流程，否则很容易得到乱码。

### [Issue #701] Subtask Generation Issue: Missing first few words

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/701](https://github.com/Physical-Intelligence/openpi/issues/701)
- 状态：`CLOSED`
- 问题（中文概述）：用户复现 pi05 的 subtask 生成时，发现生成句子总会丢掉开头一两个词，想确认是 prompt 格式还是推理实现有问题。
- 回答/结论（中文概述）：暂无官方结论/以讨论为主；社区用户 `BrunoFANG1` 分享了自定义 `gemma_05.py` / `pi05.py` / tokenizer 的实验代码，并表示默认 checkpoint 不能完美直接生成低层 prompt，但经过少量额外训练后可以输出正常英文句子。

### [Issue #790] Questions about Pi0.5 Model Training Details and High Level Planning Implementation

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/790](https://github.com/Physical-Intelligence/openpi/issues/790)
- 状态：`OPEN`
- 问题（中文概述）：用户追问 Pi0.5 论文里的两阶段预训练损失和显式高层规划实现，在当前开源代码中为什么看不到对应部分。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；线程内没有维护者回复，这两个训练/实现细节在该条目下仍未被官方澄清。

### [Issue #801] Question about using pi05 for subtask generation

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/801](https://github.com/Physical-Intelligence/openpi/issues/801)
- 状态：`CLOSED`
- 问题（中文概述）：用户想确认 pi05 是否原本就支持文本生成，并反馈自己尝试生成 subtask 时得到的是不可读乱码。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该 issue 被关闭，但线程内没有留下官方解释或可验证的生成方法。

### [Issue #813] Can we get HL output (subtask language)?

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/813](https://github.com/Physical-Intelligence/openpi/issues/813)
- 状态：`OPEN`
- 问题（中文概述）：用户基于之前的 issue 追问：开源代码是否确实无法输出论文中的 HL/subtask 语言，如果可以，有没有可靠办法把它打印出来。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该条目本身没有收到回复，问题仍停留在“现有公开代码似乎无法直接稳定输出 HL 结果”的讨论层面。

## 其他 / 杂项

### [PR #495] Pi05 support

- 链接：[https://github.com/Physical-Intelligence/openpi/pull/495](https://github.com/Physical-Intelligence/openpi/pull/495)
- 状态：`CLOSED`
- 问题（中文概述）：该 PR 试图为仓库加入 Pi05 支持，并有用户顺带追问 Pi05 权重是否也会发布。
- 回答/结论（中文概述）：该 PR 最终处于 `CLOSED` 而非合并状态；线程没有留下实现细节，也没有对“权重是否发布”给出明确答复。

### [Issue #543] open sauce pi0.5 wen

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/543](https://github.com/Physical-Intelligence/openpi/issues/543)
- 状态：`CLOSED`
- 问题（中文概述）：用户直接追问“pi0.5 什么时候开源”。
- 回答/结论（中文概述）：协作者 `kvablack` 只给了一个玩笑式回应“well, since you asked so nicely”，整体更像顺势确认会放出，但没有给具体发布日期或路线图。

### [Issue #601] Task Progress - Evaluation Details

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/601](https://github.com/Physical-Intelligence/openpi/issues/601)
- 状态：`OPEN`
- 问题（中文概述）：用户想知道论文评估里的“task progress”如何定义：是否会把长任务拆成若干子部分再统计整体完成度。
- 回答/结论（中文概述）：协作者 `kvablack` 明确回复：是的，会把任务拆成若干 sub-parts；更具体的评估细节请看 Pi05 论文附录。

### [Issue #610] OUT_OF_RANGE error when loading params

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/610](https://github.com/Physical-Intelligence/openpi/issues/610)
- 状态：`CLOSED`
- 问题（中文概述）：用户在加载 `pi0_droid` / `pi0_libero` checkpoint 时遇到 `OUT_OF_RANGE` 读取错误，怀疑即使强制重下也无法解决。
- 回答/结论（中文概述）：暂无明确官方结论/以讨论为主；社区经验认为常见原因是下载缓存损坏，删除本地 cache 后重新下载 checkpoint 可能解决同类问题。

### [Issue #635] Pi0.5 fine tuning both experts?

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/635](https://github.com/Physical-Intelligence/openpi/issues/635)
- 状态：`CLOSED`
- 问题（中文概述）：用户想确认：Pi0.5 的开源微调流程是否会同时训练 VLM/Gemma 的下一词预测部分，也就是“两个 expert 都训练”。
- 回答/结论（中文概述）：协作者 `kvablack` 明确说明：当前仓库在训练和推理中都只支持 flow matching head；README 也已更新说明这一点。换言之，公开版并不支持论文里那种完整的高层 subtask/NTP 训练链路。

### [Issue #644] question about the pi05_droid

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/644](https://github.com/Physical-Intelligence/openpi/issues/644)
- 状态：`CLOSED`
- 问题（中文概述）：用户发现 `pi05_droid` 推理返回的 action 维度是 8，而自己查到的 DROID 数据集常见 action 维度是 7，想知道为什么不一致。
- 回答/结论（中文概述）：协作者 `kpertsch` 解释：openpi 在 DROID 上使用的是“7 维关节速度 + 1 维夹爪”动作空间，因此返回 8 维是预期行为；用户查到的 7 维版本对应的是另一种 6D 末端位姿 + 1D 夹爪表示。

### [Issue #648] PyTrees have different structure

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/648](https://github.com/Physical-Intelligence/openpi/issues/648)
- 状态：`CLOSED`
- 问题（中文概述）：用户在用 `pi05_base` 做微调准备时遭遇 `PyTrees have different structure`，怀疑 checkpoint 或初始化配置有问题。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；社区有人怀疑是路径问题，也有人反馈除了排查路径外，还需要在 `Pi0Config` 中显式设置 `pi05=True`，但没有官方确认。

### [Issue #663] ValueError: Config 'pi05_libero' not found. Did you mean 'pi0_libero'?

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/663](https://github.com/Physical-Intelligence/openpi/issues/663)
- 状态：`OPEN`
- 问题（中文概述）：用户在 LIBERO 环境配置 `pi05_libero` checkpoint 时，报错提示 `Config 'pi05_libero' not found`，想知道为何路径对了仍找不到配置。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该线程没有回复，问题集中在仓库中的配置名与用户预期的 `pi05_libero` 不一致。

### [Issue #671] [bug]: TypeError: stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/671](https://github.com/Physical-Intelligence/openpi/issues/671)
- 状态：`CLOSED`
- 问题（中文概述）：用户在自己的 LIBERO 子集数据上执行 `compute_norm_stats.py --config-name pi05_libero_yc`，遇到 `stack(): ... not Column`，怀疑和数据格式或依赖版本有关。
- 回答/结论（中文概述）：暂无官方结论/以讨论为主；提问者后续自查认为核心原因是 `datasets` 版本不兼容，并反馈安装 `datasets==3.6.0` 可以解决。

### [Issue #695] Questions about finetuning Pi-0.5 on Trossen Mobile Aloha data

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/695](https://github.com/Physical-Intelligence/openpi/issues/695)
- 状态：`OPEN`
- 问题（中文概述）：用户想在 Trossen Mobile Aloha 数据上微调 Pi-0.5，追问 action prompt 是否需要显式加 control mode、真实机器人更适合 joint 还是 end-effector 控制，以及 `adapt_to_pi` 该怎么设。
- 回答/结论（中文概述）：协作者 `kpertsch` 给了部分官方答复：只有 end-effector 控制时才会额外 prepend action format；如果你只在一种动作空间上微调，通常可以不额外加标识。经验上他们默认更常用 joint position control，建议先从它开始。关于第 3 个 `adapt_to_pi` 问题，线程里没有继续给出正式答复。

### [PR #722] fix(config): correct pi0.5 LIBERO config for issue #687

- 链接：[https://github.com/Physical-Intelligence/openpi/pull/722](https://github.com/Physical-Intelligence/openpi/pull/722)
- 状态：`OPEN`
- 问题（中文概述）：该 PR 试图修正 pi0.5 在 LIBERO 上的 config，把 `discrete_state_input=True` 设为默认值，以匹配作者预期的 pi0.5 管线。
- 回答/结论（中文概述）：该 PR 目前仍是 `OPEN`；从描述看它想修正配置默认值，但线程里尚无维护者给出最终结论，且有评论者表示即使设为 `False`，实测表现似乎也不错。

### [Discussion #730] Search code, repositories, users, issues, pull requests...

- 链接：[https://github.com/Physical-Intelligence/openpi/discussions/730](https://github.com/Physical-Intelligence/openpi/discussions/730)
- 状态：`OPEN`
- 问题（中文概述）：讨论区用户想知道 π0.5 论文中提到的 stop-gradient 在开源仓库里是如何实现的。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该 discussion 里没有后续答复，stop-gradient 的具体代码位置并未在此线程中澄清。

### [Issue #800] why is attention _mask  added with attn_weight

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/800](https://github.com/Physical-Intelligence/openpi/issues/800)
- 状态：`CLOSED`
- 问题（中文概述）：用户对实现细节有疑问：为什么 pi05 的 attention mask 是加到 `attn_weight` 上，而不是像 pi0 那样用 `torch.where` 处理。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该 issue 在线程内没有收到任何解释。

### [Issue #833] Error decoding tokens when inferencing on libero with pi0_fast_libero

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/833](https://github.com/Physical-Intelligence/openpi/issues/833)
- 状态：`OPEN`
- 问题（中文概述）：用户在使用官方 `pi0_fast_libero` checkpoint 推理时遇到 token 解码异常，怀疑即便是官方权重也存在不正常情况。
- 回答/结论（中文概述）：暂无官方结论/以讨论为主；社区回复认为这未必是异常，因为 action token 会先映射到 PaliGemma 词表尾部的 id 区间，官方实现似乎也没有额外 clamp 以避免越界场景。

### [Issue #859] Possible bug in data augmentation pipeline

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/859](https://github.com/Physical-Intelligence/openpi/issues/859)
- 状态：`OPEN`
- 问题（中文概述）：用户怀疑数据增强流水线里存在一个静默 bug：不同相机在同一帧上使用不一致的 `ColorJitter`/RNG，导致颜色语义被破坏。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；这是报告潜在 bug 的分析帖，但线程里还没有维护者确认问题、解释设计意图或给出修复计划。

### [Issue #885] Request for Rollout Data from pi0.5 for Welch Labs video

- 链接：[https://github.com/Physical-Intelligence/openpi/issues/885](https://github.com/Physical-Intelligence/openpi/issues/885)
- 状态：`OPEN`
- 问题（中文概述）：Welch Labs 想做 VLA 视频，希望有人分享真实机器人运行 pi0.5 的完整 rollout 数据（视频、prompt、action 等）。
- 回答/结论（中文概述）：暂无官方结论/以讨论为主；有一组 KIT 硕士生表示正在评测 Pi05、MolmoAct 和 X-VLA，理论上可能匹配需求，但是否能分享数据还要看导师许可。提问者随后邀请其通过站外联系方式继续沟通。

## 微调/训练链路与工程故障

### [Issue #652] Clarification on converting and finetuning pi05_base with PyTorch training pipeline
- 链接：https://github.com/Physical-Intelligence/openpi/issues/652
- 状态：OPEN
- 问题（中文概述）：询问在官方只提供 `pi05_libero` 等配置、未显式提供 `pi05_base` 配置时，如何把 `pi05_base` 从 JAX 转成 PyTorch 并继续做自定义数据微调。
- 回答/结论（中文概述）：社区有人建议把转换配置先改用 `pi05_libero`，也有人提到可试 `debug_pi05`；但没有官方确认哪一套才是 `pi05_base` 的标准转换/微调流程，暂无明确结论/以讨论为主。

### [Issue #654] Triton out of resources when train and eval pi05 pytorch
- 链接：https://github.com/Physical-Intelligence/openpi/issues/654
- 状态：OPEN
- 问题（中文概述）：Pi0.5 的 PyTorch 训练/评估阶段触发 Triton `OutOfResources`，报错指向共享内存需求超过 GPU 上限，想知道如何规避。
- 回答/结论（中文概述）：讨论认为问题来自 Triton autotune 选到了超出共享内存上限的 kernel 配置；社区可行绕法是关闭 `max_autotune`、限制 GEMM backend 到 `ATEN,CUTLASS,CPP`，并关闭 Triton CUDA graphs，但没有官方修复，暂无明确结论/以讨论为主。

### [Issue #667] Out of Memory when fine-tune pi0.5 on the libero dataset
- 链接：https://github.com/Physical-Intelligence/openpi/issues/667
- 状态：OPEN
- 问题（中文概述）：用户在 2×A100 80GB 上微调 `pi05_libero` 仍然 OOM，想确认推荐配置以及实际需要的显存/内存规模。
- 回答/结论（中文概述）：讨论中唯一较明确的建议是开启 `gradient_checkpointing=True` 以显著降低训练显存占用；但没有给出官方推荐配置或明确的硬件门槛，后续也有人在更多 A100 上复现同类问题，暂无明确结论/以讨论为主。

### [Issue #672] question about lora fine-tuning for pi0.5
- 链接：https://github.com/Physical-Intelligence/openpi/issues/672
- 状态：CLOSED
- 问题（中文概述）：询问是否可以直接照着 `pi0` 的 LoRA 配置改成 `pi05=True` 来微调 pi0.5，以及 `extra_delta_transform` 在不同数据集上的正确设置方式。
- 回答/结论（中文概述）：多位社区用户反馈这种改法可行：可沿用 `Pi0Config`/`Pi0FASTConfig` 的 LoRA 结构并加载 `pi05_base` 权重；对 Libero 这类 delta end-effector 动作空间，通常把 `extra_delta_transform=False`，若数据是 absolute EE 动作则可优先尝试设为 `True`。结论主要来自社区实测而非官方答复。

### [Issue #677] CUDA out of memory when using pytorch to finetune in RTX4090
- 链接：https://github.com/Physical-Intelligence/openpi/issues/677
- 状态：OPEN
- 问题（中文概述）：在 RTX 4090 上用 PyTorch 微调时，即使 batch size=1 也会在 `optim.step()` 触发 CUDA OOM，想知道为什么 JAX 能跑而 PyTorch 不行。
- 回答/结论（中文概述）：讨论普遍认为瓶颈出在 AdamW 优化器状态初始化而不是前向/反向本身；社区给出的有效方案是改用 bitsandbytes 的 `AdamW8bit`、必要时把优化器状态搬到 CPU，并继续减小 batch size。多名用户确认这样至少能开始训练，且有人反馈最终效果未明显下降。

### [Issue #685] Questions for co-training: regarding the modifications made compared to the original Gemma model
- 链接：https://github.com/Physical-Intelligence/openpi/issues/685
- 状态：OPEN
- 问题（中文概述）：用户准备做基于网页/VQA 数据的 co-training，想弄清 openpi 相比原始 Gemma/PaliGemma 做的 RMSNorm 与归一化改动，会不会影响复用原始 VLM 权重。
- 回答/结论（中文概述）：讨论主要停留在推测：有人认为这些改动会放大图像特征幅度，可能带来视觉模块过拟合或破坏原有 VLM 表征；但没有维护者解释 `use_adarms`、去掉 normalizer 的设计意图，也没有明确 co-training 指南，暂无明确结论/以讨论为主。

### [Issue #728] Save ckpt error when fine-tuning
- 链接：https://github.com/Physical-Intelligence/openpi/issues/728
- 状态：OPEN
- 问题（中文概述）：微调 pi0.5 时在保存 checkpoint 阶段无报错中断，用户怀疑不是 GPU/磁盘不足，想知道根因与修复办法。
- 回答/结论（中文概述）：有社区回复表示把 `orbax-checkpoint` 升级到 `0.11.18` 后问题消失；但也有人明确说升级后仍未解决，因此这更像是一个可能的兼容性修复而非通用结论，暂无明确结论/以讨论为主。

### [Issue #743] How to finetune torch-based pi0.5 with libero dataset?
- 链接：https://github.com/Physical-Intelligence/openpi/issues/743
- 状态：OPEN
- 问题（中文概述）：用户先把 `pi05_base` 转成 torch 权重，再把配置里的权重路径改到转换结果，随后运行 `train_pytorch.py pi05_libero` 仍报错，想知道 torch 版如何正确用 Libero 微调 pi0.5。
- 回答/结论（中文概述）：帖子里只有其他用户复现与补充截图，没有出现可验证的解决方案；目前能确认的问题点仍集中在“用 `pi05_libero` 代替缺失的 `pi05_base` 配置进行转换”这一临时路径上，暂无明确结论/以讨论为主。

### [Issue #764] cpu memory of Commitment Ratio increasing which causes fine-tuning crash
- 链接：https://github.com/Physical-Intelligence/openpi/issues/764
- 状态：CLOSED
- 问题（中文概述）：训练过程中 CPU memory commitment ratio 持续升高并导致 fine-tuning 崩溃，用户想确认是否是系统性内存泄漏。
- 回答/结论（中文概述）：发帖人后续表示该现象“最近不再出现”，看起来不像稳定复现的大问题；帖子已关闭，但没有给出明确根因或补丁，更接近暂时自愈/偶发现象。

### [Issue #771] Error when using own local dataset to fine-tune pi05
- 链接：https://github.com/Physical-Intelligence/openpi/issues/771
- 状态：OPEN
- 问题（中文概述）：使用自建本地数据集微调 pi0.5 时，`load_dataset("parquet")` 因 0 字节 Parquet 文件报错，用户想确认是不是 openpi/lerobot 侧的问题。
- 回答/结论（中文概述）：社区判断更像是数据转成 LeRobot/Parquet 格式时发生了文件损坏或采集异常，导致读取时命中 0 字节文件；没有进一步官方排查结果，因此可操作结论主要是回头检查数据转换产物。

### [Issue #774] When importing with "import lerobot.common.datasets.lerobot_dataset as lerobot_dataset", the process crashes.
- 链接：https://github.com/Physical-Intelligence/openpi/issues/774
- 状态：OPEN
- 问题（中文概述）：微调自有数据时，进程在 `import lerobot.common.datasets.lerobot_dataset` 处直接退出，用户想知道为何只在当前环境崩溃。
- 回答/结论（中文概述）：社区给出了较明确的版本回退方案：改用指定的 LeRobot 提交 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，然后在该仓库下执行 `pip install -e .`；回复者表示这样后问题即可解决。

### [Issue #782] pi0-fast fine-tuned in my custom dataset doesn't work
- 链接：https://github.com/Physical-Intelligence/openpi/issues/782
- 状态：OPEN
- 问题（中文概述）：用户发现 `pi0-fast` 在自定义数据集上几乎无法产生合理动作，还伴随 token 解码形状错误，想知道是不是训练步数、配置或 LoRA 冻结策略有问题。
- 回答/结论（中文概述）：目前最具体的经验是：LoRA 微调时不要把 action head 冻住，应使用包含 `paligemma_variant="gemma_2b_lora"` 和 `action_expert_variant="gemma_300m_lora"` 的 pi0.5 配置；但线程里没有后续性能验证，仍属于经验性尝试，暂无明确结论/以讨论为主。

### [Issue #809] How to use multiple datasets for PI05 training
- 链接：https://github.com/Physical-Intelligence/openpi/issues/809
- 状态：OPEN
- 问题（中文概述）：询问 `TrainConfig.repo_id` 如果只支持单个路径，那么 pi0.5 训练时如何同时接入多套数据集。
- 回答/结论（中文概述）：帖子里只有其他用户表示有同样需求，没有出现多数据集混训的官方用法、配置示例或代码修改建议，暂无明确结论/以讨论为主。

### [Issue #812] How to run pi05_full_droid_finetune with pytorch?
- 链接：https://github.com/Physical-Intelligence/openpi/issues/812
- 状态：OPEN
- 问题（中文概述）：尝试在 PyTorch 里运行 `pi05_full_droid_finetune` 时命中 “PyTorch RLDS data loader is not supported yet”，想知道有没有可行跑法。
- 回答/结论（中文概述）：从报错与配置可见，这套 full DROID 微调依赖 RLDS 数据加载路径，而当前 PyTorch 训练脚本尚未支持该 loader；讨论中没有替代方案或后续修复，因此暂无明确结论/以讨论为主。

### [Issue #827] During the training process of the PI0.5 model, it automatically exits and the terminal closes without any errors.
- 链接：https://github.com/Physical-Intelligence/openpi/issues/827
- 状态：OPEN
- 问题（中文概述）：训练进行到中途会自动退出、终端直接关闭且无显式错误，用户想知道是 checkpoint、显存还是系统内存问题。
- 回答/结论（中文概述）：协作者优先判断这更像是保存 checkpoint 时触发的主机 RAM OOM，而不是磁盘或 GPU OOM；发帖人随后确认确实是 RAM OOM，并表示通过额外的缓存清理程序后，在相同条件下连续训练 42 小时没有再自动退出。协作者还建议排查 `ocp.PyTreeCheckpointHandler` 的并发保存参数。

### [Issue #831] After running the code "train" in pi05, "Imported typing modules." was displayed and then it directly exited.
- 链接：https://github.com/Physical-Intelligence/openpi/issues/831
- 状态：OPEN
- 问题（中文概述）：运行 `train` 时日志只打印到 “Imported typing modules.” 就直接退出，用户想知道为什么 `uv run` 下没有任何报错信息。
- 回答/结论（中文概述）：社区定位到这是一个导入顺序相关的段错误：若在 `train.py` 中先导入 `openpi.models.model` 再导入 `openpi.training.checkpoints`，可能直接 `Segmentation fault`；将 `_checkpoints` 的导入提前到 `_model` 之前后，有用户确认训练恢复正常。但也有人表示仍会崩溃，因此并非对所有环境都彻底解决。

### [Issue #842] [Feature Proposal] Add LoRA training support for pi0.5 (PyTorch backend)
- 链接：https://github.com/Physical-Intelligence/openpi/issues/842
- 状态：OPEN
- 问题（中文概述）：这是一个功能提案：希望 openpi 提供原生的 pi0.5 PyTorch LoRA 训练支持，以便在 4090 等消费级显卡上完成低显存微调。
- 回答/结论（中文概述）：提案作者表示自己已实现纯 PyTorch LoRA 管线，可在单张 RTX 4090 上以约 16GB 峰值显存完成 Libero 训练，并报告了接近官方水平的评测结果；随后作者说明已把实现整理为后续的 PR #854，可直接跟进该 PR 或其个人分支。

### [Issue #850] Running time is too long for Pi05 fine tuning of LoRA
- 链接：https://github.com/Physical-Intelligence/openpi/issues/850
- 状态：OPEN
- 问题（中文概述）：用户在自建 AU 插接数据集上做 pi0.5 LoRA 微调时，单次训练耗时极长，想确认当前 loss 曲线、训练时长和配置是否正常。
- 回答/结论（中文概述）：讨论里有人建议先把 batch size 从 32 下调到 8，并检查 gripper state 的编码是否与预训练假设一致（常见为 0/1）；发帖人把 batch size 降到 16、单卡训练后，60k step 约需 44 小时且 loss 稳定在 0.022 左右，但仍未形成“这就正常”的统一结论，暂无明确结论/以讨论为主。

### [Pull Request #854] Feat: Add PyTorch implementation of LoRA fine-tuning
- 链接：https://github.com/Physical-Intelligence/openpi/pull/854
- 状态：OPEN
- 问题（中文概述）：这是一条 PR：为 OpenPI 增加原生的 PyTorch LoRA fine-tuning 实现，并打通训练到推理的同一套配置链路。
- 回答/结论（中文概述）：PR 描述给出的结论较完整：新增 `lora_config`、`pi05_libero_lora_pytorch` 配置、LoRA 注入/加载逻辑，以及只优化 `requires_grad=True` 参数的 AdamW 以节省显存；作者还声明已验证单/多卡训练、断点恢复与推理加载链路。目前 PR 仍处于 OPEN 状态。

### [Issue #886] question on custom DROID dataset LoRA finetuning on pi0.5-driod
- 链接：https://github.com/Physical-Intelligence/openpi/issues/886
- 状态：OPEN
- 问题（中文概述）：用户基于 `pi05_droid` 自改出一个面向自定义 DROID 数据集的 LoRA 配置，但效果不理想，想求一套验证过的成功配置。
- 回答/结论（中文概述）：帖子没有任何后续回复或官方示例，因此没有形成可参考的配置结论；能确认的只有“直接改现有 `pi05_droid` 配置未必能得到满意效果”，暂无明确结论/以讨论为主。

### [Issue #892] Issue of fine-tuning pi05_base on the libero dataset.
- 链接：https://github.com/Physical-Intelligence/openpi/issues/892
- 状态：CLOSED
- 问题（中文概述）：用户直接用 `pi05_base` 权重在整个 `libero_spatial` 上做官方脚本微调，发现模型会固化旧任务、难以响应改写后的指令，想知道问题出在微调策略还是数据。
- 回答/结论（中文概述）：该 issue 已关闭，但当前条目中没有可见的解释或修复建议；唯一清晰现象是：同样场景下官方 `pi05_libero` 对改写指令能有 80%-90% 成功率，而用户自行从 `pi05_base` 微调得到的模型泛化明显更差，暂无明确结论/以讨论为主。

### [Issue #692] Single arm model performance is worse on pi0.5 when compared with pi0
- 链接：https://github.com/Physical-Intelligence/openpi/issues/692
- 状态：OPEN
- 问题（中文概述）：单臂机械臂场景下，用户发现 pi0.5 在相近设置下的微调效果明显差于 pi0，想确认是不是配置、batch size、LoRA 或归一化方式的问题。
- 回答/结论（中文概述）：多个用户在 WidowX、Piper 等单臂任务上都复现了“pi0.5 比 pi0 更差”的现象；讨论里唯一稍具体的建议是重新检查归一化，尤其是尝试关闭 `q01/q99` 这类统计项，但没有统一的有效修复方案，暂无明确结论/以讨论为主。

## 性能退化、不收敛与效果不达预期

### [Issue #734] Is PI0.5 finetuned by all LIBERO datasets? Success rate of LIBERO-90 is only 18%?
- 链接：https://github.com/Physical-Intelligence/openpi/issues/734
- 状态：CLOSED
- 问题（中文概述）：用户看到官方列出的 Libero 各子集成功率很高，但自己在 LIBERO-90 上只有约 18%，因此怀疑 pi0.5 是否其实并未在全部 LIBERO 数据上微调。
- 回答/结论（中文概述）：社区给出的解释较一致：官方使用的数据集是 `physical-intelligence/libero`，其中只包含 Libero-Spatial、Libero-Object、Libero-Goal 和 Libero-10，并不包含 LIBERO-90；因此在 LIBERO-90 上表现偏低是可以预期的。

### [Issue #768] Lack of adaptability and language grounding after Pi0.5 fine-tuning
- 链接：https://github.com/Physical-Intelligence/openpi/issues/768
- 状态：OPEN
- 问题（中文概述）：用户在约 250 段、90 分钟的自有 Lerobot 数据上微调 pi0.5 后，虽然简单分拣任务能完成，但模型几乎不能适应物体/容器位置变化，也基本无视语言提示。
- 回答/结论（中文概述）：帖子没有收到后续解答；发帖人自己的判断是数据规模和覆盖范围可能过小，模型更像是在复现训练轨迹而不是学习到视觉与语言条件控制。整体仍是问题陈述，暂无明确结论/以讨论为主。

### [Issue #799] Evaluation on SimplerEnv After Finetuning Pi05 Does Not Meet Expected Performance
- 链接：https://github.com/Physical-Intelligence/openpi/issues/799
- 状态：OPEN
- 问题（中文概述）：用户在 SimplerEnv/自定义 SimpleEnv 任务上评估微调后的 pi0.5，发现结果远低于预期，想知道是参数没调对，还是 benchmark/模型本身存在问题。
- 回答/结论（中文概述）：后续回复主要是更多人复现相似低表现：有人在 4090 上用 LoRA 跑 Libero 也只有约 0.65，有人在自定义 SimpleEnv 任务上表现同样很差；但没有出现明确定位或修复，暂时只能说问题未解，暂无明确结论/以讨论为主。

### [Issue #814] loss explosion and not converge
- 链接：https://github.com/Physical-Intelligence/openpi/issues/814
- 状态：OPEN
- 问题（中文概述）：训练过程中 loss 爆炸且不收敛，用户怀疑数据归一化统计、尤其是 `norm_stats.json` 中接近 0 或为 0 的标准差，可能导致数值不稳定。
- 回答/结论（中文概述）：社区讨论的主线是“零/近零 std 很可能是重要诱因”：有人把 std 中的 0 值替换后训练恢复稳定；但也有人在替换 std、改用 quantile norm 后仍然爆炸，因此目前更像是强相关线索而非单一根因。总体仍无统一修复，暂无明确结论/以讨论为主。

### [Issue #817] The actual performance on the real device is very poor after fine-tuning the pi by 0.5.
- 链接：https://github.com/Physical-Intelligence/openpi/issues/817
- 状态：OPEN
- 问题（中文概述）：用户用自采数据微调 pi0.5 后，在真实机械臂远程推理时轨迹异常，怀疑训练阶段做了归一化而推理阶段直接喂原始状态、直接执行原始动作，导致输入输出分布不一致。
- 回答/结论（中文概述）：帖子里没有给出后续解决方案，只有其他用户表示在 PyTorch 版上遇到同样问题；当前最核心的怀疑点就是推理侧缺少与训练一致的归一化/反归一化处理，但暂无明确结论/以讨论为主。

### [Issue #386] Deploying openpi0 on jetson orin encountered an error
- 链接：https://github.com/Physical-Intelligence/openpi/issues/386
- 状态：OPEN
- 问题（中文概述）：在 Jetson Orin 上部署 openpi0/openpi0.5 时，`serve_policy.py` 会卡在模型加载/恢复阶段，用户想知道这是内存、JAX 兼容性还是部署方式的问题。
- 回答/结论（中文概述）：协作者先指出原帖里并没有明确报错，只是卡住，并追问 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 是否有效；社区随后反馈该设置在部分 Orin 机型上能减轻统一内存预分配带来的交换开销，但并非通用解法。还有人认为根因与 JAX 在 Jetson GPU 上兼容性有限有关；同时也有人报告在 Orin 64GB DK + JetPack 6 上已成功部署并达到约 1 秒单次推理。整体来看问题与平台差异强相关。

## 推理部署、硬件适配与运行时兼容

### [Issue #582] Malfunction on
- 链接：https://github.com/Physical-Intelligence/openpi/issues/582
- 状态：OPEN
- 问题（中文概述）：用户在 Jetson AGX Orin 上按文档部署后运行 `serve_policy -libero`，触发 `cudaErrorNoKernelImageForDevice`，怀疑是当前 CUDA/JAX/设备架构组合不兼容。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；有用户复现同样报错，另有社区成员反馈已在 Jetson Orin 64GB DK + JetPack 6 上把 pi0.5 跑通，单次推理约 1 秒，并分享了中文部署博客。

### [Issue #657] Deployment on edge devices
- 链接：https://github.com/Physical-Intelligence/openpi/issues/657
- 状态：OPEN
- 问题（中文概述）：用户想了解是否有在 Nvidia Jetson Orin、Thor 等边缘设备上部署 pi0.5 的官方实践或推荐方案。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；社区有人给出通用 Jetson 部署建议，也有人分享 Thor/Orin 实测经验：Thor 上能跑 `examples/simple_client/main.py` 但 JAX 与 CUDA 13 仍有问题，Orin 64GB DK + JetPack 6 上已有 pi0.5 成功部署案例。

### [Issue #678] jax pi05 inference failed on 4090
- 链接：https://github.com/Physical-Intelligence/openpi/issues/678
- 状态：OPEN
- 问题（中文概述）：用户在 RTX 4090 上做 JAX 版 pi0.5 推理时失败，日志显示后端初始化与 checkpoint 加载过程异常，想确认是否有人解决过。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；线程里只有另一位用户追问“是否已经解决”，没有给出修复路径或官方解释。

### [Issue #684] Inference not working on Lerobot SO-101
- 链接：https://github.com/Physical-Intelligence/openpi/issues/684
- 状态：OPEN
- 问题（中文概述）：用户在 LeRobot SO-101 机械臂上做 pi0.5 推理时，机械臂只会轻微抖动且几乎不动，怀疑是显存不足或推理链路有问题。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；社区反馈这更像模型/适配问题而不是显存问题，因为 3090、4090、A100 上也出现相同行为，有人怀疑与 LeRobot 集成代码或 JAX 转 PyTorch 权重效果有关。

### [Issue #736] error occurs when run remote inference
- 链接：https://github.com/Physical-Intelligence/openpi/issues/736
- 状态：OPEN
- 问题（中文概述）：用户执行远程推理命令时加载 checkpoint 报错，不确定是命令写法还是模型目录选错。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；有用户建议把 `--policy.dir` 从 `gs://openpi-assets/checkpoints/pi0_base` 改成 `gs://openpi-assets/checkpoints/pi05_libero`，提问者确认这样可以运行，但为什么 `pi0_base` 不能直接用于这条命令并未解释清楚。

### [Issue #765] Vectorized inference for pi0 and pi0.5 pytorch version
- 链接：https://github.com/Physical-Intelligence/openpi/issues/765
- 状态：OPEN
- 问题（中文概述）：用户发现 JAX 版 pi0 不支持 batch size 大于 1 的向量化推理，因此追问最新的 PyTorch 版 pi0 / pi0.5 是否已经支持。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该条目没有任何回复，线程里没有给出能力边界或实现现状。

### [Issue #766] Gripper close too early
- 链接：https://github.com/Physical-Intelligence/openpi/issues/766
- 状态：OPEN
- 问题（中文概述）：用户在 pybullet 仿真里做抓取放置任务时，发现 gripper 还没到方块就提前闭合，导致整体成功率偏低。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；社区怀疑问题与较长 action horizon 和非实时 chunk 执行有关，模型会一次性给出过于“乐观”的长轨迹，在仿真数据上更明显；经验上 pi0.5 diffusion 效果最好，缩短 horizon 可能提升精度但并非稳定解法。

### [Issue #830] Pi0.5 Inference in agilex cobot magic
- 链接：https://github.com/Physical-Intelligence/openpi/issues/830
- 状态：OPEN
- 问题（中文概述）：用户在 AgileX Cobot Magic 上跑 pi0.5 推理时，夹爪始终无法张开，想确认是否与控制模式、关节映射、缩放或归一化统计有关。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；线程没有任何回复，问题仍停留在现象描述与归一化统计数据展示阶段。

### [Issue #841] Non‑deterministic inference across episodes with identical inputs/noise
- 链接：https://github.com/Physical-Intelligence/openpi/issues/841
- 状态：CLOSED
- 问题（中文概述）：用户在 SimplerEnv 中评估微调后的 pi0.5 时，发现同样输入、同样 seed、同样噪声下，第一局正常，后续 episode 却会输出不同动作甚至出现 NaN，怀疑存在隐藏状态或缓存污染。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；不过提问者后来定位到根因是 `imageio` 录视频时通过 `os.fork()` 拉起 ffmpeg，而 JAX 是多线程的，fork 会导致不安全行为和非确定性/NaN；关闭视频保存后问题消失，建议改为离线编码视频。

### [Issue #620] Max supported action dimension and number of input images？
- 链接：https://github.com/Physical-Intelligence/openpi/issues/620
- 状态：CLOSED
- 问题（中文概述）：用户想知道 pi0/pi0.5 在微调时最多支持多少维动作和多少路图像输入，例如 28 维动作与 4 路相机是否可行。
- 回答/结论（中文概述）：协作者 `kpertsch` 明确建议改用 pi0.5，并表示它预训练时支持最多 4 路相机（两路外部相机加两路腕部相机）；但后续用户指出当前 PyTorch 预处理仍只接受 3 个固定图像键，因此 4 相机在 PyTorch 版里的实际支持情况仍未被完全澄清。

## 数据接口、状态/动作表示与归一化

### [Issue #637] State/Action representation in LIBERO dataset (EEF vs Joint)
- 链接：https://github.com/Physical-Intelligence/openpi/issues/637
- 状态：OPEN
- 问题（中文概述）：用户对 LIBERO 数据集中 state/action 的表示方式感到困惑，无法判断它们究竟是关节空间还是末端执行器空间，以及 `extra_delta_transform` 在这里到底意味着什么。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；社区倾向认为 OpenPI 沿用了 OpenVLA 重新生成的 LIBERO 约定，动作更像 delta EEF，`extra_delta_transform=False` 也暗示数据本身已是 delta 形式；但 state 的精确定义、action 是绝对量还是相对量、以及为何数值尺度看起来偏大，都没有得到官方最终澄清。

### [Issue #650] question about the state as input for pi05
- 链接：https://github.com/Physical-Intelligence/openpi/issues/650
- 状态：CLOSED
- 问题（中文概述）：用户认为 pi0.5 似乎没有使用 state 输入，这与论文描述不一致，因此追问 state 在模型里究竟如何接入。
- 回答/结论（中文概述）：协作者先说明 pi0.5 其实支持 state，但会把 state 离散化成 token，经 VLM 路径输入，而不是把连续 state 直接送入 action expert；后续又补充说已发布 checkpoint 采用单阶段训练，至于为何不把 state 接到 action expert，只留下训练流程层面的解释，没有更完整的官方消融结论。

### [Issue #662] Compute norm stats error about data format
- 链接：https://github.com/Physical-Intelligence/openpi/issues/662
- 状态：OPEN
- 问题（中文概述）：用户把数据转成 LeRobot 格式后运行 `compute_norm_stats.py`，因为图像字段变成带 `byte`/`path` 的字典，导致 `hf_transform_to_torch` 无法推断 dtype 并报错。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；后续只有其他用户补充自己也遇到同样问题，没有给出修复方法或确认是转换脚本还是依赖版本导致。

### [Issue #680] state not used in PI05 forward
- 链接：https://github.com/Physical-Intelligence/openpi/issues/680
- 状态：CLOSED
- 问题（中文概述）：用户在阅读 PI05 的 forward 逻辑时，感觉 state 只在 `embed_suffix` 中绕了一圈，最终似乎没有真正参与计算。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；提问者后来自己更正，表示是忘了 state 会先被编码成 language tokens，并直接引用 #650 作为解释依据。

### [Issue #687] pi05-libero discrete_state_input = False
- 链接：https://github.com/Physical-Intelligence/openpi/issues/687
- 状态：OPEN
- 问题（中文概述）：用户发现 `pi05-libero` 配置里把 `discrete_state_input` 设成了 `False`，因此怀疑该配置是否根本没有把 state 输入给 pi0.5。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；大量用户连续追问，但线程没有维护者或协作者给出正式说明，仍停留在猜测阶段。

### [Issue #711] PI05 lora fine-tune on libero dataset & differences about the norm_stats
- 链接：https://github.com/Physical-Intelligence/openpi/issues/711
- 状态：OPEN
- 问题（中文概述）：用户尝试用 LoRA 复现 pi0.5 在 LIBERO 上的效果，但微调后成功率只有约 1%，同时自己算出的 `norm_stats` 与官方资产差别很大，想确认配置是否有误。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；有社区用户分享了一份自己可用的 LoRA 配置，核心是沿用 `pi05_base` 权重、关闭 EMA、按官方风格设置 freeze filter 与学习率，并把训练步数设回 30k；但 `norm_stats` 差异为什么出现并未得到官方解释。

### [Issue #741] how to fine tune model on dataset of more images and states
- 链接：https://github.com/Physical-Intelligence/openpi/issues/741
- 状态：OPEN
- 问题（中文概述）：用户已经能按 LIBERO 示例微调 pi0.5，但自己的数据有 3 路图像和更高维的 state，想知道配置文件应如何修改。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该条目没有收到任何回复，尚无现成配置建议。

### [Issue #763] Question about using quantiles in pi 0.5 when fine tune.
- 链接：https://github.com/Physical-Intelligence/openpi/issues/763
- 状态：OPEN
- 问题（中文概述）：用户怀疑 pi0.5 在微调时默认启用的 `use_quantile_norm` 会显著拉低效果，甚至希望把它改成可配置项。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；多位用户复现了 quantile norm 比均值方差归一化更差的现象，还有人指出当某一维 `q01` 与 `q99` 非常接近时会引发 loss 爆炸，并提醒即使在 `TrainConfig` 里设成 `False`，如果 `DataConfigFactory` 仍强制开启，问题依然可能存在。

### [Issue #773] question about computation of the norm stats
- 链接：https://github.com/Physical-Intelligence/openpi/issues/773
- 状态：OPEN
- 问题（中文概述）：用户发现微调 LIBERO 时通常会先跑 `compute_norm_stats.py`，但微调基于 DROID 的自定义数据时即便跳过这一步也不会报错，因此想确认这是必须步骤还是仅仅建议。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；线程里没有维护者答复，只有提问者自己贴出的外部 AI 建议，核心疑问仍是跳过该脚本时是否只是沿用了原始 DROID/base 的统计量，以及面对 Genesis 这类自定义模拟数据时是否应该重算归一化统计。

### [PullRequest #821] Multi-RLDS Datasets (Cotraining) + PolaRiS DROID jointpos policy configs
- 链接：https://github.com/Physical-Intelligence/openpi/pull/821
- 状态：MERGED
- 问题（中文概述）：这个 PR 想把多 RLDS 数据集协同训练能力以及 PolaRiS 的 DROID joint-position policy 配置并入主线。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；线程内容几乎只有 “WIP” 占位，但从状态上看该 PR 已合并，说明相关改动最终进入了仓库。

### [Issue #848] Joint Velocity Chunk Definition
- 链接：https://github.com/Physical-Intelligence/openpi/issues/848
- 状态：OPEN
- 问题（中文概述）：用户质疑 `pi05-droid` 中把关节速度 chunk 转成绝对关节位置时所采用的公式，想知道到底该用相对初始位姿积分还是逐步递推积分。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该条目没有任何回复，尚无官方说明应以哪种定义为准。

### [PullRequest #855] Fix hardcoded action dim in pi0 pytorch model
- 链接：https://github.com/Physical-Intelligence/openpi/pull/855
- 状态：OPEN
- 问题（中文概述）：这个 PR 指出 PyTorch 版 pi0 把 action 维度硬编码成了 32，导致在非 32 维动作设置下训练或推理会出错。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；PR 的主张是改为统一读取 `config.action_dim`，从而和 JAX 实现保持一致，并顺带声称可修复 #714，但截至当前仍处于打开状态。

### [Issue #863] About pi05_base
- 链接：https://github.com/Physical-Intelligence/openpi/issues/863
- 状态：OPEN
- 问题（中文概述）：用户理解 pi0.5 预训练阶段依赖 FAST action tokenizer 的离散动作，因此不明白为什么 `pi05_base` 的模型说明却把输出写成连续动作和 flow matching。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；有社区用户解释为 base 阶段先用 FAST token 做自回归预训练，而后训练阶段再用 flow matching 生成连续动作，因此模型卡强调的是后者输出形式，但线程里没有维护者正式确认。

### [PullRequest #871] add action mask in loss computation
- 链接：https://github.com/Physical-Intelligence/openpi/pull/871
- 状态：OPEN
- 问题（中文概述）：这个 PR 认为当前 PyTorch 版 π 模型在计算 MSE loss 时把 padding 出来的动作维度也算进去了，可能会拖累训练效果。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；PR 提议新增 `actual_action_dim` 和 `action_mask`，在 loss 里屏蔽 padded action，思路参考了 LeRobot 的实现，但目前还没有评审结论。

### [Issue #260] Release original checkpoints on Hugging Face
- 链接：https://github.com/Physical-Intelligence/openpi/issues/260
- 状态：CLOSED
- 问题（中文概述）：Hugging Face 团队建议把原始 checkpoints 和数据集上传到 Hub，以便提升可发现性、统一下载方式并关联论文页面。
- 回答/结论（中文概述）：协作者 `uzhilinsky` 回复说目前仍倾向把 checkpoints 放在 S3，公开数据集则继续依托 LeRobot；也就是说 Hub 上传建议被记录了，但短期内并没有立即迁移的计划。

## 权重/checkpoint/数据资源的发布、下载与访问

### [Issue #260] Release original checkpoints on Hugging Face

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/260)
- 状态：`CLOSED`
- 问题（中文概述）：Hugging Face 团队建议把原始 checkpoint 和数据集迁移或同步到 HF Hub，方便检索、下载和展示论文关联资产。
- 回答/结论（中文概述）：协作者回复称目前继续把 checkpoint 放在 S3 即可，公开数据则已通过 LeRobot 体系提供，也有上传新数据的说明；未承诺近期把权重迁到 HF Hub。

### [Issue #465] pi0.5 model checkpoint and finetuning code

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/465)
- 状态：`CLOSED`
- 问题（中文概述）：用户询问 pi0.5 模型权重和微调代码是否会很快公开发布。
- 回答/结论（中文概述）：协作者明确表示目前没有新的可分享消息；后续若有发布或更新，会先出现在该仓库和 `pi.website`。

### [Issue #476] Will pi0.5 opensource?

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/476)
- 状态：`CLOSED`
- 问题（中文概述）：用户直接询问 pi0.5 是否会开源。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；评论区主要在讨论替代方案（如 GR00T N1.5）以及可关注的相关开发工作 `PR #495`，没有官方开源时间表。

### [Issue #597] Question about downloading checkpoint of pi0.5

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/597)
- 状态：`CLOSED`
- 问题（中文概述）：用户尝试下载 `gs://openpi-assets-preview/checkpoints/pi05_droid/params` 时遭遇 GCS 权限错误，怀疑 checkpoint 仍是私有的。
- 回答/结论（中文概述）：协作者回复称 Pi05 支持当时仍处于 closed beta，完整公开发布尚未完成；因此访问被拒绝并非普通下载姿势问题，而是资源尚未全面开放。

### [Issue #632] Access to pi05_libero_may21 checkpoint

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/632)
- 状态：`CLOSED`
- 问题（中文概述）：用户下载 `pi05_libero_may21` 预览 checkpoint 时收到 GCS `AccessDenied`，想确认该权重是否尚未对公众开放。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；帖子没有后续回复，但从报错现象看，问题大概率是该 checkpoint 当时未公开授权访问。

### [Issue #649] Does the current version (codes instead of checkpoints) use knowledge insulation during fine-tuning when using train.py?

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/649)
- 状态：`CLOSED`
- 问题（中文概述）：用户确认当前 `train.py` 微调流程是否会使用 knowledge insulation，以及是否只训练 action expert、是否应把 `paligemma_variant` 设为 `None`。
- 回答/结论（中文概述）：协作者说明当前 openpi 微调并未实现 knowledge insulation，也没有论文里的 FAST+flow-matching 联合损失；现有实现只支持 action expert 的 flow-matching loss，但梯度会同时回传到 action expert 和 VLM backbone。pi0.5 预训练 checkpoint 使用过 KI，但微调阶段官方尚未在仓库里提供该实现。

### [Issue #651] Pytorch version PI0.5 model mismatchs the jax converted pytorch checkpoint.

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/651)
- 状态：`CLOSED`
- 问题（中文概述）：用户在加载由 JAX 转出的 pi0.5 PyTorch checkpoint 时遇到大量 `Invalid key(s)`、dtype 或 shape 不匹配问题。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；有用户反馈改为严格按 README 使用 `uv` 安装而不是只靠 conda 后问题消失，但后续仍有多人在 `pi05_libero` 等场景中复现，官方没有统一修复说明。
合并条目：Issue #729（`OPEN`）[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/729)

### [Issue #669] paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight missed in p05 checkpoint

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/669)
- 状态：`CLOSED`
- 问题（中文概述）：用户把 `pi05_base` 转成 PyTorch 并用于 `pi05_libero` 训练时，发现 `embed_tokens.weight` 缺失。
- 回答/结论（中文概述）：评论区给出的可行处理是做 weight tying；发帖人随后确认用这种方式已解决，但没有维护者补充更正式的原因解释。

### [Issue #670] Question about pi05 checkpoint

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/670)
- 状态：`CLOSED`
- 问题（中文概述）：用户质疑已公开的 `pi05_base` 是否真的具备 subtask generation 能力，因为自己用原始文本测试时只得到乱码样输出。
- 回答/结论（中文概述）：发帖人随后自查发现是自己代码里有 bug，并改口确认 `pi05_base` 至少能生成一些简单的 subtask；因此该问题属于自排查后基本解决。

### [Issue #674] cannot find the download dataset

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/674)
- 状态：`OPEN`
- 问题（中文概述）：用户运行 `compute_norm_stats.py --config-name pi05_libero` 后，不清楚 LIBERO 数据集究竟被下载到了哪里。
- 回答/结论（中文概述）：评论区指出该脚本的直接产物是把数据均值/方差写入 config assets 目录；数据集位置本身由 `src/openpi/training/config.py` 中的配置决定，并不是这个脚本额外生成一个显式的新数据目录。

### [Issue #690] Unable to download Pi0.5 Model

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/690)
- 状态：`CLOSED`
- 问题（中文概述）：用户把 GCS 路径误当成 Google Drive 链接，用 `gdown` 下载 pi0.5 模型失败。
- 回答/结论（中文概述）：协作者明确说明这是 GCS 文件而不是 GDrive 链接，应改用 `gsutil` 或 gcloud API 下载，而不是 `gdown`。

### [Issue #721] CPU Memory Leak When Saving Checkpoints During Fine-tuning

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/721)
- 状态：`OPEN`
- 问题（中文概述）：用户反馈官方 `train.py` 微调时每次保存 checkpoint 后 CPU 内存都会逐步上涨，最终因 OOM 崩溃。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；后续评论只有催看和复现反馈，没有维护者给出根因分析或修复方案。

### [Issue #735] The checkpoint without the knowledge insulation?

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/735)
- 状态：`OPEN`
- 问题（中文概述）：用户希望官方另外发布一个“不使用 knowledge insulation”训练出来的 pi0.5 checkpoint，以便和当前公开版本做对照。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该帖没有任何后续答复，也没有看到额外 checkpoint 的发布承诺。

### [Issue #751] Does pi05_base contain weights for action expert?

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/751)
- 状态：`OPEN`
- 问题（中文概述）：用户想确认 `pi05_base` 里是否已经包含 action expert 权重，因为论文提到微调阶段的 action expert 是从头训练。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；有社区用户根据论文训练阶段和参数检查推断 `pi05_base` 应该已经包含 action expert 权重，但这不是官方确认。

### [Issue #752] Running `aloha_mobile`

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/752)
- 状态：`OPEN`
- 问题（中文概述）：用户询问是否会发布微调后的 `aloha_mobile`，并试图直接用 `pi05_base + pi05_aloha_mobile` 配置测试导航能力，但得到接近零的 base velocity 输出。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；帖子没有人确认这种“base 模型直接套 mobile 配置”的用法是否有效，也没有说明 `aloha_mobile` 权重的公开计划。

### [Issue #779] Using Droid checkpoint

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/779)
- 状态：`OPEN`
- 问题（中文概述）：用户想使用 pi0.5 的 Droid checkpoint，但不确定该配哪个 config、该用哪个数据仓库，以及网络输出到底是 joint velocity 还是 joint position。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该帖没有后续答复，相关使用细节没有在这个条目里得到澄清。

### [PR #784] feat: added skip norm stats

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/pull/784)
- 状态：`OPEN`
- 问题（中文概述）：该 PR 提议增加 `--skip-norm-stats` 选项，用单位变换替代 norm stats，方便直接复用 LeRobot 的 pre/post-processing 权重。
- 回答/结论（中文概述）：PR 作者表示自己已用转换后的 LeRobot 权重在 openpi 客户端上跑通，但该 PR 仍处于 `OPEN`，没有维护者给出正式评审或合并结论。

### [Issue #804] Poor training performance (3.3% success rate) with pi0 model on locally downloaded official LIBERO dataset

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/804)
- 状态：`OPEN`
- 问题（中文概述）：用户用本地下载的官方 LIBERO 数据集和 PyTorch 版 `pi0` 训练，成功率只有约 3.3%，怀疑数据加载、LoRA 配置或训练管线存在根本性问题。
- 回答/结论（中文概述）：讨论里最有操作性的结论是：在 `pi0_torch` 配置下应显式设置 `pytorch_weight_path` 来加载预训练 PyTorch 权重，`weight_loader` 只对 JAX 版生效，否则等价于从零开始训练。与此同时，也有多人反馈即便训练成功，PyTorch 版的实际表现仍明显低于论文或官方 JAX 结果。

### [Issue #810] Pi05 output mismatch between PyTorch and Jax weights

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/810)
- 状态：`CLOSED`
- 问题（中文概述）：用户发现同一输入下，JAX 权重和转换后的 PyTorch 权重推理输出差异很大，怀疑权重转换有问题。
- 回答/结论（中文概述）：发帖人后续表示转换权重本身没有问题，关键是测试时要同时固定 PyTorch 和 JAX 侧的随机种子与噪声；若不控制随机性，就会看到明显偏差。

### [Issue #826] Jetson Thor support for Pi05 - Error:"Invalid key(s) in state_dict" despite patching transformers with transformers_replace files when running basic inference with policy_config.create_trained_policy(config, checkpoint_dir) and policy.infer(example_obs)

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/826)
- 状态：`OPEN`
- 问题（中文概述）：用户希望在 Jetson Thor 上运行 Pi05 推理并逼近论文中的板载时延，但当前在完成 `transformers_replace` 补丁后仍卡在 `Invalid key(s) in state_dict)` 的模型加载错误。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；帖子详细描述了 Thor/Orin、Python 版本和容器环境，但没有维护者确认具体是 transformers 兼容性、平台差异还是权重转换链路问题。

### [Issue #840] The model's performance dropped significantly when converting the model's checkpoint from the JAX version to the PyTorch version

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/840)
- 状态：`OPEN`
- 问题（中文概述）：用户表示 JAX 版本训练效果很好，但转成 PyTorch 后机器人动作变得异常，想知道问题可能出在转换、加载还是 PyTorch/Triton 环境上。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；评论区只有社区成员猜测可能与 mixed precision、`state_dict` 严格加载或相关无效 key 问题有关，但没有经过验证的统一修复路径。

### [Issue #845] maybe_download race condition resulting in downloading file multiple times

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/845)
- 状态：`OPEN`
- 问题（中文概述）：用户定位到 `maybe_download` 在并发场景下会重复下载同一个大文件，怀疑是存在 race condition。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该帖本身没有评论，但提问者已经把根因指向“加锁前先做 exists 检查”，并给出了可复现脚本；后续已有 `PR #846` 尝试修复。

### [PR #846] Fix race condition in maybe_download

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/pull/846)
- 状态：`OPEN`
- 问题（中文概述）：该 PR 针对 `maybe_download` 的并发重复下载问题，提议在获取文件锁之后再次检查文件是否已存在。
- 回答/结论（中文概述）：PR 给出了复现代码和修复前后的下载日志，对应思路是让并发线程只有一个真正执行下载；不过该修复尚未被合并，线程里也没有维护者正式确认。

### [Issue #860] Pi 0.6 star mental model

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/860)
- 状态：`OPEN`
- 问题（中文概述）：用户尝试从 pi0.5 checkpoint 反推 pi0.6 star 的训练/筛选流程，逐步描述了奖励合成、value function、advantage 二值化和重训练过程，想确认理解是否正确。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该帖没有任何后续回复，未得到官方对这套“mental model”的确认或纠正。

### [Issue #883] `jointpose` checkpoints quality

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/883)
- 状态：`OPEN`
- 问题（中文概述）：用户询问 `pi05_droid_jointpos` 系列 checkpoint 是否使用 pose 控制、部署时是否要改 droid controller 类型、该用哪个 config，以及它和原始 `pi05_droid` 的质量对比如何。
- 回答/结论（中文概述）：社区回复认为在 droid 代码路径里 velocity 最终会积分成 delta position，因此本质上仍接近 pose 控制；其建议配置名是 `pi05_droid_jointpos`，并可自行把 joint position 目标近似转换回 velocity。在其硬件/软件环境里，`pi05_droid_jointpos` 表现明显好于 `pi05_droid`，但这不是官方结论。

### [Issue #887] Open-sourcing the pi0.5 VLM backbone

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/887)
- 状态：`OPEN`
- 问题（中文概述）：用户希望公开 pi0.5 的 VLM backbone，理由是高层输出有助于理解 flow-matching head，也可能提升微调灵活性。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该帖没有维护者回应，也没有给出是否会近期开源 VLM backbone 的时间表。

## 模型结构、预训练与实现细节追问

### [Issue #708] Question about the fast tokenizer in pi05

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/708)
- 状态：`CLOSED`
- 问题（中文概述）：用户担心 pi0.5 在混合 web 数据和机器人数据训练时，若 FAST action token 复用 PaliGemma 的 `<location>` 一类 token 范围，会与 grounding/VQA 任务产生冲突。
- 回答/结论（中文概述）：协作者澄清：pi0.5 实际占用的是 PaliGemma 词表中“很少使用”的 token（通常是冷门 Unicode 字符），因此与 web 数据常见目标基本不会相互干扰；后续还补充了相关 token range 的常量定义。

### [Issue #718] Stage 1 Training Design in Pi0.5 Model

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/718)
- 状态：`CLOSED`
- 问题（中文概述）：用户询问 pi0.5 第一阶段预训练时，VLM 侧到底使用双向注意力还是因果注意力。
- 回答/结论（中文概述）：协作者明确说明：图像和 prompt 部分使用 bidirectional attention，其余部分使用 causal attention，这与 PaliGemma 的预训练方式保持一致。

### [Issue #759] Was the data used for pi0.5 training based on the lerobot2.0 or lerobot3.0 version?

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/759)
- 状态：`OPEN`
- 问题（中文概述）：用户想知道 pi0.5 训练所用数据对应的是 LeRobot 2.0 还是 3.0 版本。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；评论区只有其他用户附和提问，没有维护者澄清具体版本。

### [Issue #761] Do the pretrained base models of pi0.5 include mobile manipulation data?

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/761)
- 状态：`OPEN`
- 问题（中文概述）：用户询问已开源的 pi0.5 预训练 base model 是否已经包含 mobile manipulation 数据。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该帖没有官方答复，只有其他用户表示也关心这个问题。

### [Issue #816] Inquiries regarding Tokenizer and Loss Function implementation for pi0.5

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/816)
- 状态：`OPEN`
- 问题（中文概述）：用户对照论文和代码后，质疑为何 `pi05=True` 时看起来调用的是 `PaligemmaTokenizer` 而不是论文所说的 FAST tokenizer，以及代码里为何只看到 flow-matching loss、看不到论文公式中的联合损失实现。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；该帖没有后续评论，所以这两个“论文与代码实现是否一致”的问题在该条目里未被正式回应。

### [Issue #836] FAST model pre-training single-arm and dual-arm data

- 链接：[GitHub 链接](https://github.com/Physical-Intelligence/openpi/issues/836)
- 状态：`OPEN`
- 问题（中文概述）：用户追问 pi0.5 FAST 预训练在混合单臂/双臂数据时，padding 出来的动作维度是否会在 loss 中被 mask，以及解码后如何恢复正确的动作维度。
- 回答/结论（中文概述）：暂无明确结论/以讨论为主；帖子没有任何答复，因此关于单臂/双臂动作维度 masking 与 FAST decode 行为的细节仍未在这里得到说明。
