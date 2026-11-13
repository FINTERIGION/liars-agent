# Tinker 小试闭环准备记录

日期：2026-11-14。协议：`trial_training_sampling_closure_v2`。

本地代码和数据准备已完成，可以进入受控的真实小试。正式训练仍须先通过真实小试：本轮没有登录 Tinker、创建模型会话或运行付费训练，也没有运行本地模型实验、重新生成数据或续玩。下述测试使用明确的内存服务替身；不能据此声称真实权重已经更新、真实 checkpoint 已验收或胜率得到改善。

## 数据与监督

沿用 `outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame`：10,000 条动作、10,000 条信念，按完整对局划分。全部并列答案每轮参与监督，按状态总权重分摊；小分差动作降权，EOS 边界和信念单 token 目标不变。正式训练仍为 3 epochs、1,500 次优化器更新。

训练侧残局补充、148 条独立检查集、全量 tokenizer 检查及已有费用统计见 [v3 验收](training_readiness_validation.md)。本轮逐一核对该记录绑定的 7 个数据／配置文件 SHA-256，全部相同；没有重跑数据生成或全量分词。

本轮另外使用缓存的真实 Qwen tokenizer 抽查 5 条观测：普通 clear 动作、tied 动作、补充残局强制动作、补充残局自由动作和 belief 各一条，构建 6 个 SDK Datum（并列动作检查两个答案）。动作首 EOS、监督 mask 和信念单标签 mask 均通过，抽查最大输入 2,277 token。该抽查只补充当前接口检查，不替代已有全量数据验收。

## 小试实际会检查什么

1. 固定抽取每任务最多 2 条验证观测，训练前重复 forward，测量同一权重下的分数抖动。按完整三轮的学习率计划运行指定的 5 步，不缩短 warmup。
2. 保存训练状态和 sampler 权重；检查更新后固定探针的分数变化。整体探针和信念探针均须超过基线噪声门槛，同时超出重载匹配容差，避免把无法区分的旧权重误判为成功更新。
3. 从保存路径新建 sampler，检查原生动作 JSON 和合法 ID；分别比较两种信念标签的原始 log probability 与保存时训练端结果。不会用规则回退掩盖非法输出，也不会仅比较归一化后概率。
4. 保留原训练客户端，另建完整状态／优化器恢复客户端及仅恢复权重的负对照。三个分支各对同一个下一 batch、同一个学习率更新一次；原客户端和完整恢复客户端应匹配，权重恢复负对照须在匹配容差及基线噪声门槛之外可区分。

闭环总计额外 7 组固定探针 forward、3 次诊断更新、2 次状态加载，另计 sampler 双标签参考 forward 和推理调用。诊断更新不覆盖已保存的 5 步 checkpoint，也不计入正式训练步数。这是有限探针的功能检查，不是所有参数相等的证明，不要求 5 步降低验证 loss。

默认匹配容差 `atol=rtol=1e-4`，更新与负对照差值还须超过 `max(1e-6, 3 × baseline_jitter)`。信号不足意味着本次未验证成功；程序不会自动放宽容差、追加训练步数或重提交付费请求。

过程记录写入 `models/sft_trial/trial_verification.json`，包括探针身份、调用阶段、已保存路径及失败原因。服务失败保留已完成的记录和未知结果的费用预留。全部检查通过后才写入 `trial_metadata.json`。正式 `sft` 在创建收费会话前检查小试协议、数据指纹、基础模型、LoRA 和训练配置；旧的保存成功标记不足以放行。小试权重不会自动用作正式 SFT 或 RS。

## 本轮验证

受影响模块的小型离线测试：**85 passed、64 subtests passed**。覆盖不更新、错误重载、优化器丢失、低于容差的伪更新、负对照低于噪声、sampler 缓存替换、非法动作、错误原始标签分数、失败预算、部分保存记录及正式训练门禁。费用测试逐项比较执行账本与估算 token，动作输出按最大长度计上界。

```bash
.venv/bin/python -m pytest \
  tests/test_trial_pipeline.py tests/test_trial_verification.py \
  tests/test_trial_sensitivity.py tests/test_trial_sampling.py \
  tests/test_supervision_cost.py tests/test_uncertain_training.py \
  tests/test_refresh_trial_estimate.py tests/test_experiment.py \
  tests/test_backend_retry.py tests/test_tinker_backend.py \
  tests/test_action_validation.py -q
```

独立只读复核确认调用顺序与估算一致。没有重复运行全项目实验；已有数据与规则证据保持原适用范围。机器可读摘要见 [trial_closure_v2.json](validation/trial_closure_v2.json)。

## 更新后的费用

复用原 v3 验收绑定的五步主体统计，核对数据哈希、监督口径、计划和费率后，仅对 4 条验证探针及 1 个逻辑 batch 分词，计算额外闭环调用。费率沿用仓库 2026-11-14 快照：训练 $1.463、prefill $0.300、采样 $0.807／百万 token；这里是 token 估算，不是服务商账单。

| 项目 | token 计数 | 估算美元 |
| --- | ---: | ---: |
| 5 步训练与第 0／5 步诊断 | 3,380,733 | 4.946012379 |
| 7 组探针、3 次诊断更新及标签参考 forward | 294,316 | 0.430584308 |
| sampler：2 次动作、4 次标签打分 | prefill 10,897；生成最多 516 | 0.003685512 |
| 完整小试 | — | **5.380282199** |

正式三轮 SFT 加既定验证仍为 **$286.536586259**；小试单独计费，独立残局评测另列。以上不含 checkpoint 存储、RS 和完整对局评测，累计预算上限仍为 $5,000。本次真实调用为 0，共享账本仍为空。

本机已有数据的小范围估算刷新命令（本轮已执行，不调用模型）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/refresh_trial_estimate.py \
  --run-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame
```

该脚本会拒绝不匹配的源文件或历史统计。`trial_cost_estimate.json` 中 `trial_total_usd_upper_bound` 为完整报价，`sft_train_and_validation_usd` 保留主体费用含义。

未来获准真实调用并核对账号权限后，在同一数据目录运行以下小试；本轮未执行：

```bash
.venv/bin/python scripts/run_full_pipeline.py \
  --config outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame/config.json \
  --output-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame \
  --stage sft_trial --max-train-steps 5
```

通过后才使用相同配置和目录执行 `--stage sft`。真实更新可辨别性、采样／训练数值一致性、模型权限和服务端优化器恢复仍待该小试验证；训练效果则需后续冻结测试集、独立残局检查及配对对局评测回答。
