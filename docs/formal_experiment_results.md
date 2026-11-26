# 正式实验结果汇总（v3：Qwen3.5-9B LoRA-SFT / RS）

> **最新实验更新**：基于 `adaptive_v4` 后验教师与 Tinker 在线 Paired GRPO/PPO RL 的 v4 正式实验（`seen 92.22%` / `unseen 76.67%`）已归档至 [`docs/formal_experiment_results_v4.md`](formal_experiment_results_v4.md)。

- **运行目录**：`selfplay_v2_20261113_01_quality_supervision_v3_endgame`
- **归档产物**：[`results/selfplay_v2_20261113_01_quality_supervision_v3_endgame/`](../results/selfplay_v2_20261113_01_quality_supervision_v3_endgame/)
- **完成时间**：2026-11-24T01:08:52Z（`real_experiments: complete`）
- **基座与平台**：`Qwen/Qwen3.5-9B`，Tinker LoRA（`r=16`, `train_attn=true`, `train_mlp=true`, `train_unembed=false`）
- **方法合同**：`liars_bar_research_v2` / `public_history_v2` / `pool_prior_counter_continuation_v2` / `action_and_binary_belief_v2` / `empirical_all_answers_weighted_actions_v3`
- **全流程账本估算总费用**：\$461.95 / \$5,000.00（`pending_requests = 0`）

---

## 1. 训练与 Checkpoint 汇总

### 1.1 SFT（3 Epochs / 1,500 步）与 RS（1 Epoch / 125 步）共享验证集表现
训练集：SFT 使用 8,000 action + 7,995 belief；RS 从 SFT 的 1,837 局 on-policy 轨迹中采集 2,000 action + 2,000 belief（`shortfall = 0`），从 `sft-epoch-3` 状态续训 1 Epoch（`lr=5e-5`）。验证集固定为 1,000 action + 1,002 belief。

| 阶段 / 节点 | `val_action_set_nll` ↓ | `val_action_acc` ↑ | `val_norm_set_prob` ↑ | `val_regret` ↓ | `val_belief_loss` ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SFT Step 0（64 条诊断子集） | 8.2596 | 39.06% | — | — | 0.8619 |
| SFT Epoch 1（Step 500） | 0.9371 | 70.10% | 62.10% | 0.02918 | 0.6167 |
| SFT Epoch 2（Step 1000） | 0.5258 | 80.10% | 74.00% | 0.02276 | 0.6547 |
| **SFT Epoch 3（Step 1500，最佳）** | **0.3542** | **86.70%** | 78.58% | **0.00758** | 0.5964 |
| **RS Epoch 1（Step 125，最佳）** | 0.3694 | 86.40% | **82.46%** | 0.00790 | **0.5869** |

- **SFT 最佳 Checkpoint**：
  - Sampler：`tinker://720673ef-c429-5b70-a199-432286de1767:train:0/sampler_weights/sft-best-epoch-3`
  - State：`tinker://720673ef-c429-5b70-a199-432286de1767:train:0/weights/sft-epoch-3`
- **RS 最佳 Checkpoint**：
  - Sampler：`tinker://f5f6173d-ab47-552e-9a53-fbd19e3b8341:train:0/sampler_weights/rs-best-epoch-1`
  - State：`tinker://f5f6173d-ab47-552e-9a53-fbd19e3b8341:train:0/weights/rs-epoch-1`

---

## 2. 主对战胜率（座位轮换配对，每池 45 Blocks = 180 局）

| 模型 | `seen_pool` 胜率（95% CI） | `unseen_pool` 胜率（95% CI） | `seen` 诈唬率 / 质疑命中率 | `unseen` 诈唬率 / 质疑命中率 | 回退率 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `zero_shot` | 2.78% `[0.56%, 5.00%]` | 4.44% `[1.67%, 7.22%]` | 9.25% / 31.29% | 12.56% / 27.06% | 0.00% |
| `small_classifier` | 35.56% `[29.44%, 41.67%]` | 48.33% `[42.22%, 54.44%]` | 29.32% / 66.94% | 25.76% / 72.50% | 0.00% |
| `rule_counter` | 39.44% `[33.33%, 45.56%]` | 42.22% `[35.00%, 49.44%]` | 42.80% / 65.79% | 41.48% / 67.93% | 0.00% |
| **`lora_sft`** | **75.56%** `[70.00%, 80.56%]` | 66.11% `[58.89%, 73.33%]` | 28.71% / 40.26% | 30.84% / 36.13% | **0.00%** |
| **`lora_rs`** | 73.33% `[67.78%, 78.89%]` | **67.78%** `[61.11%, 74.44%]` | 28.88% / 39.75% | 31.53% / 38.31% | **0.00%** |

### 配对 Bootstrap 胜率差（10,000 次 Block 重采样）
- `seen`: `lora_sft - zero_shot` = **+72.78%** `[+66.67%, +78.33%]`
- `seen`: `lora_rs - lora_sft` = `-2.22%` `[-5.00%, 0.00%]`
- `seen`: `small_classifier - rule_counter` = `-3.89%` `[-10.56%, +2.78%]`
- `unseen`: `lora_sft - zero_shot` = **+61.67%** `[+53.89%, +69.44%]`
- `unseen`: `lora_rs - lora_sft` = **+1.67%** `[-1.67%, +5.56%]`
- `unseen`: `small_classifier - rule_counter` = `+6.11%` `[-2.78%, +14.44%]`

---

## 3. 历史消融（`lora_sft` on `seen_pool`，45 Blocks）

| 历史条件 | 胜率（95% CI） | 诈唬率 | 质疑命中率 |
| :--- | :---: | :---: | :---: |
| `full` | **75.56%** `[70.00%, 80.56%]` | 28.71% | 40.26% |
| `last_action_only` | 75.00% `[69.44%, 80.00%]` | 28.76% | 39.53% |
| `current_round_only` | 74.44% `[68.89%, 80.00%]` | 28.36% | 39.32% |

---

## 4. 冻结信念测试集校准（自然分布）

| 模型 | `seen`（n=1,003）LogLoss ↓ / Brier ↓ / ECE ↓ | `unseen`（n=1,000）LogLoss ↓ / Brier ↓ / ECE ↓ |
| :--- | :---: | :---: |
| `zero_shot` | 0.8166 / 0.3031 / 0.2524 | 0.8673 / 0.3283 / 0.3011 |
| `rule_counter` | 0.6389 / 0.2142 / 0.1215 | 0.6466 / 0.2250 / 0.1091 |
| `small_classifier` | 0.5674 / 0.1920 / **0.0537** | **0.5978** / **0.2049** / **0.0497** |
| `lora_sft` | 0.5750 / 0.1944 / 0.0925 | 0.6075 / 0.2079 / 0.0865 |
| **`lora_rs`** | **0.5588** / **0.1868** / 0.0549 | 0.6132 / 0.2077 / 0.0885 |

---

## 5. 独立残局诊断（148 条状态，含 60 条强制质疑 / 133 条含声称）

| 模型 | 原生合法率 | 回退率 | 强制质疑遵从 | 动作分布 (`CHALLENGE` / `PLAY`) | 残局信念 ECE ↓ / Brier ↓ / LogLoss ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `zero_shot` | 100.00% | 0.00% | 100.00% | 133 / 15 | 0.1636 / 0.2790 / 0.7636 |
| `small_classifier` | 100.00% | 0.00% | 100.00% | 133 / 15 | 0.3153 / 0.3379 / 0.9443 |
| `rule_counter` | 100.00% | 0.00% | 100.00% | 125 / 23 | 0.3016 / 0.3347 / 0.9489 |
| `lora_sft` | 100.00% | 0.00% | 100.00% | 87 / 61 | 0.2058 / 0.2903 / 0.7889 |
| `lora_rs` | 100.00% | 0.00% | 100.00% | 87 / 61 | 0.2831 / 0.3276 / 0.9007 |

---

## 6. 归档图表与原始汇总文件

- [`experiment_summary.json`](../results/selfplay_v2_20261113_01_quality_supervision_v3_endgame/experiment_summary.json)
- [`run_status.json`](../results/selfplay_v2_20261113_01_quality_supervision_v3_endgame/run_status.json)
- [`fig1_winrate_comparison.svg`](../results/selfplay_v2_20261113_01_quality_supervision_v3_endgame/figures/fig1_winrate_comparison.svg)
- [`fig2_history_ablation.svg`](../results/selfplay_v2_20261113_01_quality_supervision_v3_endgame/figures/fig2_history_ablation.svg)
- [`fig3_risk_hand_stratified.svg`](../results/selfplay_v2_20261113_01_quality_supervision_v3_endgame/figures/fig3_risk_hand_stratified.svg)
- [`fig4_calibration_and_profile.svg`](../results/selfplay_v2_20261113_01_quality_supervision_v3_endgame/figures/fig4_calibration_and_profile.svg)
