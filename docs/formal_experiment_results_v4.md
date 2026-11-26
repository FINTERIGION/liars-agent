# v4 正式实验结果汇总（Qwen3.5-9B `adaptive_v4` LoRA-SFT + Tinker 在线 GRPO/PPO RL）

- **运行目录**：`adaptive_v4_20261124`
- **归档产物**：[`results/adaptive_v4_20261124/`](../results/adaptive_v4_20261124/)
- **完成时间**：2026-11-26T06:27:51Z（`sft_trial` / `sft` / `rl` / `evaluate` / `figures` 全部完成）
- **基座与平台**：`Qwen/Qwen3.5-9B`，Tinker LoRA（`r=16`, `train_attn=true`, `train_mlp=true`, `train_unembed=false`）
- **方法合同**：`liars_bar_research_v2` / `public_history_count_profile_v4` / `calibrated_posterior_continuation_v4` / `action_and_binary_belief_v2` / `empirical_all_answers_weighted_actions_v3`
- **本轮 v4 全流程账本估算费用**：\$455.63（含 5 步小试闭环 + 3-Epoch SFT 约 \$400、30 步在线 RL 约 **\$26.5**、5 模型全量评测约 \$29；共享账本累计 \$917.58 / \$5,000.00，`pending_requests = 0`）
- **历史对照**：v3 正式实验归档见 [`formal_experiment_results.md`](formal_experiment_results.md)

---

## 1. 零 API 本地验证（修复前 vs 修复后）

在投入收费训练前，通过 `scripts/validate_v4_adaptation.py` 完成零 API 验证（归档见 [`v4_local_validation.json`](../results/adaptive_v4_20261124/validation/v4_local_validation.json)）：

| 验证项 | 关键指标 | 初始 v4 | **修复后 v4（`ec8428c`）** |
| :--- | :--- | :---: | :---: |
| **[1/4] 跨局信念校准** | `unseen_mirror_tell` LogLoss ↓ / AUC ↑ | 0.6239 / 0.7032 | **0.5761 / 0.7385**（v3: 0.6682 / 0.5720） |
| **[1/4] 跨局信念校准** | `heldout_diverse_12` LogLoss ↓ / AUC ↑ | 0.5827 / 0.7263 | **0.5384 / 0.7231**（v3: 0.5632 / 0.6799） |
| **[2/4] 反事实历史敏感性** | `v4_posterior_teacher` 最优动作翻转率 | 16.7% (2/12) | **75.0% (9/12)**（`v2_teacher`: 16.7%，`dQ=+0.2810`） |
| **[3/4] 180 局历史消融** | `unseen_mirror_tell_pool` (`full - no_hist`) | -1.67% | **+2.77%**（75.6% → 78.3%） |
| **[3/4] 180 局历史消融** | `diverse_13_pool` (`full - no_hist`) | +0.00% | **+5.00%**（71.7% → 76.7%） |
| **[4/4] 直连教师策略 (48 局)** | `v2` → `v4_no_hist` → **`v4_full_hist`** | 77.1% / 72.9% / 72.9% | **60.4% → 70.8% → 83.3% (40/48)**（历史增益 **+12.5%**） |

---

## 2. 训练与 Checkpoint 汇总

### 2.1 数据集与 3-Epoch v4 LoRA-SFT（1,575 步）
数据集由 1,020 局对称多样化自博弈对局 + 400 条反事实历史对比样本构成（严格按 `game_id` 划分，`shortfall = 0`，最大输入长度 4,034 tokens）：
- `train`：8,397 action（含 400 反事实样本）+ 7,995 belief
- `val`：1,002 action + 1,003 belief
- `test`：1,001 action + 1,002 belief

| 阶段 / 节点 | `val_action_set_nll` ↓ | `val_action_acc` ↑ | `val_norm_set_prob` ↑ | `val_regret` ↓ | `val_belief_loss` ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SFT Step 0（64 条诊断子集） | 7.9615 | 15.62% | 27.29% | 0.22041 | 0.9192 |
| SFT Step 50（64 条诊断子集） | 0.7399 | 62.50% | 51.73% | 0.02969 | 0.5603 |
| SFT Step 100（64 条诊断子集） | 0.7182 | 73.44% | 51.96% | 0.02769 | 0.6087 |
| SFT Epoch 1（Step 525，全量 val） | 0.5790 | 68.66% | 61.56% | 0.03273 | 0.6050 |
| SFT Epoch 2（Step 1050，全量 val） | 0.5118 | **79.14%** | 66.71% | **0.01114** | 0.5886 |
| **SFT Epoch 3（Step 1575，最佳）** | **0.4973** | 77.15% | **67.84%** | 0.01188 | **0.5855** |

### 2.2 30 步 Tinker 在线 Paired GRPO/PPO RL（含信念锚定）
从 `sft-epoch-3` 状态热启动，使用 Tinker 原生 `ppo` 损失执行 30 步在线 Paired GRPO（每步 4 组 × 4 局配对 rollout = 16 局，对手为 13 类 v4 原型 + 25% `PosteriorAdaptiveBot`，奖励为 `0.70 × 整局获胜 + 0.30 × 当小局存活 - 1.0 × 非法回退`），每步同步累积 8 条真实引擎标签的信念交叉熵锚定梯度（`weight = 0.25`）：
- **在线训练指标**：总耗时 16,480.6s，30 步平均在线胜率 **67.08%**（后 5 步 66.25%），增量费用仅 **约 \$26.5**。
- **最佳 Checkpoint**：
  - **v4 SFT Sampler**：`tinker://07a161e2-828c-5b76-8e6f-4a658e024f7f:train:0/sampler_weights/sft-best-epoch-3`
  - **v4 SFT State**：`tinker://07a161e2-828c-5b76-8e6f-4a658e024f7f:train:0/weights/sft-epoch-3`
  - **v4 RL Sampler**：`tinker://46d1ff68-6083-5832-be54-14edf55179f0:train:0/sampler_weights/rl-final`
  - **v4 RL State**：`tinker://46d1ff68-6083-5832-be54-14edf55179f0:train:0/weights/rl-final`

---

## 3. 主对战胜率与行为画像（座位轮换配对，每池 45 Blocks = 180 局）

| 模型 | `seen_pool` 胜率（95% CI） | `unseen_pool` 胜率（95% CI） | `seen` 质疑率 / 命中率 / 诈唬率 / 被抓率 | `unseen` 质疑率 / 命中率 / 诈唬率 / 被抓率 | 回退率 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `zero_shot` | 4.44% `[1.67%, 7.22%]` | 13.33% `[8.33%, 18.33%]` | 92.92% / 25.80% / 6.67% / 19.51% | 92.71% / 33.96% / 8.07% / 37.50% | 0.00% |
| `rule_counter` | 13.33% `[8.89%, 18.33%]` | 17.78% `[11.67%, 23.89%]` | 29.18% / 14.29% / 34.28% / 25.65% | 25.65% / 23.09% / 35.47% / 27.16% | 0.00% |
| `small_classifier` | 43.33% `[36.67%, 49.44%]` | 44.44% `[37.78%, 51.11%]` | 10.68% / 29.51% / 18.73% / 53.22% | 11.79% / 40.78% / 21.09% / 37.59% | 0.00% |
| `lora_sft` | 87.22% `[82.78%, 91.67%]` | 74.44% `[68.33%, 80.56%]` | 1.87% / 20.00% / 33.79% / 20.26% | 8.43% / 41.13% / 30.11% / 24.95% | **0.00%** |
| **`lora_rl`** | **92.22%** `[88.89%, 95.56%]` | **76.67%** `[70.56%, 82.78%]` | 3.55% / **34.48%** / **27.53%** / **10.41%** | 9.56% / 40.97% / **24.92%** / **21.73%** | **0.00%** |

### 配对 Bootstrap 胜率差（10,000 次 Block 重采样）
- **`seen`: `lora_rl - lora_sft` = `+5.00%` `[+1.11%, +9.44%]`**（统计显著，对比 v3 RS 的 `-2.22% [-5.00%, 0.00%]`）
- `seen`: `lora_rl - zero_shot` = **+87.78%** `[+82.78%, +92.22%]`
- `seen`: `lora_sft - zero_shot` = **+82.78%** `[+76.11%, +88.89%]`
- `seen`: `small_classifier - rule_counter` = +30.00% `[+22.78%, +37.22%]`
- **`unseen`: `lora_rl - lora_sft` = `+2.22%` `[-1.11%, +5.56%]`**
- `unseen`: `lora_rl - zero_shot` = **+63.33%** `[+56.67%, +70.00%]`
- `unseen`: `lora_sft - zero_shot` = **+61.11%** `[+53.89%, +68.33%]`
- `unseen`: `small_classifier - rule_counter` = +26.67% `[+19.44%, +33.89%]`

---

## 4. 冻结信念测试集校准（自然分布）

| 模型 | `seen`（n=1,002）LogLoss ↓ / Brier ↓ / ECE ↓ | `unseen`（n=1,000）LogLoss ↓ / Brier ↓ / ECE ↓ |
| :--- | :---: | :---: |
| `zero_shot` | 0.8560 / 0.3186 / 0.3115 | 0.8520 / 0.3191 / 0.2468 |
| `rule_counter` | 0.8164 / 0.2860 / 0.2538 | 0.9219 / 0.3337 / 0.2567 |
| `small_classifier` | 0.5880 / 0.1996 / 0.0933 | **0.7094** / **0.2560** / **0.1098** |
| `lora_sft` | **0.5790** / **0.1954** / 0.0254 | 0.7313 / 0.2645 / 0.1782 |
| **`lora_rl`** | **0.5793** / **0.1955** / **0.0242** | 0.7313 / 0.2645 / 0.1764 |

- RL 信念锚定效果：`lora_rl - lora_sft` 在 `seen` 上 `LL_diff = +0.0002 [-0.0019, +0.0024]`（ECE 从 0.0254 微降至 **0.0242**），在 `unseen` 上 `LL_diff = +0.0000 [-0.0024, +0.0023]`，实现了**零校准退化**的在线策略强化。

---

## 5. 历史消融（`lora_sft` on `seen_pool`，45 Blocks）

| 历史条件 | 胜率（95% CI） | 诈唬率 | 质疑命中率 |
| :--- | :---: | :---: | :---: |
| `full` | 87.22% `[82.78%, 91.67%]` | 33.79% | 20.00% |
| `current_round_only` | 90.56% `[86.11%, 94.44%]` | 31.18% | 18.60% |
| `last_action_only` | 90.56% `[86.67%, 94.44%]` | 31.04% | 25.00% |

说明：在 `seen_pool` 默认阵容（含 `paranoid_sheriff`）上，`lora_sft` 主动质疑率仅 1.87%，单靠稳健出牌即逼近 90% 胜率，较长的全历史上下文（最多 4,034 tokens）未进一步提高 `seen` 胜率；跨局历史在更高质疑频次（8.4%~9.6%）的 `unseen_pool` 及 `lora_rl` 对照评测中值得进一步单独消融验证。

---

## 6. 归档图表与原始汇总文件

- [`experiment_summary.json`](../results/adaptive_v4_20261124/experiment_summary.json)
- [`run_status.json`](../results/adaptive_v4_20261124/run_status.json)
- [`ui_models.yaml`](../results/adaptive_v4_20261124/ui_models.yaml)
- [`v4_local_validation.json`](../results/adaptive_v4_20261124/validation/v4_local_validation.json)
- [`fig1_winrate_comparison.svg`](../results/adaptive_v4_20261124/figures/fig1_winrate_comparison.svg)
- [`fig2_history_ablation.svg`](../results/adaptive_v4_20261124/figures/fig2_history_ablation.svg)
- [`fig3_risk_hand_stratified.svg`](../results/adaptive_v4_20261124/figures/fig3_risk_hand_stratified.svg)
- [`fig4_calibration_and_profile.svg`](../results/adaptive_v4_20261124/figures/fig4_calibration_and_profile.svg)
