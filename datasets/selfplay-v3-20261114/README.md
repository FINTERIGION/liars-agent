# 完整数据集 selfplay-v3-20261114

数据发布在本仓库的 [GitHub Release](https://github.com/FINTERIGION/liars-agent/releases/tag/selfplay-v3-20261114)。这里提交下载说明和固定校验清单，完整文件存放在 Release 附件中；不是 Git LFS 指针，也不包含付费模型 checkpoint。

[下载完整压缩包](https://github.com/FINTERIGION/liars-agent/releases/download/selfplay-v3-20261114/liars-agent-selfplay-v3-20261114.tar.gz)：**50,536,196 字节**（约 50.5 MB）。解压包含原准备目录全部 182 个文件，原始文件合计 **1,070,132,221 字节**，另加一个发布标记。较大的解压体积来自按任务拆分和合并副本，以及用于核验的回放；不是额外独立样本。

| 集合 | 动作样本 | 信念样本 | 合计 |
| --- | ---: | ---: | ---: |
| 训练集 `dataset/train.jsonl` | 8,000 | 7,995 | 15,995 |
| 验证集 `dataset/val.jsonl` | 1,000 | 1,002 | 2,002 |
| 测试集 `dataset/test.jsonl` | 1,000 | 1,003 | 2,003 |
| 主数据总计 | 10,000 | 10,000 | 20,000 |

独立残局检查集有 **148 条观测、148 条标签和 21 局完整回放**：133 条有待回应声称，15 条无声称且标签为 `null`。它不加入训练集，也不作为自然对局胜率数据。其对局 ID 和环境种子与主数据来源隔离；不同对局仍可能产生相同公开输入，本次核对有 1 个公开动作输入相同。

## 下载与完整性校验

使用本次发布对应的代码。已有仓库需包含 `selfplay-v3-20261114` 标签的代码；也可以新建独立工作副本：

```bash
git clone --branch selfplay-v3-20261114 https://github.com/FINTERIGION/liars-agent.git liars-agent-data
cd liars-agent-data
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

在仓库根目录使用 GitHub CLI 下载四个附件：

```bash
mkdir -p outputs/downloads/selfplay-v3-20261114
gh release download selfplay-v3-20261114 --repo FINTERIGION/liars-agent \
  --dir outputs/downloads/selfplay-v3-20261114
```

也可从 Release 页面手动下载 `liars-agent-selfplay-v3-20261114.tar.gz`、`manifest.json`、`FILES.sha256` 和 `SHA256SUMS` 至上述目录。页面自动生成的 “Source code” 压缩包只有源码，不含这些数据文件。

先用仓库中的固定清单校验压缩包和附件，再解压到一个尚不存在的目录：

```bash
(cd outputs/downloads/selfplay-v3-20261114 && \
  sha256sum -c ../../../datasets/selfplay-v3-20261114/SHA256SUMS)

mkdir -p outputs/runs
test ! -e outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame && \
  tar -xzf outputs/downloads/selfplay-v3-20261114/liars-agent-selfplay-v3-20261114.tar.gz \
    -C outputs/runs

(cd outputs/runs && sha256sum -c ../../datasets/selfplay-v3-20261114/FILES.sha256)

python -m liars_agent.training.dataset_release \
  --run-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame
```

不要覆盖本机已有的同名运行目录；可换一个解压父目录，`--run-dir` 跟随实际位置。文件校验、数据读取均不需要 Tinker 凭证，不会调用模型或重新生成数据。

压缩包 SHA-256：

```text
b0bdd2aad76e46ee1712cc8eeedd5ab7c92c6e93275cdc50d35a2dd3005862f1
```

[SHA256SUMS](SHA256SUMS) 校验附件，[FILES.sha256](FILES.sha256) 校验解压后的每个文件，[manifest.json](manifest.json) 固定文件大小、哈希、方法版本及来源对局身份摘要。

## 数据结构和使用

解压后的根目录为 `selfplay_v2_20261113_01_quality_supervision_v3_endgame/`：

- `dataset/train.jsonl`、`val.jsonl`、`test.jsonl` 是训练流程读取的主文件；`game_splits.json` 是共享的整局划分。`action_*.jsonl`、`belief_*.jsonl` 和 `sft_all_samples.jsonl` 是同一数据的拆分／合并视图，不应再次拼接进训练集。
- 动作任务学习 `action_supervision.target_action_ids` 中的全部可接受答案，按 `w/k` 分摊状态权重；不要只取旧的规范 `completion` 而丢弃并列答案。使用仓库现有训练器可保持这一目标。
- 信念任务的 `label=1` 表示上一手至少有一张假牌，`label=0` 表示全部为真。只把任务对应的 `system` 和 `prompt` 送给模型；游戏编号、种子、审计字段和隐藏真值不属于模型输入。
- `endgame_coverage/observations.jsonl` 存有声称观测，`no_pending_claim.jsonl` 存无声称观测；`labels.jsonl` 通过 `sample_id` 对齐两类观测。`private/` 保存来源位置、生成记录和完整真值回放，仅用于核验或事后复盘，不能加入模型提示。
- `augmentation/` 保存训练侧残局补充、替换清单、教师候选奖励与回放；`quality/` 保存既有监督验收。`models/small_classifier_weights.json` 是本地分类器，不是 Qwen 权重。
- `config.json`、费用估算和原有本地基线评测记录一并保留。完整训练仍需先通过真实 `sft_trial`；此数据发布不代表付费训练已经完成。

原始文件逐字节保持不变，历史绝对路径保留为来源记录。新增 `dataset_release.json` 仅标识已发布版本；运行时校验器从**源码仓库固定清单**读取期望哈希，核对不可变输入后复用既有来源验收，不要求另一台机器拥有本机的历史目录。残局仍验证包内回放、标签与来源身份隔离。没有发布标记的普通运行目录继续使用原来的来源验证流程。

训练和评测会新增输出并更新费用／状态记录，因此训练后应使用程序的运行校验；上面的逐文件全量命令用于验证刚下载的原始发布包。修改任何固定训练数据、配置或标签都会被运行校验拒绝，需要重新准备并验收新数据。

以后获准真实训练时，安装 `.[full]` 依赖并使用包内配置，例如以下小试命令；**本次发布未执行它**：

```bash
python scripts/run_full_pipeline.py \
  --config outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame/config.json \
  --output-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame \
  --stage sft_trial --max-train-steps 5
```

真实小试通过后再执行 `--stage sft`。评测使用相同配置和目录的 `--stage evaluate`，会读取冻结的独立残局集。费用、真实调用门槛与小试限制见[小试闭环准备记录](../../docs/trial_closure_readiness.md)；原数据验收见[v3 记录](../../docs/training_readiness_validation.md)。
