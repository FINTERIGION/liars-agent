# 骗子酒馆 AI 决策智能体训练与评测平台

四人卡牌博弈研究平台：确定性规则引擎、五类策略对手、观测条件续玩数据、分类器基线、Qwen3.5-9B 的 Tinker LoRA-SFT / RS 训练、座位轮换配对评测和 Gradio 人机对局。

当前已实现真实模型调用路径，并完成本地功能验证。**真实 SFT/RS 训练和 Qwen 对局实验尚未运行**；本地验证输出是规则策略与小型分类器的结果。原仓库的模拟模型输出保存在 [历史示例](examples/README.md)，不作为实验结论。

完整准备好的数据已作为本仓库的 [selfplay-v3-20261114 Release](https://github.com/FINTERIGION/liars-agent/releases/tag/selfplay-v3-20261114) 附件提供：训练／验证／测试共 20,000 条，另有 148 条独立残局检查观测及真值、回放。压缩包约 50.5 MB；[下载、SHA-256 校验和使用说明](datasets/selfplay-v3-20261114/README.md)。Git clone 获取代码和校验清单，数据包需单独下载。

## 当前固定游戏规则（`liars_bar_research_v2`）

1. **胜负：** 四人开始，死亡者淘汰，最后一名存活玩家获胜。打光手牌只退出当前小局行动，不代表整场胜利。
2. **牌与发牌：** Q/K/A 各 6 张，JOKER 2 张，共 20 张。每小局重新洗牌，存活玩家各拿 5 张，多余牌不使用。随机选择 Q/K/A 中一种目标牌，可与上一小局相同，小局内保持不变。JOKER 是万能真牌，无额外惩罚效果。
3. **行动顺序：** 第一小局随机先手，此后按座位顺序循环，跳过死亡和空手玩家。只能选择当前合法出牌或质疑动作，没有摸牌或主动跳过。
4. **出牌：** 暗置打出自己手中的 1～3 张，声称全部为目标牌。张数如实公开，可以全真、全假或真假混出，不要求与上一手张数相同或递增。第一手只能出牌。
5. **质疑：** 只翻开最近一次出牌，不检查此前累计的牌。只要有一张既非目标牌又非 JOKER，就是诈唬，由出牌者开一枪；全部为真则质疑者开一枪。下一名玩家继续出牌后，不能回头质疑更早的一手。
6. **空手退出：** 打光手牌者不再出牌或质疑；刚出的最后一手仍可被下家质疑，若是假牌则仍可受罚。下家继续出牌后，该空手玩家一直被跳过，直到下一小局重新发牌。
7. **强制质疑：** 存在上一手且仅当前行动玩家还有牌时，当前玩家必须质疑，不能继续出牌。适用于两人、三人、四人仍存活的情况。例如 A 刚打空、其余人已空手、仅 B 有牌且轮到 B，则 B 必须质疑 A；若还有其他玩家有牌，B 可以选择继续出牌。
8. **左轮：** 每人 6 个弹膛、1 发实弹，开局随机确定实弹位置，整场不重置。每次受罚开一枪，击中实弹即淘汰；仍存活且已开枪 k 次者下一枪致死率为 1/(6-k)。受罚不等于必然死亡。
9. **换小局：** 每次质疑结算一次开枪后结束小局，无论是否有人死亡。若剩余多名玩家，重新发牌，由受罚者（若存活）或其下一名存活玩家先手；开枪次数跨小局累计。只剩一人时整场结束。不设置额外的 60 步结束或判负规则。
10. **信息：** 玩家只能看到自己的手牌/本小局出过的牌、目标牌、公开行动、各人手牌数量/存活状态/开枪次数，以及过往质疑亮牌与结算。其他人的底牌、未翻开出牌、未发出的牌和所有实弹位置（包括自己的）均不可见。完整历史模式提供整场全部公开行动及结算，三种历史消融均保留本小局自己的已知牌；全量真值复盘仅整场结束后开放，新局清空旧复盘，各浏览器会话独立。

[规则规格 YAML](configs/rules_default.yaml) 是固定规则的说明，**不是运行时配置入口**；修改它不会切换玩法。原先未接入的强制质疑开关和 60 步限制已移除。实际执行见 [规则引擎](src/liars_agent/engine/state.py)，模型使用同一套 [规则提示](src/liars_agent/agents/prompt_builder.py)。对手策略参数、训练评分和历史消融不改变游戏规则。

## 安装与本地验证

需要 Python 3.10+；当前验证环境为 Python 3.13、Tinker 0.39、Gradio 6。规则引擎和本地数据生成仅依赖 Python 标准库及 PyYAML。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[full]'
python -m unittest discover -s tests -v
```

若系统提示缺少 `ensurepip`，安装系统的 Python venv 包，或使用已经带有 venv/pip 的 Python 创建环境。

运行不需要模型凭证、不会调用收费 API 的端到端检查：

```bash
python scripts/run_full_pipeline.py --stage local --smoke \
  --output-dir outputs/runs/local_check
```

此命令生成 24 条动作样本和 24 条真实真假标签样本、按完整对局划分数据、训练分类器、对本地两种策略各运行两个对手池的座位轮换评测，在同一冻结观测集上比较信念预测，并导出统计与胜率图。小规模结果用于检查流程，不用于判断模型优劣。`local` 阶段拒绝选择 LLM 模型。

## 人机对局与复盘

```bash
python -m liars_agent.ui.app --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，默认由人类控制 Seat 0，其余座位使用规则策略。选择合法动作后点击提交，AI 自动推进至下一次人类回合。服务异常会保留已经执行的动作；使用“继续 AI 回合”恢复，不会重复提交人类动作。正常对局使用隐藏的随机环境种子，固定 seed 仅在显式调试模式使用。完整对局结束后可查看各步真值、动作说明、独立信念预测与回放校验；进行中仅显示玩家自己的手牌和公开信息。不同浏览器会话独立。

使用某次运行的分类器和真实模型产物：

```bash
python -m liars_agent.ui.app --port 7860 \
  --model-config outputs/runs/experiment/ui_models.yaml
```

只有配置提供了对应 checkpoint 的 SFT/RS 模型才会出现。零样本也需要有效 Tinker 登录；默认未配置模型的界面可完全离线使用。

## 正式实验与阶段恢复

默认配置见 [实验配置](configs/sft_qwen35_9b_lora.yaml)，对手阵容见 [对手池配置](configs/opponent_pools.yaml)。同一输出目录绑定配置和方法版本（规则、观测、教师、决策协议及提示哈希）。旧版本、缺失版本或不一致的训练样本会在远程 preflight 之前被拒绝；更改方法或参数时使用新目录。旧目录保留用于历史查阅，不能继续训练。

1. 生成数据与分类器，不调用 Tinker：

   使用已发布数据时，按[下载说明](datasets/selfplay-v3-20261114/README.md)解压并校验即可，跳过此生成步骤；后续始终使用数据包内的 `config.json` 与对应目录。

   ```bash
   python scripts/run_full_pipeline.py --stage data \
     --output-dir outputs/runs/experiment
   ```

2. 按实际 tokenizer 统计 SFT 训练和验证 token、估算费用，不创建模型 API 会话：

   ```bash
   python scripts/run_full_pipeline.py --stage estimate \
     --output-dir outputs/runs/experiment
   ```

   首次运行会下载公开 tokenizer，不下载模型权重。估算保存在 `cost_estimate.json`；RS 和对局推理的 token 数取决于后续轨迹。

3. 准备真实实验时，先登录并检查账号模型权限：

   ```bash
   tinker auth login
   python scripts/run_full_pipeline.py --stage preflight \
     --output-dir outputs/runs/experiment
   ```

   也可通过环境变量 `TINKER_API_KEY` 配置凭证。密钥不写入代码、配置快照或实验产物。

4. 先完成真实小试闭环，再执行正式训练和评测（以下命令会调用付费服务）：

   ```bash
   python scripts/run_full_pipeline.py --stage sft_trial --max-train-steps 5 \
     --output-dir outputs/runs/experiment
   python scripts/run_full_pipeline.py --stage all \
     --output-dir outputs/runs/experiment
   ```

   小试必须取得匹配当前数据、训练配置、LoRA 和模型的完整验收结果，正式 `sft` 才会启动；只有保存成功的旧小试不能放行。也可依次运行 `--stage sft`、`--stage rs`、`--stage evaluate`、`--stage figures`。完整数据批次、已完成模型评测和训练 epoch 会被复用；中断训练从最近完整 epoch 的模型及优化器状态恢复。未完成 epoch 重新执行，已发生的 token 费用仍计入共享账本。

`--eval-blocks`、`--target-samples`（动作）、`--belief-target-samples`（信念）、`--train-games`、`--seed` 可覆盖相应配置；`--models small_classifier,rule_counter` 可选择本地评测子集。使用 `--smoke` 只缩小数据与评测规模，不会把模拟策略改名成真实模型；要避免远端调用，请使用 `--stage local`。

已有数据可以单独进行本地质量复评，输出到新的运行目录：

```bash
python scripts/refine_dataset.py \
  --source-run outputs/runs/selfplay_v2_20261113_01 \
  --output-dir outputs/runs/selfplay_v2_20261113_01_quality --workers 8
python scripts/build_endgame_coverage.py \
  --source-run outputs/runs/selfplay_v2_20261113_01 \
  --output-dir outputs/runs/selfplay_v2_20261113_01_quality/endgame_coverage \
  --seed 2026111301 --samples-per-cell 10
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_dataset_quality.py \
  --run-dir outputs/runs/selfplay_v2_20261113_01_quality
python scripts/audit_refined_labels.py \
  --run-dir outputs/runs/selfplay_v2_20261113_01_quality --samples 60 --workers 4
```

复评先用独立的 8 次配对续玩筛查；最高分并列、第一与第二名差距不超过 0.05、旧标签不在新最高分集合或输入重复时增加到 32 次，仍并列或接近则增加到 64 次。扩量仅追加新轨迹，同一次轨迹内所有候选共用隐藏状态与随机条件。相同输入共用报告及标签；原有“最佳分数减候选平均分至少 0.05”的筛选不变，复评后不达标的样本从同一划分的源对局未使用观测中补换。信念样本、完整对局划分及原始数据保留，逐候选奖励和未决标记写入 `quality/`。这些分差用于分配计算量，不是统计置信度；64 次后仍接近的动作保留确定性选择并明确标记，不能当作已证唯一最优。

残局检查集采用独立游戏种子及覆盖策略，通过合法动作从初始状态走完整局，单独检查两／三／四人生还时的末张、上家打空和强制质疑；它不混入训练或自然分布主评测。两人生还时“上家打空且仍可自由出牌”按规则不可达。检查集的观测与真值分文件保存，每个可达格至少来自 5 个不同对局。

最后一个命令以固定种子随机抽取训练侧保留的原样本，用独立随机流、每候选 128 次续玩比较旧标签与复评标签。它只报告这组抽查的教师收益与最优集合命中情况，不包含补换样本、不使用验证或测试集，也不把结果回流调参或宣称真实胜率提升。

复评完成后，将并列答案集合和样本权重写入独立训练目录，再进行离线核验和费用估算：

```bash
python scripts/prepare_action_supervision.py \
  --source-run outputs/runs/selfplay_v2_20261113_01_quality \
  --output-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3
python scripts/augment_endgame_training.py \
  --source-run outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3 \
  --output-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame \
  --seed 2026111407 --samples-per-cell 20 --min-games-per-cell 10 --num-rollouts 64 --workers 8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_full_pipeline.py \
  --config outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame/config.json \
  --output-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame --stage estimate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_action_supervision.py \
  --run-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame
```

监督准备目录保留原样本并写入答案集合与源文件哈希；残局补充目录从新种子完整对局提取训练侧动作，替换等量普通训练样本，保留原有残局样本。训练动作仍为 8,000 条，三轮仍为 1,500 步；验证、测试、全部信念样本、独立检查集和分类器权重保持不变。补充对局只归入 train，且与原数据及检查集的游戏 ID、环境种子和所选动作输入隔离。16 个可达格各至少 20 条、至少来自 10 局；自由动作每候选至少 64 次共享随机条件续玩，沿用全并列答案与小分差权重，不再以优势门槛筛空稀有格；强制单动作直接按合法表监督，不捏造收益分差。

当前监督版本为 `empirical_all_answers_weighted_actions_v3`，动作回答截止首个 assistant EOS（含 EOS），排除其后模板换行；训练和验证共享这一边界。v1/v2 数据、checkpoint 和恢复进度不能直接进入新训练，旧目录保留为历史记录。其 `config.json` 和目录配套用于后续阶段。上述命令不创建 Tinker 服务会话，tokenizer 核验只使用本地缓存；不会运行训练。

`evaluate` 会评估运行目录中的独立残局检查集；也可用 `--stage endgame_evaluate --models small_classifier,rule_counter` 单独执行本地基线检查。LLM 使用相同冻结观测，报告原生输出合法率、强制质疑遵从率、回退率和信念校准，按场景／存活人数显示分母。自由动作没有最优标签，因此不报告最优动作准确率；这组人为覆盖分布也不代表自然对局胜率。已有预测绑定检查集哈希和模型参数，恢复时复用，错配拒绝。

早期验证固定抽取每任务最多 64 条验证样本，在第 0、50、100 步计算诊断，不参与 checkpoint 选择。`--stage estimate_trial --max-train-steps 5` 离线估算小试本体和全部闭环核验调用；`--stage sft_trial --max-train-steps 5` 执行真实小试。两者使用同一数据目录和配置，学习率计划仍按完整三轮计算。

小试另外固定抽取每任务最多 2 条 val 观测，以重复基线前向测数值抖动，检查更新后的分数变化足以区分原始权重；不要求 5 步降低 loss 或提升胜率。保存后新建 sampler，验证原生动作 JSON、两标签原始 log probability 与保存时训练端分数一致。再用相同下一 batch／学习率比较原训练客户端、完整状态／optimizer 恢复客户端及仅加载权重的负对照；前两者须匹配，负对照须可区分。三个诊断更新单独计费，不覆盖已保存的 5 步 checkpoint，也不改变正式 1,500 步。数值信号不足、非法输出、恢复不一致或服务失败均不会产生成功的小试元数据；不会自动加步、放宽容差或重试付费请求。

过程证据和失败阶段保存于 `models/sft_trial/trial_verification.json`。仅所有检查通过后写入 `trial_metadata.json`，它仍不是正式 adapter，RS／主评测不会自动使用小试权重。状态包括更新可辨别性、sampler 检查、训练状态重载和 optimizer 负对照，使用独立协议 `trial_training_sampling_closure_v2`；旧 `trial_complete` 标记不能替代这些证据。

对于本机已有且哈希仍匹配 v3 验收记录的数据，可复用已验证的 5 步主体 token 统计，只分词少量闭环探针和下一 batch 更新估算：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/refresh_trial_estimate.py \
  --run-dir outputs/runs/selfplay_v2_20261113_01_quality_supervision_v3_endgame
```

此命令不生成数据、不跑续玩、不运行模型、不登录 Tinker；源文件或估算口径不匹配时拒绝复用。新的费用总额读取 `trial_total_usd_upper_bound`；`sft_train_and_validation_usd` 仍表示小试主体。当前实现、轻量验证与费用见[小试闭环准备记录](docs/trial_closure_readiness.md)。

此前全答案监督的检查结果见 [v2 历史记录](docs/all_answer_supervision_validation.md)；当前训练准备结果见[训练前修正验收](docs/training_readiness_validation.md)。

## 方法与默认参数

- **规则：** 采用上文 `liars_bar_research_v2` 固定游戏规则。
- **信息隔离：** 模型只接收 `PlayerObservation`，不含运行 ID、环境种子或真实对手身份。环境种子与策略随机流留在执行和回放层。真值只用于引擎结算、真假监督标签与事后统计。
- **历史：** `full` 为整场全部公开行动与结算；`current_round_only` 只提供本小局公开历史；`last_action_only` 只提供最近公开动作。三者保留相同的当前公开状态与本小局私有记牌。训练全部使用 full，历史消融只发生在评测中被测玩家的输入上。
- **续玩教师：** 入口仅接受观测、固定教师配置和采样种子。隐藏牌与剩余实弹槽位按观测约束均匀采样；目标席位固定由 CardCounter 续玩，其他三席从 seen 三种策略各取一个随机排列，不读取真实局代理。每次续玩有独立策略随机种子，同次续玩的所有候选共享状态、排列和种子。
- **教师目标：** `0.65 × 整局胜利 + 0.35 × 当前小局未受罚`。每动作 8 次续玩，最佳值相对候选均值的优势门槛 0.05。这是固定规则续玩下的代理收益，不是当前 LLM 的精确策略价值或胜率保证。
- **数据：** 目标 10,000 条动作样本和 10,000 条独立信念样本。每批完整对局先按 80/10/10 约定划分，再提取两种任务；整个对局不跨集合，所有对局（含未通过动作筛选的对局）均有归属。短局不足时补充新批完整对局，直到达到目标或 max_games。分类器仅使用同一训练划分的信念样本。
- **动作任务：** 模型每次仍输出严格 JSON `{"chosen_action_id":"A0"}`。保存教师实测均值最高的全部动作（浮点容差 `1e-12`），不把分差在 0.05 内的所有动作都当成正确答案。每次访问一个动作状态时，全部可接受答案都参与监督。若该状态有 k 个答案、有效权重为 w，每个答案分配 w/k，再平分到其回答 token；状态总权重保持 w。答案顺序固定，恢复后目标和权重不变，覆盖不再依赖 epoch 数。样本生成、收费训练前检查与推理复用同一校验器；Markdown 外壳和额外字段也属于非法协议。
- **信念任务：** 事件为“上一手是否至少含一张假牌”。真实引擎标签监督单 token `0/1`，不只抽被质疑的牌，也不按动作收益筛选。推理从同一 checkpoint 的两个标签 log probability 归一化得到概率；无待回应声称时为 null，无信念调用。
- **SFT：** 一个 LoRA 同时训练上述两任务。rank 16、attention/MLP 开启、unembedding 关闭；学习率 `2e-4`，batch 32 每任务 16 条，3 epochs。只有唯一最高分且正分差不超过 0.05 的动作样本取原始权重 0.5；其他样本取 1。动作状态权重仅按训练动作集合的原始均值归一，再均分到全部可接受答案；信念标签权重始终为 1。batch 32 指 16 个动作状态和 16 条信念样本，按状态采样后才展开答案。每个逻辑 batch 只做一次优化器更新，步数及学习率计划不随答案数增加。0.5 是可配置的训练启发式，不代表置信概率。较短任务按固定种子循环，双任务期望总权重保持平衡。
- **训练验证：** 每个验证状态枚举所有合法 JSON 答案，用截至首个 assistant EOS（含 EOS、排除其后模板分隔符）的回答 token log probability 之和计算序列概率。checkpoint 按未降权、每状态等权的 `val_action_set_nll = -log(sum P(可接受答案))` 选择；单答案退化为完整序列 NLL。另报告合法候选归一化后的集合概率、候选 argmax 集合命中率、教师估计后悔值、原单答案平均 token NLL，以及`clear`／`tied`／`small_gap` 分组的指标和样本数，空组为 null。候选命中率不是自由生成动作的合法率或对局胜率；教师未评分所选动作时后悔值不可估计。信念标签 CE 单独记录；归一化二分类 log loss、Brier、ECE 另行评测。
- **RS：** 一轮，从 SFT 在训练侧新对局中的每状态 4 个候选筛选 2,000 条动作样本，排除回退候选，使用相同规则教师评分；另独立提取 2,000 条真实信念样本，不复制模型概率或理由为标签。从最佳 SFT 状态继续训练 1 epoch，学习率 `5e-5`；验证与测试保持隔离。
- **画像与说明：** 记录公开主动质疑频率、平均出牌张数、被揭示诈唬比例及分母；这些统计不等于真实策略身份，身份字段为 unknown。容量先验和人工启发式仅供规则基线使用，不充当概率真值。“动作说明”来自可核对的事实模板，不冒充模型的内部推理。
- **推理：** 零样本使用基础模型，SFT/RS 必须提供带当前方法合同的 checkpoint 元数据。非法动作 JSON 回退到记牌策略，独立模型信念保留，回退记录另存；认证、模型、checkpoint 或服务错误明确报告。全历史超出配置长度明确报错，不静默截断。
- **评测：** 五类策略分别在 seen 和含两类未见对手的混合池运行 45 blocks；每 block 抽取三对手排列并轮换四座。同 block 的各模型共用外部随机条件。按 block ID、种子、对手配置和排列校验配对，统一重采样 10,000 次计算 95% CI。25% 只作四人均分参考。
- **信念评测：** 同时保存各自对局分布上的校准，以及所有模型共享冻结测试观测的 log loss、Brier、ECE。seen 取信念 test 集，混合未见池另生成 1,000 条独立测试观测；信念差值按来源对局成组重采样。模型概率与回退概率分开，零分母返回 null、图表显示“无样本”。
- **消融：** SFT 三种历史条件使用同 checkpoint、同 block 随机排列和随机条件；完整历史结果直接复用主实验。

四类图表分别为胜率与 CI、历史消融、风险/手牌分层，以及信念校准与行为画像。只生成当前实际评测数据支持的图，不为未运行的模型填值。

并列只表示有限续玩中的实测均值相同，不证明动作真正等价。训练对每个状态的全部可接受答案计算平均 token NLL，checkpoint 集合 NLL 衡量可接受答案的总概率；两者不是同一损失。较大的答案集合更容易获得较低集合 NLL，因此比较使用固定验证集，并同时查看各组指标；不以分组难度差异宣称模型提升。

## 产物与成本

每次正式运行使用独立的 `outputs/runs/<run>/`：

| 路径 | 内容 |
| --- | --- |
| `config.json` / `run_status.json` | 实际配置、阶段完成状态、真实实验是否完成 |
| `dataset/` | 分批完整轨迹、共享 game_id 划分、双任务 train/val/test、各任务独立文件、RS 样本与冻结信念测试集 |
| `models/` | 分类器权重，真实训练状态/采样 checkpoint 元数据、损失曲线与恢复进度 |
| `quality/` | 续玩复评报告、新监督集合与权重汇总、离线 tokenizer/mask 核验报告 |
| `evaluation/` | 每条件逐 block/逐局结果、完整回放、冻结观测逐条预测、唯一权威 CI、回退率与 token 统计 |
| `experiment_summary.json` / `figures/` | 统一汇总及对应 SVG 图表 |
| `ui_models.yaml` | 此次运行可用于界面的模型和分类器配置 |

所有运行和其界面共用 `outputs/budget_usage.jsonl`，上限为本项目授权的 **5000 美元**。每次收费请求前按输入 token 和输出上限预留额度，收到响应后按 token 数结算估算；跨进程锁防止多个界面/任务重复占用额度。结果未知的失败请求保留预留。采样客户端显式关闭 SDK 外层自动重提交（`RetryConfig(enable_retry_logic=False)`），避免一次预算预留对应多个新请求；同请求的底层传输重试机制保留。信念打分的 SDK 内部每次至多额外生成 1 token，两个打分请求按完整前缀加最多 2 个生成 token 保守记账，不假定缓存折扣；`completion_tokens` 仅计实际返回的动作输出，隐藏生成另记 `belief_completion_tokens_upper_bound`，账本标注上界。估算包含全部可接受答案展开、按状态的双任务循环补齐所需的训练 token，以及每 epoch 全部合法验证候选、信念标签和第 0／50／100 步固定验证子集的 forward token；原单答案诊断复用候选结果，不重复调用。`cost_estimate.json` 另列独立残局集每个 LLM 的动作生成与信念打分费用上界，`trial_cost_estimate.json` 分列有限步数主体、闭环前向、三条分支的诊断更新和 sampler 推理上界，`trial_total_usd_upper_bound` 为完整小试报价。账本属于 token 费用估算，不是服务商账单；checkpoint 存储费用不包含在其中，正式运行时还需核对账号账单。费率快照来源为 [Tinker 官方价格](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)。

LoRA 参数使用 [Tinker SDK 实际支持的接口](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/)，checkpoint 默认持久保存。从 checkpoint 初始化训练时，在创建服务会话前核对来源 rank 和 attention／MLP／unembedding 开关；配置缺失或不一致会拒绝，不能以新配置覆盖实际继承的 LoRA 元数据。回放校验会逐步重建公开观测、隐藏手牌、真假标签、结算与终局；模型输出和时延只作为记录保存，不声称能由引擎重新生成。

没有凭证或真实 checkpoint 时，不会生成“训练完成”或虚构 Qwen 胜率。

## 作者与许可

Xi Rao — [@ReXeno56](https://github.com/ReXeno56)；Yuchao Feng — [@FINTERIGION](https://github.com/FINTERIGION)；Boyan Zhou — [@luckydrag0n](https://github.com/luckydrag0n)。

代码采用 [Apache License 2.0](LICENSE)。本项目用于学术研究与 AI 决策评测，与商业游戏开发商无关联。
