# 骗子酒馆 AI 决策智能体训练与评测平台

四人卡牌博弈研究平台：确定性规则引擎、五类策略对手、观测条件续玩数据、分类器基线、Qwen3.5-9B 的 Tinker LoRA-SFT / RS 训练、座位轮换配对评测和 Gradio 人机对局。

当前已实现真实模型调用路径，并完成本地功能验证。**真实 SFT/RS 训练和 Qwen 对局实验尚未运行**；本地验证输出是规则策略与小型分类器的结果。原仓库的模拟模型输出保存在 [历史示例](examples/README.md)，不作为实验结论。

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

4. 执行正式训练和评测：

   ```bash
   python scripts/run_full_pipeline.py --stage all \
     --output-dir outputs/runs/experiment
   ```

   或依次运行 `--stage sft`、`--stage rs`、`--stage evaluate`、`--stage figures`。完整数据批次、已完成模型评测和训练 epoch 会被复用；中断训练从最近完整 epoch 的模型及优化器状态恢复。未完成 epoch 重新执行，已发生的 token 费用仍计入共享账本。

`--eval-blocks`、`--target-samples`（动作）、`--belief-target-samples`（信念）、`--train-games`、`--seed` 可覆盖相应配置；`--models small_classifier,rule_counter` 可选择本地评测子集。使用 `--smoke` 只缩小数据与评测规模，不会把模拟策略改名成真实模型；要避免远端调用，请使用 `--stage local`。

## 方法与默认参数

- **规则：** 采用上文 `liars_bar_research_v2` 固定游戏规则。
- **信息隔离：** 模型只接收 `PlayerObservation`，不含运行 ID、环境种子或真实对手身份。环境种子与策略随机流留在执行和回放层。真值只用于引擎结算、真假监督标签与事后统计。
- **历史：** `full` 为整场全部公开行动与结算；`current_round_only` 只提供本小局公开历史；`last_action_only` 只提供最近公开动作。三者保留相同的当前公开状态与本小局私有记牌。训练全部使用 full，历史消融只发生在评测中被测玩家的输入上。
- **续玩教师：** 入口仅接受观测、固定教师配置和采样种子。隐藏牌与剩余实弹槽位按观测约束均匀采样；目标席位固定由 CardCounter 续玩，其他三席从 seen 三种策略各取一个随机排列，不读取真实局代理。每次续玩有独立策略随机种子，同次续玩的所有候选共享状态、排列和种子。
- **教师目标：** `0.65 × 整局胜利 + 0.35 × 当前小局未受罚`。每动作 8 次续玩，最佳值相对候选均值的优势门槛 0.05。这是固定规则续玩下的代理收益，不是当前 LLM 的精确策略价值或胜率保证。
- **数据：** 目标 10,000 条动作样本和 10,000 条独立信念样本。每批完整对局先按 80/10/10 约定划分，再提取两种任务；整个对局不跨集合，所有对局（含未通过动作筛选的对局）均有归属。短局不足时补充新批完整对局，直到达到目标或 max_games。分类器仅使用同一训练划分的信念样本。
- **动作任务：** 模型输出严格 JSON `{"chosen_action_id":"A0"}`，只学习合法动作选择。样本生成、收费训练前检查与推理复用同一校验器；Markdown 外壳和额外字段也属于非法协议。
- **信念任务：** 事件为“上一手是否至少含一张假牌”。真实引擎标签监督单 token `0/1`，不只抽被质疑的牌，也不按动作收益筛选。推理从同一 checkpoint 的两个标签 log probability 归一化得到概率；无待回应声称时为 null，无信念调用。
- **SFT：** 一个 LoRA 同时训练上述两任务。rank 16、attention/MLP 开启、unembedding 关闭；学习率 `2e-4`，batch 32 每任务 16 条，3 epochs。动作回答 token 权重总和为 1，信念只监督标签位置；较短任务按固定种子循环。按验证集动作 NLL 选 checkpoint，分别报告动作 NLL、信念标签 CE。归一化二分类 log loss、Brier、ECE 单独评测，不与标签 CE 混称。
- **RS：** 一轮，从 SFT 在训练侧新对局中的每状态 4 个候选筛选 2,000 条动作样本，排除回退候选，使用相同规则教师评分；另独立提取 2,000 条真实信念样本，不复制模型概率或理由为标签。从最佳 SFT 状态继续训练 1 epoch，学习率 `5e-5`；验证与测试保持隔离。
- **画像与说明：** 记录公开主动质疑频率、平均出牌张数、被揭示诈唬比例及分母；这些统计不等于真实策略身份，身份字段为 unknown。容量先验和人工启发式仅供规则基线使用，不充当概率真值。“动作说明”来自可核对的事实模板，不冒充模型的内部推理。
- **推理：** 零样本使用基础模型，SFT/RS 必须提供带当前方法合同的 checkpoint 元数据。非法动作 JSON 回退到记牌策略，独立模型信念保留，回退记录另存；认证、模型、checkpoint 或服务错误明确报告。全历史超出配置长度明确报错，不静默截断。
- **评测：** 五类策略分别在 seen 和含两类未见对手的混合池运行 45 blocks；每 block 抽取三对手排列并轮换四座。同 block 的各模型共用外部随机条件。按 block ID、种子、对手配置和排列校验配对，统一重采样 10,000 次计算 95% CI。25% 只作四人均分参考。
- **信念评测：** 同时保存各自对局分布上的校准，以及所有模型共享冻结测试观测的 log loss、Brier、ECE。seen 取信念 test 集，混合未见池另生成 1,000 条独立测试观测；信念差值按来源对局成组重采样。模型概率与回退概率分开，零分母返回 null、图表显示“无样本”。
- **消融：** SFT 三种历史条件使用同 checkpoint、同 block 随机排列和随机条件；完整历史结果直接复用主实验。

四类图表分别为胜率与 CI、历史消融、风险/手牌分层，以及信念校准与行为画像。只生成当前实际评测数据支持的图，不为未运行的模型填值。

## 产物与成本

每次正式运行使用独立的 `outputs/runs/<run>/`：

| 路径 | 内容 |
| --- | --- |
| `config.json` / `run_status.json` | 实际配置、阶段完成状态、真实实验是否完成 |
| `dataset/` | 分批完整轨迹、共享 game_id 划分、双任务 train/val/test、各任务独立文件、RS 样本与冻结信念测试集 |
| `models/` | 分类器权重，真实训练状态/采样 checkpoint 元数据、损失曲线与恢复进度 |
| `evaluation/` | 每条件逐 block/逐局结果、完整回放、冻结观测逐条预测、唯一权威 CI、回退率与 token 统计 |
| `experiment_summary.json` / `figures/` | 统一汇总及对应 SVG 图表 |
| `ui_models.yaml` | 此次运行可用于界面的模型和分类器配置 |

所有运行和其界面共用 `outputs/budget_usage.jsonl`，上限为本项目授权的 **5000 美元**。每次收费请求前按输入 token 和输出上限预留额度，收到响应后按 token 数结算估算；跨进程锁防止多个界面/任务重复占用额度。结果未知的失败请求保留预留，不自动重复计作免费调用。信念打分的 SDK 内部每次至多额外生成 1 token，两个打分请求按完整前缀加最多 2 个生成 token 保守记账，不假定缓存折扣；`completion_tokens` 仅计实际返回的动作输出，隐藏生成另记 `belief_completion_tokens_upper_bound`，账本标注上界。估算包含双任务循环补齐的训练 token。账本属于 token 费用估算，不是服务商账单；checkpoint 存储费用不包含在其中，正式运行时还需核对账号账单。费率快照来源为 [Tinker 官方价格](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)。

LoRA 参数使用 [Tinker SDK 实际支持的接口](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/)，checkpoint 默认持久保存。回放校验会逐步重建公开观测、隐藏手牌、真假标签、结算与终局；模型输出和时延只作为记录保存，不声称能由引擎重新生成。

没有凭证或真实 checkpoint 时，不会生成“训练完成”或虚构 Qwen 胜率。

## 作者与许可

Xi Rao — [@ReXeno56](https://github.com/ReXeno56)；Yuchao Feng — [@FINTERIGION](https://github.com/FINTERIGION)；Boyan Zhou — [@luckydrag0n](https://github.com/luckydrag0n)。

代码采用 [Apache License 2.0](LICENSE)。本项目用于学术研究与 AI 决策评测，与商业游戏开发商无关联。
