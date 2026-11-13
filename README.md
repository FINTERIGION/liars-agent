# 骗子酒馆 AI 决策智能体训练与评测平台 (`liars-agent`)

面向对手适应的语言模型训练与评测系统（支持四人对局规则引擎、严格观测隔离、5 类参数化策略对手池、小型分类器基线、Qwen3.5-9B LoRA 监督微调与多次续玩策略改进、座位轮换配对 Bootstrap 评测及 Gradio 决策复盘工作台）。

---

## 一、核心特性与架构

1. **确定性研究用规则引擎 (`src/liars_agent/engine/`)**：
   - 标准 20 张牌组（`Q`×6、`K`×6、`A`×6、`JOKER`×2），4 人各发 5 张牌，单次出牌 1~3 张。
   - 强制质疑：存在上一手出牌且除当前行动玩家外其余存活玩家均已空手时，当前玩家必须质疑上一手，不能继续出牌；两人残局上家打空手牌也属于该情形。例如 A 刚打空、C 已空手，仅 B 还有牌且轮到 B 时，B 必须质疑 A。
   - 空手退出：打光手牌者不再出牌或质疑；其刚出的牌仍可被下家质疑，若下家继续出牌，则该玩家被跳过，直到下一小局重新发牌。
   - **严格信息隔离 (`PlayerObservation`)**：模型仅接收自身手牌与公开历史，绝不接触其他玩家手牌或左轮实弹位置。
   - **100% 可复现回放 (`replay.py`)**：支持 JSONL 轨迹导出与 SHA-256 指纹回放一致性校验。
2. **多策略对手池与智能体 (`src/liars_agent/agents/`)**：
   - **训练可见对手池 (Seen Pool)**：`HonestConservativeBot`（诚实保守型）、`AggressiveBlufferBot`（激进诈唬型）、`CardCounterBot`（记牌精算型）。
   - **未见测试对手池 (Unseen Holdout Pool)**：`ParanoidHunterBot`（多疑猎手型）、`ChameleonAdaptiveBot`（变色龙切换型）。
   - **基线与主模型**：`SmallClassifierBot`（12 维统计特征小型分类器）、`LLMAgent`（支持 Qwen3.5-9B 零样本、LoRA-SFT、LoRA-RS 及 Tinker API 远端采样，内置非法动作回退与成本审计）。
3. **训练数据合成与按局切分 (`src/liars_agent/training/`)**：
   - 基于混合对手博弈与蒙特卡洛多次续玩（Rollout）收益筛选合成高质量 SFT 样本。
   - 严格按完整对局 (`game_id`) 划分训练/验证/测试集，防止同局步间泄漏。
4. **统一评测工作台与四类核心分析图 (`src/liars_agent/evaluation/`)**：
   - **座位轮换配对评测 (`Duplicate Block Protocol`)**：每个 Block 固定发牌与实弹种子，轮换 4 个座位，报告配对 Bootstrap 95% 置信区间。
   - 自动导出四类核心 SVG 矢量分析图（胜率及 CI 对照图、历史消融图、风险与手牌分层诊断热力图、信念校准曲线与四维行为画像图）。

---

## 二、快速开始

### 1. 运行系统正确性单元测试
```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### 2. 运行端到端数据生成、训练、配对评测与图表生成流水线
```bash
PYTHONPATH=src python3 scripts/run_full_pipeline.py --output-dir outputs --eval-blocks 45
```
运行完成后将在 `outputs/` 目录下生成：
- `outputs/dataset/`：按完整对局划分的 `train.jsonl`、`val.jsonl`、`test.jsonl`
- `outputs/models/`：LoRA 适配器配置、Loss 曲线、成本核算日志及小型分类器权重
- `outputs/figures/`：四类核心分析图（`.svg` 矢量图）
- `outputs/experiment_summary.json`：完整统计指标、配对 95% 置信区间与分层诊断结果

### 3. 启动交互式对局与决策复盘界面
```bash
PYTHONPATH=src python3 -m liars_agent.ui.app --port 7860
```

---

## 三、作者与开源协议 (Authors & License)

**项目作者 (Authors)**：
- **Xi Rao** — [@ReXeno56](https://github.com/ReXeno56)
- **Yuchao Feng** — [@FINTERIGION](https://github.com/FINTERIGION)
- **Boyan Zhou** — [@luckydrag0n](https://github.com/luckydrag0n)

本项目代码、训练配置及公开模型权重均采用 **[Apache License 2.0](LICENSE)** 开源协议。

> **声明**：本项目实现的规则引擎受桌面卡牌博弈机制启发，仅用于学术研究与人工智能决策算法评测，与任何商业游戏开发商无关联。
