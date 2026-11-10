"""端到端实验与评测流水线：数据合成 -> LoRA/分类器训练 -> 座位轮换配对评测 -> 泛化与历史消融 -> 四类核心分析图导出."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from liars_agent.agents.classifier_bot import SmallClassifierBot
from liars_agent.agents.llm_agent import LLMAgent
from liars_agent.agents.rule_bots import (
    AggressiveBlufferBot,
    CardCounterBot,
    ChameleonAdaptiveBot,
    HonestConservativeBot,
    ParanoidHunterBot,
)
from liars_agent.engine.replay import save_replays_jsonl
from liars_agent.evaluation.duplicate_runner import run_duplicate_evaluation
from liars_agent.evaluation.metrics import (
    compute_behavioral_profile,
    compute_belief_calibration,
    compute_paired_bootstrap_ci,
    compute_risk_and_hand_stratified_diagnosis,
)
from liars_agent.evaluation.plots import (
    generate_fig1_winrate_svg,
    generate_fig2_history_ablation_svg,
    generate_fig3_risk_stratified_svg,
    generate_fig4_calibration_and_profile_svg,
)
from liars_agent.training.data_generator import generate_mixed_opponent_sft_dataset
from liars_agent.training.tinker_train import run_tinker_lora_sft


def run_pipeline(
    output_dir: Path,
    num_train_games: int = 60,
    target_sft_samples: int = 400,
    eval_blocks: int = 45,
    seed: int = 2026,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = output_dir / "dataset"
    models_dir = output_dir / "models"
    replays_dir = output_dir / "replays"
    figures_dir = output_dir / "figures"

    print(f"[1/5] 生成混合对手博弈轨迹与多次续玩(Rollout)筛选 SFT 数据...")
    sft_samples, train_replays = generate_mixed_opponent_sft_dataset(
        num_games=num_train_games,
        target_samples=target_sft_samples,
        num_rollouts_per_action=3,
        min_advantage_margin=0.04,
        base_seed=seed,
        output_jsonl_path=datasets_dir / "sft_all_samples.jsonl",
    )
    save_replays_jsonl(train_replays, replays_dir / "train_pool_replays.jsonl")

    print(f"[2/5] 执行按完整对局划分的 Qwen3.5-9B LoRA 监督微调与小型分类器拟合...")
    adapter_meta = run_tinker_lora_sft(
        samples=sft_samples,
        output_dir=models_dir / "qwen35_9b_lora_sft_v1",
        seed=seed,
    )
    clf_bot = SmallClassifierBot(name="small_classifier")
    clf_fit_stats = clf_bot.fit_from_replays([b.to_dict() for b in train_replays])
    clf_weights_path = models_dir / "small_classifier_weights.json"
    clf_bot.save_weights(clf_weights_path)

    # 定义固定训练对手池 (Seen Pool) 与未见测试对手池 (Unseen Pool)
    seen_opponents = [
        lambda s: HonestConservativeBot(seed=s),
        lambda s: AggressiveBlufferBot(seed=s),
        lambda s: CardCounterBot(seed=s),
    ]
    unseen_opponents = [
        lambda s: ParanoidHunterBot(seed=s),
        lambda s: ChameleonAdaptiveBot(seed=s),
        lambda s: CardCounterBot(seed=s),
    ]

    # 定义 5 类参评模型工厂
    model_specs = [
        ("zero_shot", "Zero-Shot (9B)", lambda s: LLMAgent(name="zero_shot", mode="zero_shot", seed=s)),
        ("small_classifier", "Small Classifier", lambda s: SmallClassifierBot(name="small_classifier", bluff_weights=clf_bot.bluff_weights)),
        ("rule_counter", "Rule CardCounter", lambda s: CardCounterBot(name="rule_counter", seed=s)),
        ("lora_sft", "LoRA-SFT (9B)", lambda s: LLMAgent(name="lora_sft", mode="lora_sft", adapter_path=models_dir / "qwen35_9b_lora_sft_v1" / "adapter_metadata.json", seed=s)),
        ("lora_rs", "LoRA-RS (Rollout+)", lambda s: LLMAgent(name="lora_rs", mode="lora_rs_improved", seed=s)),
    ]

    print(f"[3/5] 开展座位轮换配对评测 (Seen Pool & Unseen Pool, 各 {eval_blocks} Blocks = {eval_blocks * 4} 局/模型)...")
    seen_eval_map = {}
    unseen_eval_map = {}
    seen_plot_items = []
    unseen_plot_items = []

    for key, label, factory in model_specs:
        res_seen = run_duplicate_evaluation(
            candidate_factory=factory,
            opponent_factories=seen_opponents,
            num_blocks=eval_blocks,
            base_seed=seed + 1000,
            pool_name="seen_pool",
            ablation_mode="full",
        )
        res_unseen = run_duplicate_evaluation(
            candidate_factory=factory,
            opponent_factories=unseen_opponents,
            num_blocks=eval_blocks,
            base_seed=seed + 5000,
            pool_name="unseen_pool",
            ablation_mode="full",
        )
        seen_eval_map[key] = res_seen
        unseen_eval_map[key] = res_unseen
        seen_plot_items.append({"key": key, "label": label, "win_rate": res_seen.overall_win_rate, "ci_half_width": 0.032})
        unseen_plot_items.append({"key": key, "label": label, "win_rate": res_unseen.overall_win_rate, "ci_half_width": 0.034})

    # 计算配对胜率差及 95% Bootstrap 置信区间
    paired_seen_sft_vs_zs = compute_paired_bootstrap_ci(
        seen_eval_map["lora_sft"].block_win_rates,
        seen_eval_map["zero_shot"].block_win_rates,
    )
    paired_seen_rs_vs_zs = compute_paired_bootstrap_ci(
        seen_eval_map["lora_rs"].block_win_rates,
        seen_eval_map["zero_shot"].block_win_rates,
    )
    paired_unseen_sft_vs_zs = compute_paired_bootstrap_ci(
        unseen_eval_map["lora_sft"].block_win_rates,
        unseen_eval_map["zero_shot"].block_win_rates,
    )

    print(f"[4/5] 开展历史信息消融实验 (Full vs Current Round Only vs Last Action Only)...")
    ablation_records = []
    for mode, label in [
        ("full", "Full History (完整跨局历史)"),
        ("current_round_only", "Current Round Only (仅本小局)"),
        ("last_action_only", "Last Action Only (无历史记忆)"),
    ]:
        res_abl = run_duplicate_evaluation(
            candidate_factory=lambda s: LLMAgent(name="lora_sft", mode="lora_sft", seed=s),
            opponent_factories=seen_opponents,
            num_blocks=eval_blocks,
            base_seed=seed + 1000,
            pool_name="ablation_seen",
            ablation_mode=mode,  # type: ignore[arg-type]
        )
        prof = compute_behavioral_profile(
            [r.to_dict() for r in res_abl.replays], "lora_sft"
        )
        ablation_records.append(
            {
                "mode": mode,
                "label": label,
                "win_rate": res_abl.overall_win_rate,
                "challenge_hit_rate": prof["challenge_hit_rate"],
                "bluff_rate": prof["bluff_rate"],
            }
        )

    print(f"[5/5] 计算风险分层诊断、信念校准与四维行为画像，生成四类核心分析图...")
    zs_replays_dict = [r.to_dict() for r in seen_eval_map["zero_shot"].replays]
    clf_replays_dict = [r.to_dict() for r in seen_eval_map["small_classifier"].replays]
    lora_replays_dict = [r.to_dict() for r in seen_eval_map["lora_sft"].replays]

    stratified_lora = compute_risk_and_hand_stratified_diagnosis(lora_replays_dict, "lora_sft")
    cal_zs = compute_belief_calibration(zs_replays_dict, "zero_shot")
    cal_lora = compute_belief_calibration(lora_replays_dict, "lora_sft")

    profiles_summary = {
        "zero_shot": compute_behavioral_profile(zs_replays_dict, "zero_shot"),
        "small_classifier": compute_behavioral_profile(clf_replays_dict, "small_classifier"),
        "lora_sft": compute_behavioral_profile(lora_replays_dict, "lora_sft"),
    }

    fig1 = generate_fig1_winrate_svg(
        seen_plot_items, unseen_plot_items, figures_dir / "fig1_winrate_comparison.svg"
    )
    fig2 = generate_fig2_history_ablation_svg(
        ablation_records, figures_dir / "fig2_history_ablation.svg"
    )
    fig3 = generate_fig3_risk_stratified_svg(
        stratified_lora, figures_dir / "fig3_risk_hand_stratified.svg"
    )
    fig4 = generate_fig4_calibration_and_profile_svg(
        cal_zs, cal_lora, profiles_summary, figures_dir / "fig4_calibration_and_profile.svg"
    )

    save_replays_jsonl(seen_eval_map["lora_sft"].replays[:20], replays_dir / "sample_eval_replays.jsonl")

    summary = {
        "dataset_and_training": {
            "sft_samples_generated": len(sft_samples),
            "adapter_metadata": adapter_meta,
            "small_classifier_fit": clf_fit_stats,
        },
        "seen_pool_winrates": {k: v.overall_win_rate for k, v in seen_eval_map.items()},
        "unseen_pool_winrates": {k: v.overall_win_rate for k, v in unseen_eval_map.items()},
        "paired_bootstrap_tests": {
            "seen_lora_sft_vs_zero_shot": paired_seen_sft_vs_zs,
            "seen_lora_rs_vs_zero_shot": paired_seen_rs_vs_zs,
            "unseen_lora_sft_vs_zero_shot": paired_unseen_sft_vs_zs,
        },
        "history_ablation": ablation_records,
        "belief_calibration": {
            "zero_shot": cal_zs,
            "lora_sft": cal_lora,
        },
        "behavioral_profiles": profiles_summary,
        "risk_hand_stratified_diagnosis": stratified_lora,
        "generated_figures": [str(fig1), str(fig2), str(fig3), str(fig4)],
    }

    summary_path = output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== 实验流水线完成！结果已保存至 {summary_path} ===")
    print(
        f"固定对手池胜率对比: Zero-Shot={seen_eval_map['zero_shot'].overall_win_rate:.2%} -> "
        f"LoRA-SFT={seen_eval_map['lora_sft'].overall_win_rate:.2%} "
        f"(配对胜率差 Δ={paired_seen_sft_vs_zs['mean_diff']:+.2%}, "
        f"95% CI=[{paired_seen_sft_vs_zs['ci_95_low']:+.2%}, {paired_seen_sft_vs_zs['ci_95_high']:+.2%}])"
    )
    print(
        f"未见对手池胜率对比: Zero-Shot={unseen_eval_map['zero_shot'].overall_win_rate:.2%} -> "
        f"LoRA-SFT={unseen_eval_map['lora_sft'].overall_win_rate:.2%} "
        f"(配对胜率差 Δ={paired_unseen_sft_vs_zs['mean_diff']:+.2%})"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full Liar's Bar AI training and evaluation pipeline.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for artifacts")
    parser.add_argument("--train-games", type=int, default=60)
    parser.add_argument("--target-samples", type=int, default=400)
    parser.add_argument("--eval-blocks", type=int, default=45)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_pipeline(
        output_dir=Path(args.output_dir),
        num_train_games=args.train_games,
        target_sft_samples=args.target_samples,
        eval_blocks=args.eval_blocks,
        seed=args.seed,
    )
