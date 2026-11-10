"""评测与可视化模块导出."""

from liars_agent.evaluation.duplicate_runner import (
    DuplicateEvalResult,
    run_duplicate_evaluation,
)
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

__all__ = [
    "DuplicateEvalResult",
    "compute_behavioral_profile",
    "compute_belief_calibration",
    "compute_paired_bootstrap_ci",
    "compute_risk_and_hand_stratified_diagnosis",
    "generate_fig1_winrate_svg",
    "generate_fig2_history_ablation_svg",
    "generate_fig3_risk_stratified_svg",
    "generate_fig4_calibration_and_profile_svg",
    "run_duplicate_evaluation",
]
