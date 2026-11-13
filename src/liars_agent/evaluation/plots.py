"""四类核心分析图生成器（原生高清 SVG 矢量图 + HTML 实验仪表盘，无外部绘图库依赖亦可直接产出）."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any


def _rate(value: float | None, denominator: int | None = None) -> str:
    label = "N/A" if value is None else f"{value * 100:.1f}%"
    return f"{label} (n={denominator if denominator is not None else '?'})"


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _svg_header(width: int, height: int, title: str) -> list[str]:
    title = escape(title)
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', sans-serif; background: #ffffff;">',
        f'  <title>{title}</title>',
        f'  <rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>',
        f'  <text x="{width // 2}" y="34" text-anchor="middle" font-size="18" font-weight="700" fill="#1e293b">{title}</text>',
    ]


def generate_fig1_winrate_svg(
    seen_results: list[dict[str, Any]],
    unseen_results: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """图 1：固定对手池 vs 未见对手池整局胜率及 95% 置信区间对照图."""
    w, h = 860, 460
    lines = _svg_header(w, h, "图1：固定对手池与未见对手池整局胜率及 95% CI 对照")

    # 坐标区
    x0, y0, pw, ph = 90, 70, 720, 300
    lines.append(f'  <line x1="{x0}" y1="{y0 + ph}" x2="{x0 + pw}" y2="{y0 + ph}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'  <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + ph}" stroke="#334155" stroke-width="1.5"/>')

    all_items = seen_results + unseen_results
    largest = max((float(item.get("ci_95_high", item["win_rate"])) for item in all_items), default=0.5)
    max_y = max(0.5, math.ceil(largest * 10) / 10)
    ticks = sorted({i / 10 for i in range(round(max_y * 10) + 1)} | {0.25})
    for pct in ticks:
        yy = y0 + ph - int((pct / max_y) * ph)
        dash = 'stroke-dasharray="4,4"' if pct != 0.25 else 'stroke-dasharray="6,3"'
        color = "#94a3b8" if pct == 0.25 else "#e2e8f0"
        lines.append(f'  <line x1="{x0}" y1="{yy}" x2="{x0 + pw}" y2="{yy}" stroke="{color}" stroke-width="1" {dash}/>')
        label = f"{pct * 100:.0f}%" + (" (四人均分参考)" if pct == 0.25 else "")
        lines.append(f'  <text x="{x0 - 10}" y="{yy + 4}" text-anchor="end" font-size="12" fill="#475569">{label}</text>')

    # 图例
    lines.append('  <rect x="560" y="52" width="14" height="14" fill="#2563eb" rx="2"/>')
    lines.append('  <text x="580" y="64" font-size="13" fill="#1e293b">固定训练对手池 (Seen Pool)</text>')
    lines.append('  <rect x="560" y="74" width="14" height="14" fill="#0d9488" rx="2"/>')
    lines.append('  <text x="580" y="86" font-size="13" fill="#1e293b">含未见对手测试池 (Unseen Pool)</text>')

    n_models = len(seen_results)
    group_w = pw / max(1, n_models)
    bar_w = 38

    unseen_by_key = {item.get("key", item["label"]): item for item in unseen_results}
    for idx, s_item in enumerate(seen_results):
        u_item = unseen_by_key.get(s_item.get("key", s_item["label"]))
        cx = x0 + int((idx + 0.5) * group_w)

        for offset, item, color in [(-bar_w - 3, s_item, "#2563eb"), (3, u_item, "#0d9488")]:
            if item is None:
                continue
            wr = float(item["win_rate"])
            bh = int((wr / max_y) * ph)
            bx = cx + offset
            by = y0 + ph - bh
            lines.append(f'  <rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" fill="{color}" rx="3" opacity="0.9"/>')

            # 误差棒
            ex = bx + bar_w // 2
            err_top = by
            if "ci_95_low" in item and "ci_95_high" in item:
                err_top = y0 + ph - int((float(item["ci_95_high"]) / max_y) * ph)
                err_bot = y0 + ph - int((float(item["ci_95_low"]) / max_y) * ph)
                lines.append(f'  <line x1="{ex}" y1="{err_top}" x2="{ex}" y2="{err_bot}" stroke="#0f172a" stroke-width="1.8"/>')
                lines.append(f'  <line x1="{ex - 6}" y1="{err_top}" x2="{ex + 6}" y2="{err_top}" stroke="#0f172a" stroke-width="1.8"/>')
                lines.append(f'  <line x1="{ex - 6}" y1="{err_bot}" x2="{ex + 6}" y2="{err_bot}" stroke="#0f172a" stroke-width="1.8"/>')
            lines.append(f'  <text x="{ex}" y="{err_top - 6}" text-anchor="middle" font-size="11" font-weight="600" fill="#0f172a">{wr * 100:.1f}%</text>')

        lines.append(f'  <text x="{cx}" y="{y0 + ph + 26}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">{escape(str(s_item["label"]))}</text>')
        counts = f"blocks={s_item.get('num_blocks', '?')}/{u_item.get('num_blocks', '?') if u_item else '—'}"
        lines.append(f'  <text x="{cx}" y="{y0 + ph + 44}" text-anchor="middle" font-size="11" fill="#475569">{counts}</text>')

    lines.append('</svg>')
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_fig2_history_ablation_svg(
    ablation_records: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """图 2：历史信息消融对照图（Full History vs Current Round Only vs Last Action Only）."""
    w, h = 820, 440
    lines = _svg_header(w, h, "图2：历史信息消融对照（整局胜率与质疑命中率）")

    x0, y0, pw, ph = 90, 75, 680, 285
    lines.append(f'  <line x1="{x0}" y1="{y0 + ph}" x2="{x0 + pw}" y2="{y0 + ph}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'  <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + ph}" stroke="#334155" stroke-width="1.5"/>')

    for pct in [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]:
        yy = y0 + ph - int(pct * ph)
        lines.append(f'  <line x1="{x0}" y1="{yy}" x2="{x0 + pw}" y2="{yy}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'  <text x="{x0 - 10}" y="{yy + 4}" text-anchor="end" font-size="12" fill="#475569">{pct * 100:.0f}%</text>')

    lines.append('  <rect x="520" y="52" width="14" height="14" fill="#4f46e5" rx="2"/>')
    lines.append('  <text x="540" y="64" font-size="13" fill="#1e293b">整局胜率与 95% CI</text>')
    lines.append('  <rect x="520" y="74" width="14" height="14" fill="#f59e0b" rx="2"/>')
    lines.append('  <text x="540" y="86" font-size="13" fill="#1e293b">质疑命中率 (Challenge Precision)</text>')

    group_w = pw / max(1, len(ablation_records))
    bar_w = 46

    for idx, rec in enumerate(ablation_records):
        cx = x0 + int((idx + 0.5) * group_w)
        for offset, key, color in [
            (-bar_w - 4, "win_rate", "#4f46e5"),
            (4, "challenge_hit_rate", "#f59e0b"),
        ]:
            val = rec.get(key)
            bh = int(val * ph) if val is not None else 0
            bx = cx + offset
            by = y0 + ph - bh
            if val is not None:
                lines.append(f'  <rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" fill="{color}" rx="4"/>')
            label_top = by
            ci = rec.get("win_rate_ci") or {}
            if (key == "win_rate" and val is not None
                    and ci.get("ci_95_low") is not None and ci.get("ci_95_high") is not None):
                ex = bx + bar_w // 2
                err_top = y0 + ph - int(float(ci["ci_95_high"]) * ph)
                err_bottom = y0 + ph - int(float(ci["ci_95_low"]) * ph)
                lines.append(f'  <line x1="{ex}" y1="{err_top}" x2="{ex}" y2="{err_bottom}" stroke="#0f172a" stroke-width="1.8"/>')
                for yy in (err_top, err_bottom):
                    lines.append(f'  <line x1="{ex - 6}" y1="{yy}" x2="{ex + 6}" y2="{yy}" stroke="#0f172a" stroke-width="1.8"/>')
                label_top = min(by, err_top)
            label = "N/A" if val is None else f"{val * 100:.1f}%"
            denominator = rec.get("denominators", {}).get(key, "?")
            lines.append(f'  <text x="{bx + bar_w // 2}" y="{label_top - 8}" text-anchor="middle" font-size="10" font-weight="600" fill="#1e293b">{label}</text>')
            lines.append(f'  <text x="{bx + bar_w // 2}" y="{label_top - 23}" text-anchor="middle" font-size="10" fill="#475569">n={denominator}</text>')

        lines.append(f'  <text x="{cx}" y="{y0 + ph + 28}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">{escape(str(rec["label"]))}</text>')

    lines.append('</svg>')
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_fig3_risk_stratified_svg(
    stratified_data: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """图 3：按手牌真牌数与左轮淘汰风险分层的策略诊断热力图."""
    w, h = 840, 460
    lines = _svg_header(w, h, "图3：按淘汰风险与手牌真牌分层的策略诊断热力矩阵")

    risk_bins = stratified_data["risk_bins"]
    hand_bins = stratified_data["hand_bins"]
    grid = stratified_data["grid"]

    x0, y0 = 210, 95
    cell_w, cell_h = 185, 95

    for c_idx, h_label in enumerate(hand_bins):
        cx = x0 + c_idx * cell_w + cell_w // 2
        lines.append(f'  <text x="{cx}" y="{y0 - 16}" text-anchor="middle" font-size="13" font-weight="700" fill="#1e293b">{h_label}</text>')

    for r_idx, r_label in enumerate(risk_bins):
        ry = y0 + r_idx * cell_h
        lines.append(f'  <text x="{x0 - 14}" y="{ry + cell_h // 2 + 5}" text-anchor="end" font-size="13" font-weight="700" fill="#1e293b">{r_label}</text>')
        for c_idx, h_label in enumerate(hand_bins):
            cx = x0 + c_idx * cell_w
            cell = grid[r_label][h_label]
            ch_rate = cell["challenge_rate"]
            denominators = cell.get("denominators", {})
            challenge_label = _rate(ch_rate, denominators.get("challenge_rate"))
            precision_label = _rate(cell["challenge_precision"], denominators.get("challenge_precision"))
            bluff_label = _rate(cell["play_bluff_rate"], denominators.get("play_bluff_rate"))

            # 根据质疑率着色深浅
            fill = "#f1f5f9" if ch_rate is None else f"rgba(37, 99, 235, {0.15 + 0.65 * ch_rate:.2f})"
            lines.append(
                f'  <rect x="{cx + 4}" y="{ry + 4}" width="{cell_w - 8}" height="{cell_h - 8}" fill="{fill}" stroke="#cbd5e1" rx="6"/>'
            )
            lines.append(
                f'  <text x="{cx + cell_w // 2}" y="{ry + 32}" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">质疑率: {challenge_label}</text>'
            )
            lines.append(
                f'  <text x="{cx + cell_w // 2}" y="{ry + 54}" text-anchor="middle" font-size="11" fill="#1e293b">质疑命中: {precision_label}</text>'
            )
            lines.append(
                f'  <text x="{cx + cell_w // 2}" y="{ry + 74}" text-anchor="middle" font-size="11" fill="#334155">出牌诈唬率: {bluff_label}</text>'
            )

    lines.append('</svg>')
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_fig4_calibration_and_profile_svg(
    zero_shot_cal: dict[str, Any],
    lora_cal: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """图 4：信念估计概率校准曲线 (Reliability Diagram) 与四维行为画像对比图."""
    w, h = 920, 490
    lines = _svg_header(w, h, "图4：诈唬信念概率校准曲线 (左) 与四维策略行为画像 (右)")

    # 左图：可靠性校准曲线
    lx0, ly0, lpw, lph = 75, 80, 320, 300
    lines.append(f'  <rect x="{lx0}" y="{ly0}" width="{lpw}" height="{lph}" fill="#f8fafc" stroke="#cbd5e1"/>')
    lines.append(f'  <line x1="{lx0}" y1="{ly0 + lph}" x2="{lx0 + lpw}" y2="{ly0}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,5"/>')
    lines.append(f'  <text x="{lx0 + lpw // 2}" y="{ly0 + lph + 34}" text-anchor="middle" font-size="12" fill="#334155">模型预测诈唬概率 (Predicted Bluff Prob)</text>')
    lines.append(
        f'  <text x="{lx0 + lpw // 2}" y="{ly0 - 10}" text-anchor="middle" font-size="13" font-weight="600" fill="#0f172a">'
        f'LoRA ECE={_number(lora_cal["ece"])} vs Zero-Shot ECE={_number(zero_shot_cal["ece"])}</text>'
    )

    for cal, color, label_y in [
        (zero_shot_cal, "#ef4444", ly0 + 22),
        (lora_cal, "#16a34a", ly0 + 42),
    ]:
        pts: list[str] = []
        for b in cal.get("bins", []):
            if b["count"] <= 0:
                continue
            px = lx0 + int(float(b["mean_predicted_prob"]) * lpw)
            py = ly0 + lph - int(float(b["empirical_bluff_freq"]) * lph)
            pts.append(f"{px},{py}")
            lines.append(f'  <circle cx="{px}" cy="{py}" r="4.5" fill="{color}"/>')
            lines.append(f'  <text x="{px + 6}" y="{py - 5}" font-size="10" fill="{color}">n={b["count"]}</text>')
        if len(pts) >= 2:
            lines.append(f'  <polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>')

    lines.append(f'  <circle cx="{lx0 + 20}" cy="{ly0 + 20}" r="5" fill="#ef4444"/><text x="{lx0 + 32}" y="{ly0 + 24}" font-size="12" fill="#1e293b">Zero-Shot n={zero_shot_cal.get("num_samples", 0)}/{zero_shot_cal.get("num_candidates", "?")}</text>')
    lines.append(f'  <circle cx="{lx0 + 20}" cy="{ly0 + 40}" r="5" fill="#16a34a"/><text x="{lx0 + 32}" y="{ly0 + 44}" font-size="12" fill="#1e293b">LoRA-SFT n={lora_cal.get("num_samples", 0)}/{lora_cal.get("num_candidates", "?")}</text>')
    population = zero_shot_cal.get("population", "fixed_bank")
    population_label = "各策略自身访问的决策（非相同样本比较）" if population == "on_policy_decisions" else "同一冻结观测集"
    lines.append(f'  <text x="{lx0 + lpw // 2}" y="455" text-anchor="middle" font-size="11" fill="#475569">{population_label}</text>')

    # 右图：四维行为画像柱状对比
    rx0, ry0, rpw, rph = 480, 80, 390, 300
    lines.append(f'  <line x1="{rx0}" y1="{ry0 + rph}" x2="{rx0 + rpw}" y2="{ry0 + rph}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'  <line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry0 + rph}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'  <text x="{rx0 + rpw // 2}" y="{ry0 - 10}" text-anchor="middle" font-size="13" font-weight="600" fill="#0f172a">四维策略行为画像对比</text>')

    metrics_keys = [
        ("bluff_rate", "诈唬率"),
        ("caught_when_bluffing_rate", "诈唬被抓率"),
        ("challenge_rate", "主动质疑率"),
        ("challenge_hit_rate", "质疑命中率"),
    ]
    agent_colors = [item for item in [
        ("zero_shot", "#ef4444", "Zero-Shot"),
        ("small_classifier", "#f59e0b", "Small-Clf"),
        ("lora_sft", "#2563eb", "LoRA-SFT"),
    ] if item[0] in profiles]

    group_w = rpw / len(metrics_keys)
    bar_w = 22
    for m_idx, (m_key, m_label) in enumerate(metrics_keys):
        cx = rx0 + int((m_idx + 0.5) * group_w)
        for a_idx, (a_key, color, _) in enumerate(agent_colors):
            prof = profiles[a_key]
            val = prof.get(m_key)
            bh = int(val * rph) if val is not None else 0
            bx = cx + (a_idx - (len(agent_colors) - 1) / 2) * (bar_w + 3) - bar_w // 2
            by = ry0 + rph - bh
            if val is not None:
                lines.append(f'  <rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" fill="{color}" rx="2"/>')
            value_label = "N/A" if val is None else f"{val * 100:.0f}%"
            denominator = prof.get("denominators", {}).get(m_key, "?")
            lines.append(f'  <text x="{bx + bar_w // 2}" y="{by - 5}" text-anchor="middle" font-size="10" fill="#1e293b">{value_label}</text>')
            lines.append(f'  <text x="{bx + bar_w // 2}" y="{ry0 + rph + 13}" text-anchor="middle" font-size="8" fill="#475569">n={denominator}</text>')
        lines.append(f'  <text x="{cx}" y="{ry0 + rph + 32}" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b">{m_label}</text>')

    for a_idx, (_, color, a_label) in enumerate(agent_colors):
        lx = rx0 + 20 + a_idx * 120
        lines.append(f'  <rect x="{lx}" y="{ry0 + rph + 42}" width="12" height="12" fill="{color}" rx="2"/>')
        lines.append(f'  <text x="{lx + 16}" y="{ry0 + rph + 52}" font-size="12" fill="#334155">{a_label}</text>')

    lines.append('</svg>')
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
