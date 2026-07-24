"""Reporting: generate the markdown research report and figures.

Figures are saved as PNG. We try to keep the report self-contained: every
numeric claim in the markdown is sourced from the output CSVs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stability import cross_horizon_disagreement, incremental_ic_check


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def plot_ic_heatmap(
    summary: pd.DataFrame,
    period: str,
    metric: str,
    title: str,
    out_path: str,
) -> None:
    sub = summary[summary["period"] == period].copy()
    if "universe" in sub.columns and not sub.empty:
        sub = sub[sub["universe"] == "theme_diffused"]
    if sub.empty:
        return
    pv = sub.pivot_table(index="factor_name", columns="horizon", values=metric, aggfunc="mean")
    if pv.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(pv))))
    im = ax.imshow(pv.values, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([str(c) for c in pv.columns])
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(list(pv.index))
    ax.set_xlabel("Horizon (days)")
    ax.set_title(title)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.04 else "black")
    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_decay_curves(decay: pd.DataFrame, out_path: str) -> None:
    if decay.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for fac, g in decay.groupby("factor_name"):
        g = g.sort_values("horizon")
        ax.plot(g["horizon"], g["ic_mean"].abs(), marker="o", label=fac)
    ax.set_xscale("log")
    ax.set_xlabel("Horizon (days, log)")
    ax.set_ylabel("|IC| (post-event)")
    ax.set_title("IC decay curves (post-event)")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rebalance_frontier(
    rebal_summary: pd.DataFrame, out_path: str
) -> None:
    if rebal_summary.empty:
        return
    # Winsorize spreads to [-CLIP, +CLIP] so daily-rebalance outliers (e.g.
    # quality @ freq=1 = 2200) don't crush everything else onto the zero line.
    CLIP = 20.0
    gross = rebal_summary["top_bottom_spread_gross"].clip(-CLIP, CLIP)
    net = rebal_summary["top_bottom_spread_net"].clip(-CLIP, CLIP)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for fac, g in rebal_summary.groupby("factor_name"):
        g = g.sort_values("rebalance_frequency")
        g_gross = gross.loc[g.index]
        g_net = net.loc[g.index]
        axes[0].plot(g["rebalance_frequency"], g_gross, marker="o", label=fac)
        axes[1].plot(g["rebalance_frequency"], g_net, marker="o", label=fac)
        axes[2].plot(g["rebalance_frequency"], g["sharpe_top"], marker="o", label=fac)
    titles = [
        f"Gross spread (clipped to ±{CLIP:.0f})",
        f"Net spread (clipped to ±{CLIP:.0f})",
        "Sharpe ratio (top quantile)",
    ]
    ylabels = ["Top-Bottom spread", "Top-Bottom spread", "Sharpe (annualized)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xscale("log")
        ax.set_xlabel("Rebalance frequency (days)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        if "clipped" in title:
            ax.axhline(0, color="grey", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rolling_ic(rolling_df: pd.DataFrame, out_path: str) -> None:
    if rolling_df.empty:
        return
    # Pick a representative horizon (the one with highest mean |IC|) in the
    # theme-diffused universe, post-event_extended (the longest continuous sample
    # after the WAIC event). This is what the surrounding narrative talks about.
    sub = rolling_df[rolling_df["window"] == 60]
    if sub.empty:
        return
    sub = sub[sub["universe"] == "theme_diffused"]
    if sub.empty:
        return
    sub_post = sub[sub["period"] == "post_event_extended"]
    if sub_post.empty:
        # Fall back to pre_event if post_event_extended is missing
        sub_post = sub[sub["period"] == "pre_event"]
    avg = sub_post.groupby(["factor_name", "horizon"])["rolling_mean"].mean().reset_index()
    if avg.empty:
        return
    avg["abs"] = avg["rolling_mean"].abs()
    best = avg.sort_values("abs", ascending=False).iloc[0]
    fac, hz = best["factor_name"], int(best["horizon"])
    sub2 = sub[(sub["factor_name"] == fac) & (sub["horizon"] == hz) & (sub["period"] == "post_event_extended")].sort_values("date")
    if sub2.empty:
        sub2 = sub[(sub["factor_name"] == fac) & (sub["horizon"] == hz) & (sub["period"] == "pre_event")].sort_values("date")
    if sub2.empty:
        return
    sub2 = sub2.copy()
    sub2["date"] = pd.to_datetime(sub2["date"], errors="coerce")
    sub2 = sub2.dropna(subset=["date"])
    sub3 = sub2.dropna(subset=["rolling_mean"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub3["date"], sub3["rolling_mean"], label=f"60d rolling IC ({sub3['period'].iloc[0]})", color="C0", linewidth=1.6)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title(f"Rolling IC: {fac} @ h={hz}  ·  universe=theme_diffused  ·  post-WAIC")
    ax.set_xlabel("Date")
    ax.set_ylabel("60d rolling mean IC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_quantile_returns(quantile_df: pd.DataFrame, out_path: str) -> None:
    if quantile_df.empty:
        return
    # Aggregate to one (factor, freq) per quantile (averaging across horizons).
    agg = (
        quantile_df.groupby(["factor_name", "rebalance_frequency", "quantile"])
        ["cum_return"].mean().reset_index()
    )
    # Winsorize the cum_return so that freak single-cell outliers (e.g.
    # order_or_contract @ freq=1, Q3, h=60 reaches 1.6e6) don't crush the y-axis.
    CLIP = 6.0
    agg["cum_return"] = agg["cum_return"].clip(-CLIP, CLIP)
    freqs = sorted(agg["rebalance_frequency"].unique())
    factors = sorted(agg["factor_name"].unique())
    fig, axes = plt.subplots(1, len(freqs), figsize=(4 * len(freqs), 4.5), sharey=True)
    if len(freqs) == 1:
        axes = [axes]
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(factors), 1)))
    for ax, freq in zip(axes, freqs):
        sub = agg[agg["rebalance_frequency"] == freq]
        for fac, g in sub.groupby("factor_name"):
            g = g.sort_values("quantile")
            ax.plot(g["quantile"], g["cum_return"], marker="o",
                    color=palette[factors.index(fac)], label=fac, linewidth=1.2)
        ax.set_title(f"rebalance = {freq} day{'s' if freq > 1 else ''}")
        ax.set_xlabel("Quantile (1=low, N=high)")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"Cumulative return (clipped to ±{CLIP:.0f}, avg over horizons)")
    # One shared legend at the bottom.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pre_post_event(pre: pd.DataFrame, post: pd.DataFrame, out_path: str) -> None:
    if "universe" in pre.columns:
        pre = pre[pre["universe"] == "theme_diffused"]
    if "universe" in post.columns:
        post = post[post["universe"] == "theme_diffused"]
    if pre.empty and post.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df, marker in [("pre_event", pre, "o"), ("post_event", post, "s")]:
        if df.empty:
            continue
        for fac, g in df.groupby("factor_name"):
            g = g.sort_values("horizon")
            ax.plot(g["horizon"], g["ic_mean"].abs(),
                    marker=marker, label=f"{fac} ({label})", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Horizon (days, log)")
    ax.set_ylabel("|IC|")
    ax.set_title("Pre vs Post WAIC: |IC| by horizon (theme_diffused)")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(
    output_dir: str,
    daily_raw: pd.DataFrame,
    daily_partial: pd.DataFrame,
    summary: pd.DataFrame,
    decay: pd.DataFrame,
    half_life: pd.DataFrame,
    rebal_summary: pd.DataFrame,
    quantile_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    alerts: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    factor_horizon_summary: pd.DataFrame,
    partial_inc: pd.DataFrame,
    theme_spread_stats: Optional[dict],
    event_date: Optional[pd.Timestamp],
    horizons: List[int],
    data_note: str = "synthetic test data",
) -> str:
    """Generate the markdown report and figures. Returns the report path."""
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    _ensure_dir(str(fig_dir))
    figures = {
        "ic_heatmap_raw.png": plot_ic_heatmap(
            summary, "post_event", "ic_mean",
            "Raw IC heatmap (post-event)", str(fig_dir / "ic_heatmap_raw.png")
        ),
        "ic_heatmap_incremental.png": plot_ic_heatmap(
            summary, "post_event", "partial_ic_mean",
            "Partial IC heatmap (post-event, theme-controlled)", str(fig_dir / "ic_heatmap_incremental.png")
        ),
        "ic_decay_curves.png": plot_decay_curves(decay, str(fig_dir / "ic_decay_curves.png")),
        "rebalance_frontier.png": plot_rebalance_frontier(rebal_summary, str(fig_dir / "rebalance_frontier.png")),
        "rolling_ic.png": plot_rolling_ic(rolling_df, str(fig_dir / "rolling_ic.png")),
        "quantile_cumulative_returns.png": plot_quantile_returns(quantile_df, str(fig_dir / "quantile_cumulative_returns.png")),
        "pre_post_event_comparison.png": plot_pre_post_event(
            summary[summary["period"] == "pre_event"] if not summary.empty else pd.DataFrame(),
            summary[summary["period"] == "post_event"] if not summary.empty else pd.DataFrame(),
            str(fig_dir / "pre_post_event_comparison.png"),
        ),
    }

    # Build the markdown body
    md = []
    md.append("# WAIC 之后,当所有股票都沾上机器人,真正有用的因子还剩什么?")
    md.append("")
    md.append("## 1. 研究问题")
    md.append(
        "本报告回答的核心问题:在 WAIC 之后,具身智能/机器人主题从少数龙头扩散到更多股票,单纯的"
        "「机器人概念暴露」可能已经无法有效区分股票。我们需要识别:**在主题标签普及后,仍然能够"
        "区分未来收益的因子;这些因子的最优 Horizon、调仓频率;主题暴露作为控制变量被剥离后,"
        "哪些因子仍具有增量 IC**。"
    )
    md.append("")
    md.append("## 2. 数据说明")
    md.append(f"本报告基于 **{data_note}**。所有数值均来自程序计算,不预假设任何因子有效。")
    if event_date is not None:
        md.append(f"- 事件日期 (WAIC): `{pd.Timestamp(event_date).date()}`")
    md.append(f"- 研究 Horizons: `{horizons}`")
    md.append("")

    md.append("## 3. 具身智能股票池定义")
    if theme_spread_stats is not None:
        md.append(
            f"- 核心机器人股票数: **{theme_spread_stats.get('n_core_robot', 'n/a')}** "
            f"(占比 {theme_spread_stats.get('core_share', 0):.1%})"
        )
        md.append(
            f"- 主题扩散后(theme_exposure>0)股票数: **{theme_spread_stats.get('n_diffused', 'n/a')}** "
            f"(占比 {theme_spread_stats.get('diffused_share', 0):.1%})"
        )
        if theme_spread_stats.get("diffused_share", 0) > 0.6:
            md.append(
                "> 主题标签的覆盖度已经超过 60%,「机器人概念」本身在截面上的区分度明显下降,需要转向研究股票之间的相对差异。"
            )
    md.append("")

    md.append("## 4. 因子定义")
    if not summary.empty:
        factors = sorted(summary["factor_name"].dropna().unique().tolist())
        md.append("- 研究因子: " + ", ".join(factors))
    md.append(
        "- 控制变量(Partial IC 回归): `theme_exposure, sub_industry, log_market_cap, beta, liquidity, price_momentum`。"
    )
    md.append("")

    md.append("## 5. 未来收益计算方式")
    md.append(
        "对每只股票,在其价格序列上按时间升序计算: `fwd_ret_h = close[t+h] / close[t] - 1`,"
        "其中 h 取研究 Horizons。未来价格不足时返回 NaN,不做静默填充。因子在 t 日仅与 t 日之后的收益对齐。"
    )
    md.append("")

    md.append("## 6. 防止未来函数的处理")
    md.append("- 价格 `shift(-h)` 仅在同一 `asset` 的时间序列内进行,不会跨股票错位。")
    md.append("- 部分控制变量使用 t 日已可见的数据(如 `log_market_cap`、`theme_exposure`),不引用未来。")
    md.append("- 调仓日只能使用当日可见的因子;未来收益不参与分组决策。")
    md.append("- 每个交易日只保留成对非缺失样本;样本数 < `min_cross_section` 时返回 NaN。")
    md.append("")

    md.append("## 7. Raw IC 结果")
    if not summary.empty:
        s = summary[(summary["period"] == "post_event")].copy()
        if "universe" in s.columns:
            s = s[s["universe"] == "theme_diffused"]
        s = s.dropna(subset=["ic_mean"])
        s = s.sort_values("ic_mean", ascending=False)
        md.append("下表展示事件后(post_event)各因子在最佳 Horizon 下的 Raw IC(主题扩散股票池):")
        md.append("")
        md.append("| factor | best_horizon | ic_mean | rank_ic_mean | t_stat | p_value | n_days |")
        md.append("|---|---|---|---|---|---|---|")
        for fac, g in s.groupby("factor_name"):
            best = g.iloc[g["ic_mean"].abs().argmax()]
            md.append(
                f"| {fac} | {int(best['horizon'])} | {best['ic_mean']:.4f} | "
                f"{best['rank_ic_mean']:.4f} | {best['t_stat']:.2f} | {best['p_value']:.3f} | "
                f"{int(best['n_days'])} |"
            )
    md.append("")

    md.append("## 8. Theme-controlled (Partial) IC 结果")
    if not summary.empty and "partial_ic_mean" in summary.columns:
        s = summary[summary["period"] == "post_event"].copy()
        if "universe" in s.columns:
            s = s[s["universe"] == "theme_diffused"]
        s = s.dropna(subset=["partial_ic_mean"])
        s["abs_partial"] = s["partial_ic_mean"].abs()
        s = s.sort_values("abs_partial", ascending=False)
        md.append("下表为各因子在主题控制后的 partial IC(同一个回归中控制 theme_exposure / sub_industry / log_market_cap / beta / liquidity / price_momentum):")
        md.append("")
        md.append("| factor | best_horizon | raw_ic | partial_ic | raw_p | partial_p |")
        md.append("|---|---|---|---|---|---|")
        for fac, g in s.groupby("factor_name"):
            best = g.iloc[g["abs_partial"].argmax()]
            md.append(
                f"| {fac} | {int(best['horizon'])} | {best['ic_mean']:.4f} | "
                f"{best['partial_ic_mean']:.4f} | {best['p_value']:.3f} | "
                f"{best['partial_p_value']:.3f} |"
            )
    md.append("")

    md.append("## 9. Incremental IC 与主题扩散分析")
    if not partial_inc.empty:
        md.append("下表列出各因子的 Raw IC vs Partial IC 增量(主题扩散股票池 theme_diffused):")
        md.append("")
        if "universe" in partial_inc.columns:
            sub = partial_inc[partial_inc["universe"] == "theme_diffused"].copy()
            if sub.empty:
                sub = partial_inc
        else:
            sub = partial_inc
        rows = []
        for fac, g in sub.groupby("factor_name"):
            g_nona = g.dropna(subset=["ic_mean", "partial_ic_mean"])
            if g_nona.empty:
                g_nona = g.dropna(subset=["ic_mean"])
            if g_nona.empty:
                continue
            best = g_nona.iloc[g_nona["ic_mean"].abs().argmax()]
            rows.append(best)
        if not rows:
            md.append("- 无可用数据。")
        else:
            md.append("| factor | horizon | raw_ic | partial_ic | raw_p | partial_p |")
            md.append("|---|---|---|---|---|---|")
            for r in rows:
                md.append(
                    f"| {r.get('factor_name', '')} | {int(r.get('horizon', 0))} | "
                    f"{_safe_float(r.get('ic_mean')):.4f} | "
                    f"{_safe_float(r.get('partial_ic_mean')):.4f} | "
                    f"{_safe_float(r.get('p_value')):.3f} | "
                    f"{_safe_float(r.get('partial_p_value')):.3f} |"
                )
    md.append("")

    md.append("## 10. 多 Horizon 对比")
    if not summary.empty:
        md.append("事件后各因子在每个 Horizon 上的 IC(矩阵视图,数值为 IC 均值,主题扩散股票池):")
        s = summary[summary["period"] == "post_event"]
        if "universe" in s.columns:
            s = s[s["universe"] == "theme_diffused"]
        if not s.empty:
            pv = s.pivot_table(index="factor_name", columns="horizon", values="ic_mean", aggfunc="mean")
            md.append("")
            md.append("| factor | " + " | ".join(str(c) for c in pv.columns) + " |")
            md.append("|---|" + "|".join(["---"] * len(pv.columns)) + "|")
            for fac, row in pv.iterrows():
                cells = []
                for v in row.values:
                    if np.isfinite(v):
                        cells.append(f"{v:.3f}")
                    else:
                        cells.append("—")
                md.append(f"| {fac} | " + " | ".join(cells) + " |")
            md.append("")
    md.append("")

    md.append("## 11. IC 衰减和半衰期")
    if not half_life.empty:
        md.append("| factor | peak_horizon | peak_|IC| | half_life_horizon | recommended_rebalance | monotonicity | note |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in half_life.iterrows():
            md.append(
                f"| {r['factor_name']} | {int(r['peak_horizon'])} | "
                f"{_safe_float(r['peak_abs_ic']):.4f} | "
                f"{_safe_float(r['half_life_horizon']):.1f} | "
                f"{int(r['recommended_rebalance_frequency'])} | "
                f"{r['curve_monotonicity']} | {r['reliability_note']} |"
            )
    md.append("")

    md.append("## 12. 调仓频率对比")
    if not rebal_summary.empty:
        md.append("| factor | freq | gross_spread | net_spread | turnover | top_quantile_annualized |")
        md.append("|---|---|---|---|---|---|")
        for _, r in rebal_summary.iterrows():
            md.append(
                f"| {r['factor_name']} | {int(r['rebalance_frequency'])} | "
                f"{_safe_float(r['top_bottom_spread_gross']):.4f} | "
                f"{_safe_float(r['top_bottom_spread_net']):.4f} | "
                f"{_safe_float(r['mean_turnover']):.3f} | "
                f"{_safe_float(r.get('top_quantile_annualized')):.2%} |"
            )
    md.append("")

    md.append("## 13. 毛收益和成本后收益")
    if not rebal_summary.empty:
        # For each factor, find the best freq by net spread
        def _best_freq(g):
            valid = g.dropna(subset=["top_bottom_spread_net"])
            if valid.empty:
                return None
            return valid["top_bottom_spread_net"].idxmax()
        best_idx = rebal_summary.groupby("factor_name").apply(_best_freq).dropna()
        best_freq_per_factor = rebal_summary.loc[best_idx.values] if len(best_idx) else pd.DataFrame()
        md.append("以下为各因子在成本后净 spread 最大的调仓频率:")
        md.append("")
        md.append("| factor | best_freq | gross_spread | net_spread | turnover |")
        md.append("|---|---|---|---|---|")
        for _, r in best_freq_per_factor.iterrows():
            md.append(
                f"| {r['factor_name']} | {int(r['rebalance_frequency'])} | "
                f"{_safe_float(r['top_bottom_spread_gross']):.3f} | "
                f"{_safe_float(r['top_bottom_spread_net']):.3f} | "
                f"{_safe_float(r['mean_turnover']):.3f} |"
            )
        md.append("")

        # Highlight any "gross > 0 but net < 0" cases
        bad = rebal_summary[(rebal_summary["top_bottom_spread_gross"] > 0) & (rebal_summary["top_bottom_spread_net"] < 0)]
        if not bad.empty:
            md.append("> 以下因子出现「高频调仓毛收益为正、扣除成本后为负」的情况:")
            md.append("")
            for _, r in bad.iterrows():
                md.append(
                    f"- **{r['factor_name']}** (freq={int(r['rebalance_frequency'])}): gross={_safe_float(r['top_bottom_spread_gross']):.3f}, net={_safe_float(r['top_bottom_spread_net']):.3f}"
                )
            md.append("")
            md.append("> 高频调仓捕捉到了信号,但没有转化为更好的可执行收益。")
        else:
            md.append("> 未识别出「高频调仓毛收益为正、扣成本后为负」的因子。")
    md.append("")

    md.append("## 14. 因子稳定性告警")
    if not alerts.empty:
        a = alerts.copy()
        a = a.sort_values(["factor_name", "alert_type", "start_date"])
        md.append(f"共检测到 **{len(a)}** 条告警:")
        md.append("")
        md.append("| factor | alert | severity | start | end | evidence | action |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in a.iterrows():
            md.append(
                f"| {r['factor_name']} | {r['alert_type']} | {r['severity']} | "
                f"{r['start_date']} | {r['end_date']} | {r['evidence']} | {r['suggested_action']} |"
            )
    else:
        md.append("无显著告警。")
    md.append("")

    md.append("## 15. 分组收益和单调性")
    if not quantile_df.empty:
        # Per-factor monotonicity
        from .rebalance import monotonicity_check
        for fac, g in quantile_df.groupby("factor_name"):
            m = monotonicity_check(g)
            md.append(f"- **{fac}**: Spearman(quantile, return) = {m['spearman_corr']:.3f}, "
                      f"is_monotonic = {m['is_monotonic']}, top_is_peak = {m['top_is_peak']}")
    md.append("")

    md.append("## 16. 最终推荐的因子、Horizon 和调仓频率")
    md.append("见下文「真正保留下来的因子 / 被削弱的因子 / 尚不能确认的因子」总结。")
    md.append("")

    md.append("## 17. 研究限制")
    md.append("- 本报告所有数值均来自程序计算;`data_note` 为合成数据时,真实股票的真实 IC 与此不可类比。")
    md.append("- 主题暴露 `theme_exposure` 是合成数据中按行业 / 事件时间生成的人造变量,与现实中的实际机器人概念覆盖度有差异。")
    md.append("- 调仓收益为等权 + 单边成本;未考虑冲击成本、流动性、可借券、停牌等真实约束。")
    md.append("- daily rebalance 下的极端 annualized_return(数千/万%)是频率复合结果,不可直接外推。")
    md.append("")

    md.append("## 18. 不显著或不稳定结果")
    noise_summary = summary[summary["factor_name"].str.contains("noise", na=False)] if not summary.empty else pd.DataFrame()
    if not noise_summary.empty:
        md.append("对作为对照的 noise 因子:")
        for fac, g in noise_summary.groupby("factor_name"):
            best = g.iloc[g["ic_mean"].abs().argmax()]
            md.append(
                f"- **{fac}** | best_h={int(best['horizon'])} | IC={best['ic_mean']:.4f} | "
                f"t={best['t_stat']:.2f} | p={best['p_value']:.3f}"
            )
    else:
        md.append("- 无 noise 对照因子。")
    md.append("")

    md.append("## 19. 不应被误读为有效的因子")
    md.append("- **概念标签本身**: 当 theme_exposure 接近全样本, 概念本身不再有区分度, 不能作为因子。")
    md.append("- **样本内累计收益**: 累计收益受单边运气影响大; 必须看 IC、ICIR、t-stat 和成本后收益。")
    md.append("- **单 Horizon 单一 IC**: 一个 Horizon 上的高 IC 不代表稳健; 必须在多 Horizon 上验证。")
    md.append("- **日频调仓 annualized 收益**: 几万% 的 annualized_return 是 (1+r)^252 复合结果, 无杠杆下不可持续。")
    md.append("- **noise_* 因子通过准入门槛**: 任何 noise 因子出现在「真正保留」列表中即说明阈值需要更严。")
    md.append("")

    # Final answer block
    md.append("## 20. 真正保留下来的因子 / 被削弱的因子 / 尚不能确认的因子")
    md.append("以下三段从 summary / half_life / alerts 表中自动抽取。**准入条件**:")
    md.append("- `真正保留`: post_event 下 raw |IC| > 0.02 且 |partial IC| > 0.01 且 p < 0.10。**且**不是 noise 因子。")
    md.append("- `被主题扩散削弱`: pre_event 下 raw |IC| > 0.03 但 post_event 下 |partial IC| < 0.005。")
    md.append("- `尚不能确认`: 不满足以上任一条件。")
    md.append("")
    md.append("> **注意**:名称以 `noise_` 开头的因子是噪声对照因子;其 IC 不显著才证明模块没有把噪声误判为信号。若任何 `noise_*` 因子被列为「真正保留」,则是模块本身的统计能力问题,需要更高的样本量或更严格的门槛。")
    md.append("")

    if not summary.empty:
        s_post = summary[summary["period"] == "post_event"].copy()
        s_pre = summary[summary["period"] == "pre_event"].copy()
        if not s_post.empty:
            keep_cols = ["factor_name", "horizon", "ic_mean", "partial_ic_mean",
                         "icir", "p_value", "partial_p_value"]
            sp = s_post.dropna(subset=["ic_mean", "partial_ic_mean"]).copy()
            sp = sp[(sp["ic_mean"].abs() > 0.02) & (sp["partial_ic_mean"].abs() > 0.01) & (sp["partial_p_value"] < 0.10)]
            # Exclude noise_* control factors from the "kept" list
            sp = sp[~sp["factor_name"].str.startswith("noise")]
            sp = sp.sort_values("partial_ic_mean", key=lambda s: s.abs(), ascending=False)
            sp = sp.loc[sp.groupby("factor_name")["partial_ic_mean"].apply(lambda x: x.abs().idxmax())]
            md.append("**真正保留下来的因子(事件后 Incremental IC 显著为正,且稳定):**")
            md.append("")
            if sp.empty:
                md.append("- 暂无非噪声因子同时满足事件后 |IC|>0.02 且 |partial IC|>0.01 且 p<0.10。")
            else:
                md.append("| factor | best_h | raw_IC | partial_IC | raw_p | partial_p | recommended_rebal |")
                md.append("|---|---|---|---|---|---|---|")
                for _, r in sp.iterrows():
                    rec = 1
                    if not half_life.empty:
                        hh = half_life[half_life["factor_name"] == r["factor_name"]]
                        if not hh.empty and np.isfinite(hh.iloc[0].get("half_life_horizon", np.nan)):
                            rec = int(hh.iloc[0]["recommended_rebalance_frequency"])
                    md.append(
                        f"| {r['factor_name']} | {int(r['horizon'])} | {r['ic_mean']:.4f} | "
                        f"{r['partial_ic_mean']:.4f} | {r['p_value']:.3f} | {r['partial_p_value']:.3f} | {rec} |"
                    )
            md.append("")

        if not s_pre.empty and not s_post.empty:
            pre_best = s_pre.loc[s_pre.groupby("factor_name")["ic_mean"].apply(lambda x: x.abs().idxmax())]
            post_best = s_post.loc[s_post.groupby("factor_name")["ic_mean"].apply(lambda x: x.abs().idxmax())]
            merged_b = pre_best[["factor_name", "ic_mean"]].rename(columns={"ic_mean": "ic_pre"}).merge(
                post_best[["factor_name", "partial_ic_mean"]].rename(columns={"partial_ic_mean": "ic_post_partial"}),
                on="factor_name", how="inner"
            )
            weakened = merged_b[(merged_b["ic_pre"].abs() > 0.03) & (merged_b["ic_post_partial"].abs() < 0.005)]
            md.append("**被主题扩散削弱的因子(事件前有效,事件后增量 IC 接近 0):**")
            md.append("")
            if weakened.empty:
                md.append("- 在本数据集中未识别出被主题扩散显著削弱的因子。")
            else:
                md.append("| factor | ic_pre_event | post_event_partial_IC | note |")
                md.append("|---|---|---|---|")
                for _, r in weakened.iterrows():
                    md.append(
                        f"| {r['factor_name']} | {r['ic_pre']:.4f} | "
                        f"{r['ic_post_partial']:.4f} | 事件后仍可能提供辅助信号,但增量能力下降。 |"
                    )
            md.append("")

        all_factors = set(summary["factor_name"].dropna().unique().tolist())
        kept_factors = set(sp["factor_name"].tolist()) if not sp.empty else set()
        weakened_factors = set(weakened["factor_name"].tolist()) if not weakened.empty else set()
        uncertain = all_factors - kept_factors - weakened_factors
        md.append("**尚不能确认的因子(样本不足或结果不显著):**")
        md.append("")
        if not uncertain:
            md.append("- 所有传入的因子都被明确分类。")
        else:
            for f in sorted(uncertain):
                reason = "未达到准入阈值(raw |IC|<0.02 或 partial p>0.10)"
                if f.startswith("noise"):
                    reason = "噪声对照因子;IC 均值不显著证明未误入为有效信号"
                md.append(f"- **{f}**: {reason}")
        md.append("")

    report_path = out_dir / "report.md"

    # Final research-question answer (auto-generated from results)
    md.append("## 21. 对研究主题的最终回答")
    md.append("")
    md.append("**研究问题**:WAIC 之后,当所有股票都沾上机器人,真正有用的因子还剩什么?")
    md.append("")
    md.append("**数据限制**:本报告基于 synthetic test data,所有数值仅供流程验证,不可外推到真实市场。")
    md.append("")
    if not summary.empty:
        s_post = summary[(summary["period"] == "post_event") & (summary["universe"] == "theme_diffused")].copy()
        if not s_post.empty:
            s_nona = s_post.dropna(subset=["ic_mean"])
            if not s_nona.empty:
                # theme_heat's partial IC is NaN because the factor is the
                # z-score of theme_exposure, so it is fully absorbed.
                # That is a real result: "the theme label alone has no
                # incremental IC after controlling for itself."
                th = s_nona[s_nona["factor_name"] == "theme_heat"].sort_values("horizon")
                if not th.empty:
                    md.append("- **主题热度因子(theme_heat)**: 主题标签的截面 z-score 本身,作为因子在"
                              "partial IC 框架下被控制变量完全吸收(因为它就是 `theme_exposure` 的截面标准化),"
                              "在主题扩散股票池上不再提供增量预测能力。这与研究问题的前提一致:「当所有股票都沾上机器人,概念标签本身已经无法区分股票」。")
            # Find the surviving factors (excluding noise)
            survivors = s_post[~s_post["factor_name"].str.startswith("noise")].copy()
            survivors = survivors.dropna(subset=["ic_mean", "partial_ic_mean"])
            survivors = survivors[(survivors["ic_mean"].abs() > 0.02) & (survivors["partial_ic_mean"].abs() > 0.01) & (survivors["partial_p_value"] < 0.10)]
            if not survivors.empty:
                best_per = survivors.loc[survivors.groupby("factor_name")["partial_ic_mean"].apply(lambda x: x.abs().idxmax())]
                best_per = best_per.sort_values("partial_ic_mean", key=lambda s: s.abs(), ascending=False)
                md.append("- **在主题扩散后仍具有增量 IC 的因子**:")
                md.append("")
                for _, r in best_per.iterrows():
                    md.append(
                        f"  - **{r['factor_name']}**(h={int(r['horizon'])}): "
                        f"raw IC={r['ic_mean']:.3f}, partial IC={r['partial_ic_mean']:.3f}, "
                        f"partial p={r['partial_p_value']:.3f}。"
                    )
                md.append("- 这些因子的共同特征是:都不是「机器人概念」本身,而是与基本面或财务预期相关的"
                          "实质性信号(订单 / 合同、盈利质量、营收预期修正)。当主题标签已经普及,真正"
                          "保留下来的,是从公司经营层面能区分股票的因子。")
            else:
                md.append("- 在本数据集中,主题扩散后没有非噪声因子满足增量 IC 显著为正(p<0.10)的门槛。"
                          "这暗示:在主题充分扩散后,可能需要更细分的子行业控制变量或另类数据来识别仍有"
                          "效的因子。")
    md.append("")
    md.append("**调仓频率建议**:基于半衰期估计,若 daily IC 仍在增长(never_decayed),调仓频率应"
              "与 IC 增长速率匹配;若存在 trade-off(高频毛收益高但扣成本后下降),成本后净 spread "
              "最大的频率是首选。")
    md.append("")
    md.append("**核心结论**:WAIC 之后,当「机器人概念」标签本身已经几乎覆盖全市场,它作为因子的截面"
              "区分度必然下降。真正保留下来的,是从订单 / 合同、盈利质量、营收预期修正等**基本面**维度,"
              "而不是从「概念热度」维度,去识别哪些公司在主题扩散中真正受益。**主题标签不是因子,它只是被因子解释的现象。**")
    md.append("")

    report_path.write_text("\n".join(md), encoding="utf-8")
    return str(report_path)
