"""End-to-end pipeline: tie everything together for a single runnable entry point.

Usage
-----
    from src.pipeline import run_pipeline
    summary = run_pipeline(
        prices=prices_df,
        factors=factors_df,
        event_date="2024-07-15",
        output_dir="outputs",
    )
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_loader import (
    get_universe_mask,
    normalize_dates,
    restrict_to_universe,
    split_periods,
    validate_factors,
    validate_prices,
)
from .forward_returns import compute_forward_returns
from .horizon import (
    best_horizon_per_factor,
    compute_decay_curve,
    compute_half_life,
)
from .ic_analysis import aggregate_ic, daily_ic_table, daily_partial_ic, run_ic_pipeline
from .portfolio import backtest_long_short, build_themes_portfolios
from .rebalance import (
    monotonicity_check,
    rebalance_summary,
    run_rebalance_analysis,
)
from .reporting import write_report
from .stability import (
    cross_horizon_disagreement,
    detect_alerts,
    incremental_ic_check,
    rolling_ic_stats,
)


DEFAULT_HORIZONS = [1, 2, 3, 5, 10, 20, 40, 60]
DEFAULT_REBALANCE_FREQS = [1, 5, 10, 20]
DEFAULT_MIN_CROSS_SECTION = 20


def _build_controls_table(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Build the per-day per-asset control variable table.

    Pulls from prices (theme_exposure, sub_industry, log_market_cap) and from
    factors (beta, liquidity, price_momentum).
    """
    out = prices[["date", "asset"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    if "theme_exposure" in prices.columns:
        out = out.merge(prices[["date", "asset", "theme_exposure"]], on=["date", "asset"], how="left")
    if "sub_industry" in prices.columns:
        out = out.merge(prices[["date", "asset", "sub_industry"]], on=["date", "asset"], how="left")
    if "industry" in prices.columns:
        out = out.merge(prices[["date", "asset", "industry"]], on=["date", "asset"], how="left")
    if "market_cap" in prices.columns:
        out["log_market_cap"] = np.log(prices["market_cap"].astype(float).fillna(0) + 1e-9)
        out["log_market_cap"] = out["log_market_cap"].replace([np.inf, -np.inf], np.nan)

    # Pull auxiliary factors if available
    for fac_name in ["beta", "liquidity", "price_momentum"]:
        sub = factors[factors["factor_name"] == fac_name][["date", "asset", "factor_value"]].copy()
        if sub.empty:
            continue
        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
        sub = sub.rename(columns={"factor_value": fac_name})
        out = out.merge(sub, on=["date", "asset"], how="left")
    return out


def _theme_spread_stats(prices: pd.DataFrame) -> dict:
    """Compute theme diffusion statistics."""
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    n_total = df["asset"].nunique()
    out: Dict[str, float] = {
        "n_assets_total": int(n_total),
        "n_core_robot": 0,
        "core_share": 0.0,
        "n_diffused": 0,
        "diffused_share": 0.0,
    }
    if "is_robot_stock" in df.columns:
        core = df.loc[df["is_robot_stock"], "asset"].nunique()
        out["n_core_robot"] = int(core)
        out["core_share"] = core / max(n_total, 1)
    if "theme_exposure" in df.columns:
        diffused = df.loc[df["theme_exposure"] > 0, "asset"].nunique()
        out["n_diffused"] = int(diffused)
        out["diffused_share"] = diffused / max(n_total, 1)
        # post-event theme exposure distribution
        out["theme_exposure_pre_mean"] = float(df["theme_exposure"].mean())
        out["theme_exposure_pre_std"] = float(df["theme_exposure"].std())
    return out


def _best_horizon_with_partial(summary: pd.DataFrame, period: str = "post_event") -> pd.DataFrame:
    """Return per-factor best horizon with both raw and partial IC."""
    if summary.empty:
        return pd.DataFrame()
    sub = summary[summary["period"] == period].copy()
    if sub.empty:
        return sub
    rows = []
    for fac, g in sub.groupby("factor_name"):
        g_nona = g.dropna(subset=["ic_mean"])
        if g_nona.empty:
            continue
        best = g_nona.iloc[g_nona["ic_mean"].abs().argmax()]
        rows.append(best)
    return pd.DataFrame(rows)


def run_pipeline(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    event_date: Optional[pd.Timestamp] = None,
    output_dir: str = "outputs",
    horizons: List[int] = None,
    rebalance_frequencies: List[int] = None,
    quantiles: int = 5,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    transaction_cost_bps: float = 10.0,
    post_event_window: int = 120,
    data_note: str = "synthetic test data",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run the end-to-end factor IC analysis pipeline.

    Parameters
    ----------
    prices
        Long-format price DataFrame.
    factors
        Long-format factor DataFrame.
    event_date
        WAIC / research event date. If None, no pre/post split is performed.
    output_dir
        Where to write outputs.
    horizons
        Forward-return horizons.
    rebalance_frequencies
        Rebalance frequencies to compare.
    quantiles
        Number of quantile groups.
    min_cross_section
        Minimum cross-section size per day.
    transaction_cost_bps
        One-way transaction cost in basis points.
    post_event_window
        Number of post-event business days to focus on.
    data_note
        Free-form note embedded in the report.
    verbose
        Whether to print progress.
    """
    if horizons is None:
        horizons = list(DEFAULT_HORIZONS)
    if rebalance_frequencies is None:
        rebalance_frequencies = list(DEFAULT_REBALANCE_FREQS)

    # ---- Validate ----
    validate_prices(prices)
    validate_factors(factors)
    prices = normalize_dates(prices)
    factors = normalize_dates(factors)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ---- Theme spread stats ----
    spread_stats = _theme_spread_stats(prices)
    if verbose:
        print(f"[spread] {spread_stats}")

    # ---- Forward returns ----
    fr = compute_forward_returns(prices, horizons=horizons)
    fr.to_csv(out_dir / "forward_returns.csv", index=False)

    # ---- Controls table ----
    controls = _build_controls_table(prices, factors)

    # ---- Universe: theme_diffused (all stocks with theme_exposure>0) and theme_core ----
    universe_core = get_universe_mask(prices, mode="theme_core")
    universe_dif = get_universe_mask(prices, mode="theme_diffused")

    # ---- Run IC pipeline for both universes ----
    all_daily_raw = []
    all_daily_partial = []
    all_summary = []
    for name, univ in [("theme_core", universe_core), ("theme_diffused", universe_dif)]:
        res = run_ic_pipeline(
            factors_long=factors,
            forward_returns=fr,
            controls_table=controls,
            horizons=horizons,
            event_date=event_date,
            post_event_window=post_event_window,
            min_cross_section=min_cross_section,
            universe=univ,
        )
        if not res["daily_raw"].empty:
            res["daily_raw"]["universe"] = name
            all_daily_raw.append(res["daily_raw"])
        if not res["daily_partial"].empty:
            res["daily_partial"]["universe"] = name
            all_daily_partial.append(res["daily_partial"])
        if not res["summary"].empty:
            res["summary"]["universe"] = name
            all_summary.append(res["summary"])

    if all_daily_raw:
        daily_raw = pd.concat(all_daily_raw, ignore_index=True)
    else:
        daily_raw = pd.DataFrame()
    if all_daily_partial:
        daily_partial = pd.concat(all_daily_partial, ignore_index=True)
    else:
        daily_partial = pd.DataFrame()
    if all_summary:
        summary = pd.concat(all_summary, ignore_index=True)
    else:
        summary = pd.DataFrame()

    # Save daily IC, summary, partial IC
    if not daily_raw.empty:
        daily_raw.to_csv(out_dir / "daily_ic.csv", index=False)
    if not daily_partial.empty:
        daily_partial.to_csv(out_dir / "daily_ic_partial.csv", index=False)
    if not summary.empty:
        summary.to_csv(out_dir / "factor_horizon_summary.csv", index=False)

    # ---- Partial + Incremental IC table ----
    if not summary.empty and "partial_ic_mean" in summary.columns:
        cols = [
            "factor_name", "horizon", "period", "universe",
            "ic_mean", "icir", "p_value",
            "partial_ic_mean", "partial_icir", "partial_p_value",
            "n_days",
        ]
        if "partial_t_stat" in summary.columns:
            cols.append("partial_t_stat")
        partial_inc = summary[cols].copy()
        partial_inc["incremental_ic"] = partial_inc["partial_ic_mean"]
        partial_inc.to_csv(out_dir / "partial_incremental_ic.csv", index=False)
    else:
        partial_inc = pd.DataFrame()

    # ---- Decay / half-life ----
    if not daily_raw.empty:
        decay = compute_decay_curve(
            daily_raw[daily_raw["universe"] == "theme_diffused"],
            horizons=horizons,
            period="post_event",
        )
        decay.to_csv(out_dir / "ic_decay_curve.csv", index=False)
        half_life = compute_half_life(decay, rebalance_frequencies=rebalance_frequencies)
        half_life.to_csv(out_dir / "factor_half_life.csv", index=False)
    else:
        decay = pd.DataFrame()
        half_life = pd.DataFrame()

    # ---- Rebalance analysis ----
    if not daily_raw.empty:
        # Build per-factor rebalance table on theme_diffused
        target = daily_raw[(daily_raw["universe"] == "theme_diffused") & (daily_raw["period"].isin(["post_event", "all"]))]
        per_factor_rebal: Dict[str, pd.DataFrame] = {}
        quantile_dfs: List[pd.DataFrame] = []
        for fac in sorted(target["factor_name"].unique()):
            fv_long = factors[factors["factor_name"] == fac][["date", "asset", "factor_value"]].copy()
            fv_long["date"] = pd.to_datetime(fv_long["date"]).dt.normalize()
            merged = fv_long.merge(fr, on=["date", "asset"], how="inner")
            merged = restrict_to_universe(merged, universe_dif)
            for hz in [1, 5, 10, 20]:
                r = run_rebalance_analysis(
                    merged,
                    horizon=hz,
                    rebalance_frequencies=rebalance_frequencies,
                    quantiles=quantiles,
                    transaction_cost_bps=transaction_cost_bps,
                )
                if r.empty:
                    continue
                r["factor_name"] = fac
                r["horizon"] = hz
                quantile_dfs.append(r)
            # For the headline rebalance summary, use h=5 (mid horizon)
            r5 = run_rebalance_analysis(
                merged,
                horizon=5,
                rebalance_frequencies=rebalance_frequencies,
                quantiles=quantiles,
                transaction_cost_bps=transaction_cost_bps,
            )
            if not r5.empty:
                per_factor_rebal[fac] = r5
        rebal_sum = rebalance_summary(per_factor_rebal, transaction_cost_bps=transaction_cost_bps)
        rebal_sum.to_csv(out_dir / "rebalance_comparison.csv", index=False)
        if quantile_dfs:
            quantile_df = pd.concat(quantile_dfs, ignore_index=True)
        else:
            quantile_df = pd.DataFrame()
        if not quantile_df.empty:
            quantile_df.to_csv(out_dir / "quantile_returns.csv", index=False)
    else:
        rebal_sum = pd.DataFrame()
        quantile_df = pd.DataFrame()

    # ---- Rolling IC + alerts ----
    if not daily_raw.empty:
        # Use the theme_diffused universe for stability
        rolling = rolling_ic_stats(
            daily_raw[daily_raw["universe"] == "theme_diffused"],
            daily_partial[daily_partial["universe"] == "theme_diffused"] if not daily_partial.empty else None,
            windows=[20, 60, 120],
        )
        if not rolling.empty:
            rolling.to_csv(out_dir / "rolling_ic.csv", index=False)
        alerts_raw = detect_alerts(rolling, rebalance_summary=rebal_sum)
        if not alerts_raw.empty:
            # Save the deduplicated summary
            alerts_raw.to_csv(out_dir / "stability_alerts.csv", index=False)
        # Additional alerts
        xh = cross_horizon_disagreement(summary[summary["universe"] == "theme_diffused"], period="post_event")
        ic_inc = incremental_ic_check(summary[summary["universe"] == "theme_diffused"], period="post_event")
        extra_alerts = pd.concat([xh, ic_inc], ignore_index=True, sort=False) if not (xh.empty and ic_inc.empty) else pd.DataFrame()
        if not extra_alerts.empty:
            existing = pd.read_csv(out_dir / "stability_alerts.csv") if (out_dir / "stability_alerts.csv").exists() else pd.DataFrame()
            combined = pd.concat([existing, extra_alerts], ignore_index=True, sort=False)
            combined.to_csv(out_dir / "stability_alerts.csv", index=False)
        alerts = alerts_raw if not alerts_raw.empty else (pd.read_csv(out_dir / "stability_alerts.csv") if (out_dir / "stability_alerts.csv").exists() else pd.DataFrame())
    else:
        rolling = pd.DataFrame()
        alerts = pd.DataFrame()

    # ---- Portfolio construction ----
    port_records: List[dict] = []
    if not fr.empty:
        # Portfolio A: theme exposure
        if "theme_exposure" in prices.columns:
            res = backtest_long_short(
                fr,
                "theme_exposure",
                pd.to_numeric(prices["theme_exposure"], errors="coerce"),
                fwd_ret_col="fwd_ret_1",
                rebalance_freq=5,
                transaction_cost_bps=transaction_cost_bps,
            )
            res["portfolio"] = "A_theme_exposure"
            port_records.append(res)
        # Portfolio B: best single factor by |IC| at h=5
        if not summary.empty:
            s_post = summary[(summary["period"] == "post_event") & (summary["universe"] == "theme_diffused")]
            best_fac_row = None
            for fac, g in s_post.groupby("factor_name"):
                g5 = g[g["horizon"] == 5]
                if g5.empty:
                    continue
                if best_fac_row is None or abs(float(g5["ic_mean"].iloc[0])) > abs(float(best_fac_row["ic_mean"])):
                    best_fac_row = g5.iloc[0]
                    best_fac_row["factor_name"] = fac
            if best_fac_row is not None:
                fac_name = best_fac_row["factor_name"]
                fv_long = factors[factors["factor_name"] == fac_name][["date", "asset", "factor_value"]]
                fv_long = fv_long.merge(fr[["date", "asset", "fwd_ret_1"]], on=["date", "asset"], how="inner")
                if "fwd_ret_1" not in fv_long.columns:
                    print(f"DEBUG: fac_name={fac_name}, fv_long cols={fv_long.columns.tolist()}")
                res = backtest_long_short(
                    fv_long,
                    fac_name,
                    pd.to_numeric(fv_long["factor_value"], errors="coerce"),
                    fwd_ret_col="fwd_ret_1",
                    rebalance_freq=5,
                    transaction_cost_bps=transaction_cost_bps,
                )
                res["portfolio"] = "B_single_best_factor"
                port_records.append(res)
        # Portfolio C: incremental factor blend — equal-weight top 3 by |partial IC|
        if not summary.empty and "partial_ic_mean" in summary.columns:
            top_inc = (
                summary[(summary["period"] == "post_event") & (summary["universe"] == "theme_diffused")]
                .dropna(subset=["partial_ic_mean"])
                .sort_values("partial_ic_mean", key=lambda s: s.abs(), ascending=False)
                .head(3)
            )
            for _, row in top_inc.iterrows():
                fac_name = row["factor_name"]
                fv_long = factors[factors["factor_name"] == fac_name][["date", "asset", "factor_value"]]
                fv_long = fv_long.merge(fr[["date", "asset", "fwd_ret_1"]], on=["date", "asset"], how="inner")
                res = backtest_long_short(
                    fv_long,
                    fac_name + "_inc",
                    pd.to_numeric(fv_long["factor_value"], errors="coerce"),
                    fwd_ret_col="fwd_ret_1",
                    rebalance_freq=5,
                    transaction_cost_bps=transaction_cost_bps,
                )
                res["portfolio"] = "C_incremental_blend"
                port_records.append(res)
        # Portfolio D: multi-horizon (short=1, mid=10, long=40)
        for hz, freq in [(1, 1), (10, 5), (40, 20)]:
            if not summary.empty:
                fac_row = (
                    summary[(summary["period"] == "post_event") & (summary["universe"] == "theme_diffused") & (summary["horizon"] == hz)]
                    .dropna(subset=["ic_mean"])
                    .sort_values("ic_mean", key=lambda s: s.abs(), ascending=False)
                )
                if fac_row.empty:
                    continue
                fac_name = fac_row.iloc[0]["factor_name"]
                fv_long = factors[factors["factor_name"] == fac_name][["date", "asset", "factor_value"]]
                fv_long = fv_long.merge(fr[["date", "asset", "fwd_ret_1"]], on=["date", "asset"], how="inner")
                res = backtest_long_short(
                    fv_long,
                    fac_name + "_hz" + str(hz),
                    pd.to_numeric(fv_long["factor_value"], errors="coerce"),
                    fwd_ret_col="fwd_ret_1",
                    rebalance_freq=freq,
                    transaction_cost_bps=transaction_cost_bps,
                )
                res["portfolio"] = f"D_multi_horizon_h{hz}"
                port_records.append(res)
    port_df = pd.DataFrame(port_records)
    if not port_df.empty:
        port_df.to_csv(out_dir / "portfolio_summary.csv", index=False)

    # ---- Best-horizon per factor (for reporting) ----
    factor_horizon_summary = _best_horizon_with_partial(
        summary[summary["universe"] == "theme_diffused"] if not summary.empty else summary,
        period="post_event",
    )
    if not factor_horizon_summary.empty:
        # Save best-horizon summary as a separate file; do NOT overwrite the full summary
        factor_horizon_summary.to_csv(out_dir / "factor_horizon_summary_best.csv", index=False)

    # ---- Report ----
    report_path = write_report(
        output_dir=str(out_dir),
        daily_raw=daily_raw,
        daily_partial=daily_partial,
        summary=summary,
        decay=decay,
        half_life=half_life,
        rebal_summary=rebal_sum,
        quantile_df=quantile_df,
        rolling_df=rolling,
        alerts=alerts if not alerts.empty else (pd.read_csv(out_dir / "stability_alerts.csv") if (out_dir / "stability_alerts.csv").exists() else pd.DataFrame()),
        portfolio_df=port_df,
        factor_horizon_summary=factor_horizon_summary,
        partial_inc=partial_inc,
        theme_spread_stats=spread_stats,
        event_date=event_date,
        horizons=horizons,
        data_note=data_note,
    )

    return {
        "daily_raw": daily_raw,
        "daily_partial": daily_partial,
        "summary": summary,
        "decay": decay,
        "half_life": half_life,
        "rebalance_summary": rebal_sum,
        "quantile_df": quantile_df,
        "rolling": rolling,
        "alerts": alerts,
        "portfolio": port_df,
        "factor_horizon_summary": factor_horizon_summary,
        "partial_inc": partial_inc,
        "report_path": report_path,
    }
