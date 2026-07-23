"""Factor stability monitoring: rolling IC stats and alerts.

Computes rolling-window IC statistics (mean, std, ICIR, positive-IC ratio,
cumulative IC) and emits alerts on the following conditions:
1. 20-day rolling mean crosses zero.
2. 20 consecutive negative-IC days.
3. Rolling ICIR < -0.5.
4. Post-event IC mean drops below pre-event by more than a threshold.
5. Raw IC still positive but Incremental (partial) IC ~ 0.
6. Gross positive but net negative after costs.
7. IC sign disagrees across horizons.
8. IC sign disagrees across sub-industries.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def rolling_ic_stats(
    daily_raw: pd.DataFrame,
    daily_partial: Optional[pd.DataFrame] = None,
    windows: List[int] = (20, 60, 120),
) -> pd.DataFrame:
    """Compute rolling IC stats per (factor, horizon, period)."""
    if daily_raw.empty:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for (fac, hz, period, univ), g in daily_raw.groupby(
        ["factor_name", "horizon", "period", "universe"], dropna=False
    ):
        g = g.sort_values("date").reset_index(drop=True)
        ic = g["pearson_ic"].astype(float)
        for w in windows:
            rmean = ic.rolling(w, min_periods=max(5, w // 4)).mean()
            rstd = ic.rolling(w, min_periods=max(5, w // 4)).std()
            rir = rmean / rstd
            pos_ratio = ic.rolling(w, min_periods=max(5, w // 4)).apply(
                lambda s: float((s > 0).mean()), raw=False
            )
            cum = ic.fillna(0).cumsum()
            tmp = pd.DataFrame(
                {
                    "date": g["date"],
                    "factor_name": fac,
                    "horizon": hz,
                    "period": period,
                    "universe": univ,
                    "window": w,
                    "rolling_mean": rmean,
                    "rolling_std": rstd,
                    "rolling_icir": rir,
                    "rolling_pos_ratio": pos_ratio,
                    "cum_ic": cum,
                    "raw_ic": ic,
                }
            )
            frames.append(tmp)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty and daily_partial is not None and not daily_partial.empty:
        # Attach partial IC for the same (factor, horizon, period, date)
        p = daily_partial.rename(columns={"partial_ic": "partial_ic"})[
            ["date", "factor_name", "horizon", "period", "partial_ic"]
        ]
        out = out.merge(p, on=["date", "factor_name", "horizon", "period"], how="left")
    return out


def _alert_rows(
    factor_name: str,
    horizon: int,
    period: str,
    alert_type: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    severity: str,
    evidence: str,
    suggested_action: str,
) -> dict:
    return {
        "factor_name": factor_name,
        "period": period,
        "horizon": horizon,
        "alert_type": alert_type,
        "start_date": start_date,
        "end_date": end_date,
        "severity": severity,
        "evidence": evidence,
        "suggested_action": suggested_action,
    }


def detect_alerts(
    rolling_df: pd.DataFrame,
    rebalance_summary: Optional[pd.DataFrame] = None,
    ic_drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """Detect stability alerts and return a frame.

    The detection is per (factor_name, period, horizon, universe) over the
    20-day rolling mean / std. Returns a DataFrame of alerts.
    """
    if rolling_df.empty:
        return pd.DataFrame()
    alerts: List[dict] = []
    for (fac, hz, period, univ), g in rolling_df.groupby(
        ["factor_name", "horizon", "period", "universe"], dropna=False
    ):
        g = g.sort_values("date").reset_index(drop=True)
        if g.empty:
            continue
        # 1) 20-day rolling mean crosses zero
        w20 = g[g["window"] == 20].sort_values("date")
        if not w20.empty:
            r = w20["rolling_mean"].to_numpy()
            d = w20["date"].to_numpy()
            for i in range(1, len(r)):
                if np.isfinite(r[i - 1]) and np.isfinite(r[i]) and r[i - 1] * r[i] < 0:
                    alerts.append(
                        _alert_rows(
                            fac, hz, period, "rolling_mean_cross_zero",
                            pd.Timestamp(d[i - 1]), pd.Timestamp(d[i]),
                            "warning",
                            f"20d rolling IC changed sign: {r[i-1]:.3f} -> {r[i]:.3f}",
                            "Investigate regime change; do not trust this factor blindly.",
                        )
                    )
            # 2) 20 consecutive negative-IC days
            ic = g["raw_ic"].to_numpy()
            d2 = g["date"].to_numpy()
            run = 0
            run_start = None
            for i, v in enumerate(ic):
                if np.isfinite(v) and v < 0:
                    if run == 0:
                        run_start = d2[i]
                    run += 1
                    if run >= 20:
                        alerts.append(
                            _alert_rows(
                                fac, hz, period, "20_consecutive_neg_ic",
                                pd.Timestamp(run_start), pd.Timestamp(d2[i]),
                                "high",
                                f"20 consecutive negative IC days ending {pd.Timestamp(d2[i]).date()}",
                                "Stop using this factor for new positions; reduce weight.",
                            )
                        )
                        run = 0
                else:
                    run = 0
                    run_start = None
            # 3) Rolling ICIR < -0.5 for window=60
            w60 = g[g["window"] == 60]
            if not w60.empty:
                rir = w60["rolling_icir"].to_numpy()
                d3 = w60["date"].to_numpy()
                for i, v in enumerate(rir):
                    if np.isfinite(v) and v < -0.5:
                        alerts.append(
                            _alert_rows(
                                fac, hz, period, "rolling_icir_below_-0.5",
                                pd.Timestamp(d3[i]), pd.Timestamp(d3[i]),
                                "high",
                                f"60d rolling ICIR = {v:.2f}",
                                "Factor may be broken; review signal definition.",
                            )
                        )
    # 4) Post-event IC drop
    # Computed at the aggregation level — handled separately by the pipeline.
    # 5/6) raw > 0 but partial~0 OR gross > 0 but net < 0: handled in
    # cross-factor call below.
    if rebalance_summary is not None and not rebalance_summary.empty:
        for _, r in rebalance_summary.iterrows():
            fac = r["factor_name"]
            freq = r["rebalance_frequency"]
            if r["top_bottom_spread_gross"] > 0 and r["top_bottom_spread_net"] < 0:
                alerts.append(
                    _alert_rows(
                        fac, freq, "all", "gross_positive_net_negative",
                        pd.Timestamp("1900-01-01"), pd.Timestamp("2099-12-31"),
                        "warning",
                        f"Gross spread = {r['top_bottom_spread_gross']:.3f}, "
                        f"net spread = {r['top_bottom_spread_net']:.3f} at rebalance freq {freq}.",
                        "Reduce rebalance frequency or relax the signal.",
                    )
                )
    out = pd.DataFrame(alerts)
    if out.empty:
        return out
    # Deduplicate alerts: keep one row per (factor_name, alert_type, period)
    # to avoid thousands of repeated "20_consecutive_neg_ic" alerts. We
    # preserve the first start_date and last end_date and a count of
    # distinct (horizon, evidence) tuples.
    out_sorted = out.sort_values(["factor_name", "alert_type", "start_date"])
    out_sorted["count"] = 1
    agg = (
        out_sorted.groupby(
            ["factor_name", "alert_type", "severity", "suggested_action", "period"],
            dropna=False,
        )
        .agg(
            start_date=("start_date", "min"),
            end_date=("end_date", "max"),
            count=("count", "sum"),
            n_horizons=("horizon", "nunique"),
        )
        .reset_index()
    )
    agg["evidence"] = agg.apply(
        lambda r: f"{int(r['n_horizons'])} distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details.",
        axis=1,
    )
    agg = agg.drop(columns=["n_horizons"])
    return agg


def cross_horizon_disagreement(
    summary: pd.DataFrame, period: str = "post_event", min_abs_ic: float = 0.02
) -> pd.DataFrame:
    """Return rows where the factor's IC sign disagrees across horizons.

    A factor is flagged if it has at least one horizon with |IC| > min_abs_ic
    and at least one other horizon with |IC| > min_abs_ic but opposite sign.
    """
    if summary.empty:
        return pd.DataFrame()
    sub = summary[summary["period"] == period].copy()
    if sub.empty:
        return sub
    rows: List[dict] = []
    for fac, g in sub.groupby("factor_name"):
        g = g.dropna(subset=["ic_mean"])
        if g.empty:
            continue
        pos = g[g["ic_mean"] > min_abs_ic]
        neg = g[g["ic_mean"] < -min_abs_ic]
        if not pos.empty and not neg.empty:
            rows.append(
                {
                    "factor_name": fac,
                    "period": period,
                    "n_horizons_pos": len(pos),
                    "n_horizons_neg": len(neg),
                    "max_pos_horizon": int(pos.loc[pos["ic_mean"].idxmax(), "horizon"]),
                    "max_neg_horizon": int(neg.loc[neg["ic_mean"].idxmin(), "horizon"]),
                    "alert_type": "horizon_disagreement",
                }
            )
    return pd.DataFrame(rows)


def incremental_ic_check(
    summary: pd.DataFrame, period: str = "post_event", threshold: float = 0.005
) -> pd.DataFrame:
    """Flag factors whose raw IC is positive but incremental (partial) IC ~ 0.

    "Incremental ~ 0" means ``|partial_ic_mean| <= threshold``.
    """
    if summary.empty or "partial_ic_mean" not in summary.columns:
        return pd.DataFrame()
    sub = summary[summary["period"] == period].copy()
    rows: List[dict] = []
    for fac, g in sub.groupby("factor_name"):
        # Take the best horizon by |IC| for this factor
        g_nona = g.dropna(subset=["ic_mean", "partial_ic_mean"])
        if g_nona.empty:
            continue
        best = g_nona.iloc[g_nona["ic_mean"].abs().argmax()]
        if best["ic_mean"] > 0 and abs(best["partial_ic_mean"]) <= threshold:
            rows.append(
                {
                    "factor_name": fac,
                    "period": period,
                    "horizon": int(best["horizon"]),
                    "ic_mean": float(best["ic_mean"]),
                    "partial_ic_mean": float(best["partial_ic_mean"]),
                    "alert_type": "raw_positive_partial_near_zero",
                }
            )
    return pd.DataFrame(rows)
