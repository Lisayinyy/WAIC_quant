"""Multi-horizon analysis and IC decay / half-life estimation.

IC decay is computed by sweeping horizons in the daily IC table. The
half-life is defined as the *earliest* horizon at which the absolute IC
falls to half of its peak. If the curve is non-monotonic or never decays
to half, the half-life is NaN.

We also smooth the curve with a simple monotone-aware (Hampel-like)
filter so that a single noisy spike does not produce a misleading
half-life, but we always preserve the raw values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DecayResult:
    factor_name: str
    period: str
    universe: str
    peak_horizon: int
    peak_abs_ic: float
    half_life_horizon: float
    recommended_rebalance_frequency: int
    curve_monotonicity: str
    reliability_note: str


def _hampel_smooth(y: np.ndarray, window: int = 3, n_sigmas: float = 3.0) -> np.ndarray:
    """Robust local-median smoother; replaces outliers with the local median."""
    out = y.copy().astype(float)
    half = window // 2
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        local = y[lo:hi]
        med = np.nanmedian(local)
        mad = np.nanmedian(np.abs(local - med)) + 1e-8
        if np.isfinite(y[i]) and abs(y[i] - med) > n_sigmas * mad:
            out[i] = med
    return out


def _monotonicity_label(y: np.ndarray) -> str:
    """Categorize the curve as 'monotonic_up', 'monotonic_down', 'peak_then_decay', 'flat', 'non_monotonic'."""
    y = np.asarray(y, dtype=float)
    if np.all(np.isnan(y)):
        return "flat"
    abs_y = np.abs(y)
    if np.nanstd(abs_y) < 1e-4:
        return "flat"
    peak_idx = int(np.nanargmax(abs_y))
    # Look at left and right of peak
    if peak_idx == 0:
        # Strictly descending from start
        right = abs_y[1:]
        if np.all(np.diff(right) <= 1e-8):
            return "monotonic_down"
        return "non_monotonic"
    if peak_idx == len(abs_y) - 1:
        left = abs_y[:-1]
        if np.all(np.diff(left) >= -1e-8):
            return "monotonic_up"
        return "non_monotonic"
    left = abs_y[: peak_idx + 1]
    right = abs_y[peak_idx:]
    if np.all(np.diff(left) >= -1e-8) and np.all(np.diff(right) <= 1e-8):
        return "peak_then_decay"
    return "non_monotonic"


def _estimate_half_life(horizons: np.ndarray, abs_ic: np.ndarray) -> Tuple[float, str]:
    """Estimate the half-life of the IC decay curve.

    The half-life is the *earliest* horizon at which |IC| falls to half the
    peak value. The peak is defined as the maximum |IC| on the curve.
    """
    if len(horizons) < 2 or np.all(np.isnan(abs_ic)):
        return float("nan"), "insufficient_data"
    peak = float(np.nanmax(abs_ic))
    if peak < 1e-4:
        return float("nan"), "ic_too_weak"
    target = peak / 2.0
    # Walk forward from the peak
    peak_idx = int(np.nanargmax(abs_ic))
    for i in range(peak_idx, len(horizons)):
        if not np.isnan(abs_ic[i]) and abs_ic[i] <= target:
            # Linear interpolation between i-1 and i
            if i == 0:
                return float(horizons[i]), "at_peak"
            y_prev, y_cur = abs_ic[i - 1], abs_ic[i]
            h_prev, h_cur = horizons[i - 1], horizons[i]
            if y_prev == y_cur:
                return float(h_cur), "at_target"
            # Solve linear interpolation: target = y_prev + (y_cur - y_prev) * t
            t = (target - y_prev) / (y_cur - y_prev)
            return float(h_prev + t * (h_cur - h_prev)), "ok"
    return float("nan"), "never_decayed"


def _estimate_plateau_horizon(horizons: np.ndarray, abs_ic: np.ndarray) -> float:
    """If the curve never decays, return the horizon at which |IC| is still
    at >= 70% of the peak. This is the *effective horizon* under a 'no-decay'
    regime. NaN if peak is too weak to be meaningful.
    """
    if len(horizons) < 2 or np.all(np.isnan(abs_ic)):
        return float("nan")
    peak = float(np.nanmax(abs_ic))
    if peak < 1e-4:
        return float("nan")
    peak_idx = int(np.nanargmax(abs_ic))
    # Walk forward from peak until |IC| < 0.7 * peak (or end of sample).
    threshold = 0.7 * peak
    for i in range(peak_idx + 1, len(horizons)):
        if not np.isnan(abs_ic[i]) and abs_ic[i] < threshold:
            return float(horizons[i])
    # If still above 70% of peak all the way to the last horizon, the entire
    # sample is the "plateau" — return the last horizon we observed.
    return float(horizons[-1])


def compute_decay_curve(
    daily_raw: pd.DataFrame,
    horizons: List[int],
    period: str = "post_event",
    universe: str = "all",
) -> pd.DataFrame:
    """Compute the IC decay curve for each factor.

    Returns
    -------
    pd.DataFrame
        Columns: factor_name, period, universe, horizon, ic_mean, abs_ic_mean,
        ic_rank_mean, n_days, smoothed_abs_ic.
    """
    if daily_raw.empty:
        return pd.DataFrame()
    sub = daily_raw[(daily_raw["period"] == period)]
    records: List[dict] = []
    for fac, g in sub.groupby("factor_name"):
        for hz in horizons:
            gd = g[g["horizon"] == hz]
            ic = gd["pearson_ic"].dropna()
            ric = gd["rank_ic"].dropna()
            if len(ic) < 2:
                records.append(
                    {
                        "factor_name": fac,
                        "period": period,
                        "universe": universe,
                        "horizon": hz,
                        "ic_mean": np.nan,
                        "abs_ic_mean": np.nan,
                        "rank_ic_mean": np.nan,
                        "n_days": int(len(ic)),
                        "smoothed_abs_ic": np.nan,
                    }
                )
                continue
            records.append(
                {
                    "factor_name": fac,
                    "period": period,
                    "universe": universe,
                    "horizon": hz,
                    "ic_mean": float(ic.mean()),
                    "abs_ic_mean": float(ic.mean()),  # signed IC mean
                    "rank_ic_mean": float(ric.mean()) if len(ric) else np.nan,
                    "n_days": int(len(ic)),
                    "smoothed_abs_ic": np.nan,  # filled later
                }
            )
    out = pd.DataFrame(records)
    # Smooth |IC| per factor for visualization
    if not out.empty:
        for fac, g in out.groupby("factor_name"):
            idx = g.index
            y = g["abs_ic_mean"].to_numpy(dtype=float)
            sm = _hampel_smooth(y, window=3)
            out.loc[idx, "smoothed_abs_ic"] = np.abs(sm)
    return out


def compute_half_life(
    decay_df: pd.DataFrame,
    rebalance_frequencies: List[int] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Estimate half-life & recommended rebalance frequency for each factor.

    Returns
    -------
    pd.DataFrame
        Columns: factor_name, period, universe, peak_horizon, peak_abs_ic,
        half_life_horizon, plateau_horizon, recommended_rebalance_frequency,
        curve_monotonicity, reliability_note.
    """
    if decay_df.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    for (fac, period, univ), g in decay_df.groupby(["factor_name", "period", "universe"]):
        g_sorted = g.sort_values("horizon")
        horizons = g_sorted["horizon"].to_numpy()
        ic = g_sorted["ic_mean"].to_numpy(dtype=float)
        abs_ic = np.abs(ic)
        peak_idx = int(np.nanargmax(abs_ic)) if not np.all(np.isnan(abs_ic)) else 0
        peak_h = int(horizons[peak_idx]) if len(horizons) else 0
        peak_v = float(abs_ic[peak_idx]) if len(abs_ic) else float("nan")
        hl, note = _estimate_half_life(horizons, abs_ic)
        plateau = _estimate_plateau_horizon(horizons, abs_ic)
        mono = _monotonicity_label(ic)
        # Recommend rebalance.
        if np.isnan(hl) or hl <= 0:
            # Curve never decays to half peak within sample: use plateau horizon
            # (the horizon at which |IC| is still at >= 70% of peak). This is a
            # much better proxy than 'rebalance=1' when the IC keeps climbing.
            if not np.isnan(plateau) and plateau > 0:
                candidates = [f for f in rebalance_frequencies if f <= plateau]
                rec = max(candidates) if candidates else rebalance_frequencies[-1]
                note_full = f"never_decayed_within_sample; plateau_horizon={int(plateau)}; recommend rebalance={rec} (IC still strong)"
            else:
                rec = 1
                note_full = f"{note}; recommend rebalance=1 (low confidence)"
        else:
            # Pick the largest rebalance frequency that is <= half_life
            candidates = [f for f in rebalance_frequencies if f <= hl]
            rec = max(candidates) if candidates else 1
            note_full = note
        rows.append(
            {
                "factor_name": fac,
                "period": period,
                "universe": univ,
                "peak_horizon": peak_h,
                "peak_abs_ic": peak_v,
                "half_life_horizon": hl,
                "plateau_horizon": plateau,
                "recommended_rebalance_frequency": rec,
                "curve_monotonicity": mono,
                "reliability_note": note_full,
            }
        )
    return pd.DataFrame(rows)


def best_horizon_per_factor(
    summary: pd.DataFrame, period: str = "post_event"
) -> pd.DataFrame:
    """For each factor in the given period, pick the horizon with the largest |IC|."""
    if summary.empty:
        return pd.DataFrame()
    sub = summary[summary["period"] == period].copy()
    if sub.empty:
        return sub
    sub["abs_ic"] = sub["ic_mean"].abs()
    idx = sub.groupby("factor_name")["abs_ic"].idxmax()
    return sub.loc[idx].sort_values("abs_ic", ascending=False)
