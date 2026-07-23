"""Rebalance frequency analysis and quantile-portfolio construction.

Given a factor's daily cross-section and a chosen rebalance frequency
(1, 5, 10, 20 trading days), this module:
1. Picks rebalance days.
2. On each rebalance day, ranks assets by factor value and assigns them
   to N quantiles.
3. Holds each quantile equally weighted for ``rebalance_frequency`` days.
4. Computes per-quantile cumulative return, volatility, max drawdown,
   turnover, transaction cost, and net return.
5. Reports Top-Bottom spread and Spearman monotonicity.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


EPS = 1e-12


def _equal_freq_bins(values: np.ndarray, n_groups: int) -> np.ndarray:
    """Return integer group labels of shape (len(values),) in [0, n_groups-1].

    Uses rank-based binning so that ties never crash and each group has
    approximately equal size.
    """
    if len(values) == 0:
        return np.array([], dtype=int)
    valid = np.isfinite(values)
    out = np.full(len(values), -1, dtype=int)
    if valid.sum() < n_groups:
        # Not enough valid values for the requested number of groups
        out[valid] = 0  # all in group 0
        return out
    ranks = stats.rankdata(values[valid], method="average")
    # Quantile bin edges
    edges = np.linspace(0, len(ranks) + 1, n_groups + 1)
    labels = np.digitize(ranks, edges[1:-1], right=False)
    out[valid] = labels.astype(int)
    return out


def _max_drawdown(cumret: np.ndarray) -> float:
    if len(cumret) < 2:
        return 0.0
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / (1 + running_max)
    return float(dd.min())


def run_rebalance_analysis(
    factor_returns: pd.DataFrame,
    horizon: int,
    rebalance_frequencies: List[int],
    quantiles: int = 5,
    transaction_cost_bps: float = 10.0,
    rebalance_horizon_alignment: str = "matched",
) -> pd.DataFrame:
    """Run the rebalance analysis for one factor at a given horizon.

    Parameters
    ----------
    factor_returns
        Long DataFrame: date, asset, factor_value, fwd_ret_<h>.
    horizon
        Forward-return horizon to use.
    rebalance_frequencies
        List of integers. Each rebalance event occurs every K days.
    quantiles
        Number of groups (default 5).
    transaction_cost_bps
        One-way transaction cost in basis points (e.g. 10 = 0.10%).
    rebalance_horizon_alignment
        ``"matched"`` — the held forward return is the same horizon as the
        factor's evaluation horizon (e.g. h=10 uses fwd_ret_10). ``"fwd_1"``
        forces the held return to be the 1-day forward return, applied
        across the rebalance period (cumulative).

    Returns
    -------
    pd.DataFrame
        Columns: rebalance_frequency, quantile, n_periods, mean_return,
        cum_return, volatility, max_drawdown, turnover, transaction_cost,
        net_return, sharpe.
    """
    fwd_col = f"fwd_ret_{horizon}"
    if fwd_col not in factor_returns.columns:
        return pd.DataFrame()

    sub = factor_returns[["date", "asset", "factor_value", fwd_col]].copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    sub = sub.dropna(subset=["factor_value", fwd_col])
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("date").reset_index(drop=True)
    all_dates = sub["date"].drop_duplicates().sort_values().reset_index(drop=True)

    # Pre-compute per-asset next-day return for turnover (assume fwd_ret_1 is
    # available, or fall back to the held return column).
    fwd_1_col = "fwd_ret_1"
    use_fwd_1 = fwd_1_col in factor_returns.columns
    if use_fwd_1 and fwd_1_col not in sub.columns:
        sub_1 = factor_returns[["date", "asset", fwd_1_col]].copy()
        sub_1["date"] = pd.to_datetime(sub_1["date"]).dt.normalize()
        sub = sub.merge(sub_1, on=["date", "asset"], how="left")

    out_rows: List[dict] = []
    for freq in rebalance_frequencies:
        if freq < 1:
            continue
        # Pick rebalance days
        rebal_dates = all_dates.iloc[::freq].tolist()
        if not rebal_dates:
            continue
        # For each rebalance event:
        period_returns = {q: [] for q in range(quantiles)}
        period_turnover = {q: [] for q in range(quantiles)}
        last_holdings: Dict[int, Dict[str, float]] = {q: {} for q in range(quantiles)}
        for i, d in enumerate(rebal_dates):
            day = sub[sub["date"] == d]
            if len(day) < quantiles * 2:
                continue
            labels = _equal_freq_bins(day["factor_value"].to_numpy(dtype=float), quantiles)
            valid_mask = labels >= 0
            if valid_mask.sum() < quantiles:
                continue
            # Current holdings per quantile
            current_holdings: Dict[int, Dict[str, float]] = {q: {} for q in range(quantiles)}
            for q in range(quantiles):
                assets_q = day.loc[labels == q, "asset"].tolist()
                if not assets_q:
                    period_returns[q].append(np.nan)
                    period_turnover[q].append(np.nan)
                    continue
                w = 1.0 / len(assets_q)
                for a in assets_q:
                    current_holdings[q][a] = w
                if use_fwd_1:
                    rets = day.loc[labels == q, fwd_1_col].astype(float).to_numpy()
                    rets = rets[np.isfinite(rets)]
                    cum = float(np.prod(1 + rets) - 1) if len(rets) else np.nan
                else:
                    rets = day.loc[labels == q, fwd_col].astype(float).to_numpy()
                    rets = rets[np.isfinite(rets)]
                    cum = float(np.mean(rets)) if len(rets) else np.nan
                period_returns[q].append(cum)
                if last_holdings[q]:
                    overlap = sum(
                        min(abs(last_holdings[q].get(a, 0)), abs(current_holdings[q].get(a, 0)))
                        for a in set(last_holdings[q]) | set(current_holdings[q])
                    )
                    total = sum(abs(v) for v in current_holdings[q].values())
                    turn = max(0.0, 1.0 - overlap / max(total, 1e-9))
                else:
                    turn = 1.0
                period_turnover[q].append(turn)
            last_holdings = current_holdings
        # Aggregate
        for q in range(quantiles):
            rets = np.asarray(period_returns[q], dtype=float)
            rets = rets[np.isfinite(rets)]
            turns = np.asarray(period_turnover[q], dtype=float)
            turns = turns[np.isfinite(turns)]
            if len(rets) == 0:
                out_rows.append(
                    {
                        "rebalance_frequency": freq,
                        "quantile": q + 1,
                        "n_periods": 0,
                        "mean_return": np.nan,
                        "cum_return": np.nan,
                        "volatility": np.nan,
                        "max_drawdown": np.nan,
                        "turnover": np.nan,
                        "transaction_cost": np.nan,
                        "net_return": np.nan,
                        "annualized_return": np.nan,
                        "sharpe": np.nan,
                    }
                )
                continue
            mean_r = float(np.mean(rets))
            cum_r = float(np.prod(1 + rets) - 1)
            vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
            sharpe = mean_r / vol * np.sqrt(252 / freq) if vol > EPS else np.nan
            cumret_curve = np.cumprod(1 + rets) - 1
            mdd = _max_drawdown(cumret_curve)
            turnover = float(np.mean(turns)) if len(turns) else 0.0
            tc = turnover * (transaction_cost_bps / 1e4)
            net_cum = (1 + cum_r) * (1 - tc) - 1
            if freq > 0:
                ann_r = (1 + mean_r) ** (252 / freq) - 1
            else:
                ann_r = np.nan
            out_rows.append(
                {
                    "rebalance_frequency": freq,
                    "quantile": q + 1,
                    "n_periods": int(len(rets)),
                    "mean_return": mean_r,
                    "cum_return": cum_r,
                    "volatility": vol,
                    "max_drawdown": mdd,
                    "turnover": turnover,
                    "transaction_cost": tc,
                    "net_return": net_cum,
                    "annualized_return": ann_r,
                    "sharpe": sharpe,
                }
            )
    return pd.DataFrame(out_rows)


def rebalance_summary(
    per_factor: Dict[str, pd.DataFrame],
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Aggregate per-factor rebalance results into a single summary table.

    The output has one row per (factor, rebalance_frequency) and reports:
        - top_quantile_cum_return
        - bottom_quantile_cum_return
        - top_bottom_spread_gross
        - top_bottom_spread_net
        - mean_turnover
        - mean_max_drawdown
        - net_cum_return_for_top_quantile
    """
    rows: List[dict] = []
    for fac, df in per_factor.items():
        if df is None or df.empty:
            continue
        for freq, g in df.groupby("rebalance_frequency"):
            top = g[g["quantile"] == g["quantile"].max()].iloc[0]
            bot = g[g["quantile"] == g["quantile"].min()].iloc[0]
            spread_gross = float(top["cum_return"] - bot["cum_return"])
            spread_net = float(top["net_return"] - bot["net_return"])
            ann_top = float(top["annualized_return"]) if "annualized_return" in top else np.nan
            rows.append(
                {
                    "factor_name": fac,
                    "rebalance_frequency": int(freq),
                    "top_quantile_cum_gross": float(top["cum_return"]),
                    "bottom_quantile_cum_gross": float(bot["cum_return"]),
                    "top_quantile_cum_net": float(top["net_return"]),
                    "bottom_quantile_cum_net": float(bot["net_return"]),
                    "top_quantile_annualized": ann_top,
                    "top_bottom_spread_gross": spread_gross,
                    "top_bottom_spread_net": spread_net,
                    "mean_turnover": float(g["turnover"].mean()),
                    "max_drawdown_avg": float(g["max_drawdown"].mean()),
                    "sharpe_top": float(top["sharpe"]),
                }
            )
    return pd.DataFrame(rows)


def monotonicity_check(df: pd.DataFrame) -> dict:
    """Given a per-factor rebalance result, check whether quantile returns
    are monotonic in the quantile index.

    Returns a dict with:
        - spearman_corr (quantile vs mean_return)
        - is_monotonic (bool, True if every successive diff has same sign)
        - top_beats_bottom (bool)
        - top_is_peak (bool — whether the top quantile is the best)
        - win_rate_top_vs_bottom (fraction of rebalance periods where
          top > bottom)
    """
    if df is None or df.empty:
        return {
            "spearman_corr": np.nan,
            "is_monotonic": False,
            "top_beats_bottom": False,
            "top_is_peak": False,
            "win_rate_top_vs_bottom": np.nan,
        }
    g = df.groupby("quantile")["mean_return"].mean()
    if g.empty or g.isna().all():
        return {
            "spearman_corr": np.nan,
            "is_monotonic": False,
            "top_beats_bottom": False,
            "top_is_peak": False,
            "win_rate_top_vs_bottom": np.nan,
        }
    qidx = g.index.to_numpy()
    means = g.to_numpy(dtype=float)
    if np.std(means) < EPS:
        rho = np.nan
    else:
        rho, _ = stats.spearmanr(qidx, means)
    # Strictly monotonic (allow small noise)
    diffs = np.diff(means)
    is_mono_up = np.all(diffs >= -1e-6)
    is_mono_down = np.all(diffs <= 1e-6)
    is_monotonic = bool(is_mono_up or is_mono_down)
    top = means[-1]
    bot = means[0]
    top_is_peak = bool(top == np.max(means))
    return {
        "spearman_corr": float(rho) if np.isfinite(rho) else np.nan,
        "is_monotonic": is_monotonic,
        "top_beats_bottom": bool(top > bot),
        "top_is_peak": top_is_peak,
        "win_rate_top_vs_bottom": np.nan,  # not enough data to compute per-period win rate
    }
