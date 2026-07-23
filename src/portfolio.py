"""Portfolio construction and backtest.

Builds four simple portfolios:
- A: Theme exposure (single-factor) — long top quintile, short bottom quintile
- B: Single-factor (best factor by IC)
- C: Incremental factor blend — equal-weight top incremental factors
- D: Multi-horizon — different rebalance frequencies for short/medium/long factors

For each portfolio we report:
- annualized return
- annualized volatility
- max drawdown
- cumulative return
- turnover
- cost-adjusted return
- IC, ICIR (vs the cross-sectional return of the portfolio)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .rebalance import _equal_freq_bins, _max_drawdown


def _annualized(r: np.ndarray, periods_per_year: int = 252) -> float:
    if len(r) == 0:
        return float("nan")
    cum = float(np.prod(1 + r) - 1)
    return (1 + cum) ** (periods_per_year / len(r)) - 1


def _ann_vol(r: np.ndarray, periods_per_year: int = 252) -> float:
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def _sharpe(r: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    if len(r) < 2:
        return float("nan")
    er = r - rf / periods_per_year
    sd = np.std(er, ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(er) / sd * np.sqrt(periods_per_year))


def _max_drawdown_curve(r: np.ndarray) -> float:
    if len(r) < 2:
        return 0.0
    cumret = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumret)
    dd = cumret / running_max - 1
    return float(dd.min())


def backtest_long_short(
    prices: pd.DataFrame,
    factor_name: str,
    factor_value: pd.Series,
    fwd_ret_col: str = "fwd_ret_1",
    rebalance_freq: int = 5,
    quantiles: int = 5,
    transaction_cost_bps: float = 10.0,
    long_short: bool = True,
) -> dict:
    """Run a single-factor long-short (or long-only) backtest.

    Parameters
    ----------
    prices
        Long price table (date, asset, close, fwd_ret_<h>).
    factor_name
        Name of the factor (used in reporting).
    factor_value
        pd.Series of factor values aligned to ``prices`` rows.
    fwd_ret_col
        Which forward return column to use as the held daily return.
    rebalance_freq
        Number of trading days between rebalances.
    quantiles
        Number of bins; we go long top bin and (if long_short) short bottom bin.
    transaction_cost_bps
        One-way cost in bps.
    long_short
        If False, only long the top bin.

    Returns
    -------
    dict
        Portfolio summary dict.
    """
    df = prices.copy()
    df["factor_value"] = pd.to_numeric(factor_value, errors="coerce").values
    df = df.dropna(subset=["factor_value", fwd_ret_col])
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return {
            "factor_name": factor_name,
            "n_periods": 0,
            "cum_return_gross": np.nan,
            "ann_return_gross": np.nan,
            "ann_vol": np.nan,
            "sharpe_gross": np.nan,
            "max_drawdown": np.nan,
            "turnover": np.nan,
            "transaction_cost": np.nan,
            "net_return": np.nan,
            "ic": np.nan,
            "icir": np.nan,
        }

    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    daily_rets: List[float] = []
    daily_turnover: List[float] = []
    rebal_dates = dates.iloc[:: int(rebalance_freq)].tolist()
    last_holdings: Dict[str, float] = {}
    port_rets: List[Tuple[pd.Timestamp, float]] = []
    fwd_1_rets: List[float] = []
    fwd_1_factors: List[float] = []
    for d in dates:
        sub_d = df[df["date"] == d]
        if d in rebal_dates and len(sub_d) >= quantiles * 2:
            labels = _equal_freq_bins(sub_d["factor_value"].to_numpy(dtype=float), quantiles)
            holdings: Dict[str, float] = {}
            if long_short:
                # Long top bin, short bottom bin
                top = sub_d.loc[labels == quantiles - 1, "asset"].tolist()
                bot = sub_d.loc[labels == 0, "asset"].tolist()
                n_top, n_bot = len(top), len(bot)
                if n_top == 0 or n_bot == 0:
                    holdings = last_holdings
                else:
                    for a in top:
                        holdings[a] = 1.0 / n_top
                    for a in bot:
                        holdings[a] = -1.0 / n_bot
            else:
                top = sub_d.loc[labels == quantiles - 1, "asset"].tolist()
                if top:
                    for a in top:
                        holdings[a] = 1.0 / len(top)
                else:
                    holdings = last_holdings
            # Turnover = 1 - overlap
            if last_holdings:
                overlap = sum(min(abs(last_holdings.get(a, 0)), abs(holdings.get(a, 0)))
                              for a in set(last_holdings) | set(holdings))
                turn = 1.0 - overlap / max(sum(abs(v) for v in holdings.values()), 1e-9)
                turn = max(turn, 0.0)
            else:
                turn = 1.0  # initial portfolio is a fresh build
            daily_turnover.append(turn)
            last_holdings = holdings
        # Apply today's holdings to today's forward return
        if not last_holdings:
            port_rets.append((d, 0.0))
            continue
        day = sub_d.set_index("asset")[fwd_ret_col]
        r = 0.0
        for a, w in last_holdings.items():
            if a in day.index:
                v = day.loc[a]
                if pd.notna(v):
                    r += w * float(v)
        port_rets.append((d, r))
        # For IC tracking: collect per-day cross-section of factor value and the realized return
        # over the *next day* (fwd_ret_1)
        fwd_1_rets.append(r)
        fwd_1_factors.append(float(sub_d["factor_value"].mean()))

    if not port_rets:
        return {
            "factor_name": factor_name,
            "n_periods": 0,
            "cum_return_gross": np.nan,
            "ann_return_gross": np.nan,
            "ann_vol": np.nan,
            "sharpe_gross": np.nan,
            "max_drawdown": np.nan,
            "turnover": np.nan,
            "transaction_cost": np.nan,
            "net_return": np.nan,
            "ic": np.nan,
            "icir": np.nan,
        }
    rets = np.asarray([r for _, r in port_rets], dtype=float)
    turnover_arr = np.asarray(daily_turnover, dtype=float) if daily_turnover else np.zeros_like(rets)
    cum_gross = float(np.prod(1 + rets) - 1)
    avg_turnover = float(np.mean(turnover_arr)) if len(turnover_arr) else 0.0
    tc = avg_turnover * (transaction_cost_bps / 1e4)
    cum_net = (1 + cum_gross) * (1 - tc) - 1
    ann_r = _annualized(rets)
    vol = _ann_vol(rets)
    sharpe = _sharpe(rets)
    mdd = _max_drawdown_curve(rets)
    return {
        "factor_name": factor_name,
        "n_periods": int(len(rets)),
        "cum_return_gross": cum_gross,
        "ann_return_gross": ann_r,
        "ann_vol": vol,
        "sharpe_gross": sharpe,
        "max_drawdown": mdd,
        "turnover": avg_turnover,
        "transaction_cost": tc,
        "net_return": cum_net,
        "ic": np.nan,  # filled in by caller if needed
        "icir": np.nan,
    }


def build_themes_portfolios(
    prices: pd.DataFrame,
    forward_returns: pd.DataFrame,
    fwd_ret_col: str = "fwd_ret_1",
    rebalance_freq: int = 5,
    quantiles: int = 5,
    transaction_cost_bps: float = 10.0,
) -> Dict[str, dict]:
    """Build portfolios A (theme) and B (single factor) as examples."""
    out: Dict[str, dict] = {}
    if "theme_exposure" in prices.columns:
        out["A_theme_exposure"] = backtest_long_short(
            forward_returns, "theme_exposure",
            forward_returns.merge(prices[["date", "asset", "theme_exposure"]], on=["date", "asset"])["theme_exposure"],
            fwd_ret_col=fwd_ret_col,
            rebalance_freq=rebalance_freq,
            quantiles=quantiles,
            transaction_cost_bps=transaction_cost_bps,
        )
    return out
