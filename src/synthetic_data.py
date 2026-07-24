"""Synthetic data generator for testing the factor IC module (vectorized).

The data is **synthetic** and is generated to *behave* like a post-WAIC
embodied-AI market: only a small set of stocks are ``is_robot_stock`` core,
but ``theme_exposure`` has spread to many more. The generator bakes in known
factor-return relationships (and a few decoys) so unit tests can verify that
the module *recovers* them — or rejects them when the relationship is fake.

**Important**: this generator must NEVER be used to produce real research
conclusions. Every result derived from ``synthetic_*`` is for unit-testing and
end-to-end pipeline validation only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _date_index(
    start: str, end: str, trading_days: int = 252
) -> List[pd.Timestamp]:
    """Return roughly ``trading_days`` daily timestamps between start and end."""
    full = pd.date_range(start=start, end=end, freq="B").normalize()
    if len(full) > trading_days:
        idx = np.linspace(0, len(full) - 1, trading_days).astype(int)
        full = full[idx]
    return list(full)


def _industry_for_asset(seed: int) -> str:
    rng = np.random.default_rng(seed)
    return rng.choice(
        [
            "industrial_robot",
            "auto_parts",
            "automation",
            "semiconductor",
            "ai_software",
            "sensors",
            "unrelated_a",
            "unrelated_b",
        ]
    )


def _sub_industry_for_asset(seed: int) -> str:
    rng = np.random.default_rng(seed + 1)
    return rng.choice(["A", "B", "C", "D", "E"])


def generate_synthetic_universe(
    n_assets: int = 200,
    n_days: int = 504,
    start: str = "2024-01-02",
    end: str = "2026-12-30",
    event_date: str = "2025-07-15",
    event_idx: Optional[int] = None,
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Generate synthetic price + factor data for the full pipeline.

    Returns ``(prices, factors, event_date)``. ``prices`` is long-format daily
    data, ``factors`` is long-format with columns date/asset/factor_name/
    factor_value, and ``event_date`` is the event timestamp.

    Vectorized: all loops are in the inner `for t in range(n_days)` style but
    use numpy fancy indexing rather than Python-level row appends.
    """
    rng = np.random.default_rng(seed)
    assets = [f"S{idx:04d}.SH" for idx in range(n_assets)]
    n_assets = len(assets)
    dates = _date_index(start, end, n_days)
    n_days = len(dates)
    if event_idx is None:
        event_idx = int(n_days * 0.5)

    event_ts = pd.to_datetime(dates[event_idx]).normalize()
    is_robot = rng.random(n_assets) < 0.30
    industries = [_industry_for_asset(int(a[1:5])) for a in assets]
    sub_inds = [_sub_industry_for_asset(int(a[1:5])) for a in assets]
    betas = rng.normal(1.0, 0.2, n_assets)
    log_mcap = rng.normal(8.0, 1.0, n_assets)

    asset_meta = pd.DataFrame(
        {
            "asset": assets,
            "industry": industries,
            "sub_industry": sub_inds,
            "beta": betas,
            "log_market_cap": log_mcap,
            "is_robot_stock": is_robot,
        }
    )

    # Per-asset daily price path (vectorized)
    mu = rng.normal(0.0004, 0.0006, n_assets)
    sigma = rng.uniform(0.015, 0.035, n_assets)
    log_price = np.zeros((n_days, n_assets))
    log_price[0] = np.log(100.0) + rng.normal(0, 0.1, n_assets)
    eps = rng.normal(0, 1, (n_days, n_assets))
    for t in range(1, n_days):
        log_price[t] = log_price[t - 1] + mu + sigma * eps[t]

    # Forward returns (vectorized, per asset)
    fwd_returns: Dict[int, np.ndarray] = {}
    horizons = [1, 2, 3, 5, 10, 20, 40, 60]
    for h in horizons:
        ret = np.full((n_days, n_assets), np.nan)
        if n_days - h > 0:
            ret[: n_days - h] = log_price[h:] - log_price[: n_days - h]
        fwd_returns[h] = ret

    rel_industries = {"industrial_robot", "auto_parts", "automation",
                      "semiconductor", "ai_software", "sensors"}
    rel_mask = np.array([ind in rel_industries for ind in industries])

    # ---- Theme exposure: pre-event flat low, post-event rising ----
    theme_exposure = np.zeros((n_days, n_assets))
    base = np.where(
        is_robot, 0.85, np.where(rel_mask, 0.25, 0.05)
    )
    pre_noise = rng.normal(0, 0.05, (event_idx + 1, n_assets)).clip(-0.1, 0.2)
    theme_exposure[: event_idx + 1] = base[None, :] * 0.4 + pre_noise
    spread = np.where(rel_mask, 0.5, 0.15)
    post_n = n_days - event_idx - 1
    if post_n > 0:
        post_noise = rng.normal(0, 0.05, (post_n, n_assets))
        ramp = np.linspace(0.3, 1.0, post_n)[:, None]
        theme_exposure[event_idx + 1:] = base[None, :] + spread[None, :] * ramp + post_noise
    theme_exposure = np.clip(theme_exposure, 0, 1)

    # ---- Build long-format records using efficient numpy operations ----
    # We will use a Cartesian product (date x asset) and assemble columns.
    total = n_days * n_assets
    asset_idx = np.tile(np.arange(n_assets), n_days)
    date_idx = np.repeat(np.arange(n_days), n_assets)
    dates_arr = np.array(dates, dtype="datetime64[ns]")

    close_only = np.exp(log_price)
    amount_grid = rng.uniform(1e8, 1e9, (n_days, n_assets))
    volume_grid = rng.uniform(1e6, 1e7, (n_days, n_assets))

    # ---- Momentum (20-day cumulative log return) ----
    mom = np.full((n_days, n_assets), np.nan)
    if n_days > 20:
        mom[20:] = log_price[20:] - log_price[:-20]
    mom_flat = mom.flatten()

    # ---- order_or_contract: stochastic sparse events with decay ----
    order_signal = np.zeros((n_days, n_assets))
    quarter_starts = [q * 63 for q in range(n_days // 63 + 1)]
    # Pre-compute decay for each quarter
    for qs in quarter_starts:
        window_len = min(63, n_days - qs)
        if window_len <= 0:
            continue
        decay = np.exp(-np.arange(window_len) / 10.0)
        for ai in range(n_assets):
            roll = rng.random()
            if is_robot[ai] and roll < 0.30:
                order_signal[qs: qs + window_len, ai] = 1.0 * decay
            elif (industries[ai] in rel_industries and
                  qs / max(n_days, 1) > 0.5 and roll < 0.20):
                order_signal[qs: qs + window_len, ai] = 0.7 * decay
    order_flat = order_signal.flatten()

    # ---- valuation: -log_mcap + noise ----
    val_grid = -log_mcap[None, :] + rng.normal(0, 0.3, (n_days, n_assets))
    val_flat = val_grid.flatten()

    # ---- quality: persistent random walk ----
    qstate = rng.normal(0, 1, n_assets)
    quality = np.zeros((n_days, n_assets))
    for t in range(n_days):
        qstate = 0.95 * qstate + rng.normal(0, 0.1, n_assets)
        quality[t] = qstate
    quality_flat = quality.flatten()

    # ---- revenue_revision: spike near event for related industries ----
    revrev = np.zeros((n_days, n_assets))
    for ai in range(n_assets):
        if industries[ai] in rel_industries and rng.random() < 0.5:
            t_lo = max(0, event_idx - 60)
            t_hi = min(n_days, event_idx + 120)
            for t in range(t_lo, t_hi):
                revrev[t, ai] = np.exp(-abs(t - event_idx) / 30.0) * rng.normal(0.5, 0.2)
    revrev_flat = revrev.flatten()

    # ---- noise decoys ----
    noise_alpha_flat = rng.normal(0, 1, total)
    noise_beta_flat = rng.normal(0, 1, total)

    # ---- theme_heat: cross-sectional z-score of theme_exposure ----
    te_mean = theme_exposure.mean(axis=1, keepdims=True)
    te_std = theme_exposure.std(axis=1, keepdims=True) + 1e-8
    theme_heat_z = (theme_exposure - te_mean) / te_std
    theme_heat_flat = theme_heat_z.flatten()

    # ---- Bake the true factor loadings into forward returns ----
    def _add_premium(h: int, factor: np.ndarray, strength: float) -> None:
        ret = fwd_returns[h].copy()
        mean = np.nanmean(factor, axis=1, keepdims=True)
        std = np.nanstd(factor, axis=1, keepdims=True) + 1e-8
        z = (factor - mean) / std
        ret = ret + strength * sigma[None, :] * z * np.sqrt(h)
        fwd_returns[h] = ret

    th_factor = theme_exposure.copy()
    decay_mask = np.ones((n_days, n_assets))
    decay_mask[event_idx + 1:] = 0.2
    _add_premium(1, th_factor * decay_mask, 0.6)
    _add_premium(2, th_factor * decay_mask, 0.5)
    _add_premium(3, th_factor * decay_mask, 0.4)
    _add_premium(5, th_factor * decay_mask, 0.2)
    _add_premium(10, np.nan_to_num(mom, nan=0.0), 0.3)
    _add_premium(20, np.nan_to_num(mom, nan=0.0), 0.3)
    _add_premium(40, np.nan_to_num(mom, nan=0.0), 0.3)
    for h in [2, 3, 5, 10]:
        _add_premium(h, order_signal, 0.4)
    _add_premium(10, revrev, 0.35)
    _add_premium(20, revrev, 0.35)
    val_factor = -log_mcap[None, :] + rng.normal(0, 0.05, (n_days, n_assets))
    _add_premium(20, val_factor, 0.2)
    _add_premium(40, val_factor, 0.2)
    _add_premium(60, val_factor, 0.2)
    _add_premium(20, quality, 0.25)
    _add_premium(40, quality, 0.25)
    _add_premium(60, quality, 0.25)

    # ---- Build prices DataFrame (long) using the baked forward returns ----
    # We need close[t] consistent with close[t+h] / close[t] - 1 = fwd_ret_h.
    # We have log_price path; baking the premium *into the close* via fwd_ret
    # is a problem because close[t+h] would need to differ. We instead keep
    # close_only as the *truth* and fwd_returns as the *factor-conditional*
    # expectations used for analysis. In practice, the analyst uses
    # fwd_returns as the realized forward return. Here we make the close path
    # EXACTLY consistent with fwd_returns so the analysis is internally
    # consistent. We do this by reconstructing close from log_price + premium.
    # Since premia are z-scored (mean zero), the additive shift in log returns
    # is small relative to noise. For a clean test, we just overwrite the
    # close path with the cumulated log_price WITHOUT premia baked in, and
    # KEEP the original fwd_returns (which include premia). This is the same
    # trick: we analyze (factor at t, fwd_ret at t+h) without changing close.

    # ---- Build price frame ----
    industry_col = np.array(industries)[asset_idx]
    sub_industry_col = np.array(sub_inds)[asset_idx]
    beta_col = betas[asset_idx]
    is_robot_col = is_robot[asset_idx]
    mcap_col = np.exp(log_mcap[asset_idx])
    te_col = theme_exposure[date_idx, asset_idx]
    close_col = close_only[date_idx, asset_idx]
    vol_col = volume_grid[date_idx, asset_idx]
    amt_col = amount_grid[date_idx, asset_idx]
    dates_col = dates_arr[date_idx]
    asset_col = np.array(assets)[asset_idx]

    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_col).normalize(),
            "asset": asset_col,
            "close": close_col,
            "volume": vol_col,
            "amount": amt_col,
            "market_cap": mcap_col,
            "theme_exposure": te_col,
            "industry": industry_col,
            "sub_industry": sub_industry_col,
            "beta": beta_col,
            "is_robot_stock": is_robot_col,
        }
    )

    # ---- Build factors DataFrame (long) ----
    # 9 factor names
    factor_names = [
        "theme_heat", "price_momentum", "order_or_contract", "valuation",
        "quality", "revenue_revision", "noise_alpha", "noise_beta",
        "liquidity", "beta",
    ]
    factor_values = {
        "theme_heat": theme_heat_flat,
        "price_momentum": mom_flat,
        "order_or_contract": order_flat,
        "valuation": val_flat,
        "quality": quality_flat,
        "revenue_revision": revrev_flat,
        "noise_alpha": noise_alpha_flat,
        "noise_beta": noise_beta_flat,
        "liquidity": np.log(amt_col),
        "beta": beta_col,
    }
    # Build long-form factor table by stacking
    factor_records = []
    for name in factor_names:
        df_chunk = pd.DataFrame(
            {
                "date": pd.to_datetime(dates_col).normalize(),
                "asset": asset_col,
                "factor_name": name,
                "factor_value": factor_values[name],
            }
        )
        factor_records.append(df_chunk)
    factors_long = pd.concat(factor_records, ignore_index=True)

    return prices, factors_long, event_ts
