"""Data validation and alignment utilities.

Provides:
- Validation of prices DataFrame (date, asset, close + optional fields)
- Validation of long-format factors DataFrame (date, asset, factor_name, factor_value)
- Robust conversion of dates to pandas Timestamps
- Cross-sectional alignment of factors and forward returns
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLS = ["date", "asset", "close"]
REQUIRED_FACTOR_COLS = ["date", "asset", "factor_name", "factor_value"]


def validate_prices(prices: pd.DataFrame) -> None:
    """Validate that prices DataFrame has the required schema.

    Required columns: date, asset, close.
    Optional columns: volume, amount, market_cap, industry, sub_industry,
        theme_exposure, beta, is_robot_stock.

    Raises
    ------
    ValueError
        If required columns are missing, dates cannot be parsed, or the frame is empty.
    """
    if prices is None or prices.empty:
        raise ValueError("prices DataFrame is empty or None")
    missing = [c for c in REQUIRED_PRICE_COLS if c not in prices.columns]
    if missing:
        raise ValueError(f"prices missing required columns: {missing}")
    if prices["close"].isna().all():
        raise ValueError("prices 'close' column is all NaN")
    try:
        pd.to_datetime(prices["date"])
    except Exception as exc:
        raise ValueError(f"prices 'date' cannot be parsed: {exc}")


def validate_factors(factors: pd.DataFrame) -> None:
    """Validate that factors DataFrame is in long format.

    Required columns: date, asset, factor_name, factor_value.

    Raises
    ------
    ValueError
        If the frame is missing required columns or cannot be parsed.
    """
    if factors is None or factors.empty:
        raise ValueError("factors DataFrame is empty or None")
    missing = [c for c in REQUIRED_FACTOR_COLS if c not in factors.columns]
    if missing:
        raise ValueError(f"factors missing required columns: {missing}")
    try:
        pd.to_datetime(factors["date"])
    except Exception as exc:
        raise ValueError(f"factors 'date' cannot be parsed: {exc}")


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with 'date' column normalized to midnight Timestamp."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def list_factor_names(factors: pd.DataFrame) -> List[str]:
    """Return sorted list of distinct factor_name values."""
    if factors is None or factors.empty:
        return []
    return sorted(factors["factor_name"].dropna().unique().tolist())


def align_factor_returns(
    factor_long: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_cross_section: int = 20,
) -> pd.DataFrame:
    """Align a single factor's long table against forward returns.

    Parameters
    ----------
    factor_long
        DataFrame with columns: date, asset, factor_value.
    forward_returns
        DataFrame with columns: date, asset, and one or more `fwd_ret_<h>` columns.
    min_cross_section
        If a date has fewer than this many aligned observations, all returns are set NaN.

    Returns
    -------
    pd.DataFrame
        Columns: date, asset, factor_value, fwd_ret_<h> for each h.
    """
    if factor_long is None or factor_long.empty:
        return pd.DataFrame()
    if forward_returns is None or forward_returns.empty:
        return pd.DataFrame()

    f = factor_long[["date", "asset", "factor_value"]].copy()
    r = forward_returns.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()

    merged = f.merge(r, on=["date", "asset"], how="inner")
    if merged.empty:
        return merged

    # Filter dates with too few observations (per date) per factor column
    keep_mask = pd.Series(True, index=merged.index)
    fwd_cols = [c for c in merged.columns if c.startswith("fwd_ret_")]
    for col in fwd_cols:
        grp_count = merged.groupby("date")[col].transform("count")
        keep_mask &= grp_count >= min_cross_section

    merged = merged.loc[keep_mask].copy()
    return merged


def get_universe_mask(
    prices: pd.DataFrame,
    mode: str = "theme_core",
    theme_exposure_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a per (date, asset) boolean mask defining the study universe.

    Parameters
    ----------
    prices
        Price DataFrame (with optional theme_exposure / is_robot_stock).
    mode
        - ``"theme_core"``: ``is_robot_stock`` is True (fallback: theme_exposure >= 0.5).
        - ``"theme_diffused"``: all assets that have theme_exposure defined and > 0.
        - ``"all"``: every (date, asset) pair.
    theme_exposure_threshold
        Threshold for theme_core when is_robot_stock is missing.

    Returns
    -------
    pd.DataFrame
        Columns: date, asset, in_universe.
    """
    df = prices[["date", "asset"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    in_universe = pd.Series(True, index=df.index)

    if mode == "theme_core":
        if "is_robot_stock" in prices.columns:
            in_universe = prices["is_robot_stock"].fillna(False).astype(bool).values
        elif "theme_exposure" in prices.columns:
            in_universe = (
                prices["theme_exposure"].fillna(0) >= theme_exposure_threshold
            ).values
    elif mode == "theme_diffused":
        if "theme_exposure" in prices.columns:
            in_universe = (prices["theme_exposure"].fillna(0) > 0).values
        else:
            in_universe = pd.Series(True, index=df.index).values
    elif mode == "all":
        in_universe = pd.Series(True, index=df.index).values
    else:
        raise ValueError(f"unknown universe mode: {mode}")

    df["in_universe"] = in_universe.astype(bool)
    return df


def restrict_to_universe(
    df: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    """Inner-merge a long DataFrame with a universe mask on (date, asset)."""
    if df is None or df.empty:
        return df
    if universe is None or universe.empty:
        return df.iloc[0:0]
    return df.merge(universe, on=["date", "asset"], how="inner")


def split_periods(
    dates: Iterable[pd.Timestamp],
    event_date: Optional[pd.Timestamp],
    post_event_window: int = 120,
) -> dict:
    """Classify dates into pre/post event periods.

    Returns a dict with keys ``pre_event`` and ``post_event`` (each a sorted
    unique list of Timestamps). If ``event_date`` is None, all dates are
    returned in ``"all"`` and the split is empty.
    """
    uniq = pd.Series(sorted(set(pd.to_datetime(list(dates)).normalize())))
    if event_date is None:
        return {"all": uniq, "pre_event": [], "post_event": [], "event_date": None}

    ev = pd.to_datetime(event_date).normalize()
    pre = uniq[uniq < ev].tolist()
    post_dates = uniq[uniq > ev]
    if len(post_dates) > post_event_window:
        post_dates = post_dates.iloc[:post_event_window]
    return {
        "all": uniq,
        "pre_event": pre,
        "post_event": post_dates.tolist(),
        "event_date": ev,
    }
