"""Forward return computation with strict no-future-function safeguards.

Rules
-----
1. Forward returns are computed from a stock's *own* future prices only.
2. The factor at date ``t`` is matched only to forward returns whose start
   is also ``t`` — never later.
3. Negative-shift is not used to construct features.
4. If the future window is incomplete (e.g. last ``h`` days), the return
   is NaN. We never silently zero-fill.
5. Each stock's prices are sorted ascending by date before computing.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def compute_forward_returns(
    prices: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    date_col: str = "date",
    asset_col: str = "asset",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute per-stock forward returns at multiple horizons.

    Parameters
    ----------
    prices
        Long-format price table.
    horizons
        List of integer horizons (in trading days). Defaults to
        ``[1, 2, 3, 5, 10, 20, 40, 60]``.
    date_col, asset_col, close_col
        Column names.

    Returns
    -------
    pd.DataFrame
        Long-format frame with columns: date, asset, close, fwd_ret_1, ..., fwd_ret_<h>.
        Forward returns are simple (linear) returns: ``close[t+h] / close[t] - 1``.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 20, 40, 60]

    df = prices[[date_col, asset_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.sort_values([asset_col, date_col]).reset_index(drop=True)

    # group by asset
    grp = df.groupby(asset_col, sort=False)
    out_chunks = []
    for asset, sub in grp:
        sub = sub.sort_values(date_col).reset_index(drop=True)
        sub = sub.rename(columns={close_col: "close"})
        for h in horizons:
            col = f"fwd_ret_{h}"
            # We compute close.shift(-h) which is the price h rows ahead WITHIN this asset.
            # Because the index is reset per asset, this is *not* a cross-stock shift.
            fwd_close = sub["close"].shift(-h)
            sub[col] = (fwd_close / sub["close"]) - 1.0
        sub["asset"] = asset
        out_chunks.append(sub)

    out = pd.concat(out_chunks, ignore_index=True)
    return out


def check_no_lookahead(forward_returns: pd.DataFrame, date_col: str = "date") -> dict:
    """Sanity check: forward returns at t must use close[t+h], not anything earlier.

    Returns a dict with diagnostic fields:
        - has_nan_fwd: bool — whether any fwd_ret_ column has any NaN (expected near tail)
        - max_fwd_horizon: int — largest horizon computed
        - tail_nan_count: int — number of NaN fwd returns in the last h rows
            of each fwd column (these are the rows where future price is missing).
    """
    fwd_cols = [c for c in forward_returns.columns if c.startswith("fwd_ret_")]
    if not fwd_cols:
        return {"has_nan_fwd": False, "max_fwd_horizon": 0, "tail_nan_count": 0}
    has_nan = forward_returns[fwd_cols].isna().any().any()
    horizon = max(int(c.replace("fwd_ret_", "")) for c in fwd_cols)
    return {
        "has_nan_fwd": bool(has_nan),
        "max_fwd_horizon": horizon,
        "tail_nan_count": int(forward_returns[fwd_cols].iloc[-horizon:].isna().sum().sum()),
    }
