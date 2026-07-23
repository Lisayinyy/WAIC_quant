"""IC analysis: daily cross-sectional Pearson / Spearman correlation.

Computes per-date:
- Pearson IC
- Spearman RankIC
- IC mean, std, ICIR, positive-IC ratio
- t-stat, p-value, 95% confidence interval
- n_obs

Then aggregates per (factor, horizon, period, universe).

Theme-controlled (partial) IC is computed by:
1. Standardizing continuous control variables cross-sectionally.
2. One-hot encoding categorical controls (sub_industry, industry).
3. Regressing factor_value on controls → residual_factor.
4. Regressing fwd_ret on controls → residual_return.
5. Pearson correlation between residual_factor and residual_return.

If controls have insufficient rank (e.g. all values equal), partial IC is
returned as NaN with a reason code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


MIN_OBS_FOR_REGRESSION = 8  # below this we skip partial IC
EPS = 1e-12


@dataclass
class ICDaily:
    date: pd.Timestamp
    factor_name: str
    horizon: int
    period: str
    universe: str
    n_obs: int
    pearson_ic: float
    rank_ic: float
    partial_ic: Optional[float]


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    if len(x) < 3:
        return float("nan"), len(x)
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan"), len(x)
    r, _ = stats.pearsonr(x, y)
    return float(r), len(x)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    if len(x) < 3:
        return float("nan"), len(x)
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan"), len(x)
    r, _ = stats.spearmanr(x, y)
    return float(r), len(x)


def _ols_residual(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return residual of y ~ X. If X has insufficient rank, return NaN-filled array + ok=False."""
    if X.shape[0] != len(y):
        return np.full_like(y, np.nan, dtype=float), False
    rank = np.linalg.matrix_rank(X, tol=1e-8)
    if rank < X.shape[1]:
        return np.full_like(y, np.nan, dtype=float), False
    # Solve normal equations
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return resid, True
    except np.linalg.LinAlgError:
        return np.full_like(y, np.nan, dtype=float), False


def _cross_section_standardize(s: pd.Series) -> pd.Series:
    m = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd < EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd


def _build_control_matrix(
    df: pd.DataFrame, controls: List[str]
) -> Tuple[Optional[np.ndarray], List[str], str]:
    """Build a per-day design matrix from the listed control variables.

    Continuous controls are cross-sectionally standardized. Categorical controls
    are one-hot encoded (dropping the first level to avoid the dummy trap).

    Returns ``(X, used_columns, reason_if_failed)``. ``X`` is None if it cannot
    be built (e.g. all controls missing).
    """
    parts: List[pd.DataFrame] = []
    used_cols: List[str] = []
    if not controls:
        return None, [], "no_controls_specified"

    for ctrl in controls:
        if ctrl not in df.columns:
            continue
        s = df[ctrl]
        if s.isna().all():
            continue
        if pd.api.types.is_numeric_dtype(s):
            std = _cross_section_standardize(s.fillna(s.median()))
            parts.append(std.rename(ctrl))
            used_cols.append(ctrl)
        else:
            # Categorical
            filled = s.fillna("__missing__").astype(str)
            dummies = pd.get_dummies(filled, prefix=ctrl, drop_first=True, dummy_na=False)
            if dummies.shape[1] == 0:
                continue
            parts.append(dummies)
            used_cols.extend(list(dummies.columns))

    if not parts:
        return None, [], "no_valid_controls"

    X = pd.concat(parts, axis=1)
    # Drop constant columns
    keep = X.std() > EPS
    X = X.loc[:, keep]
    if X.shape[1] == 0:
        return None, [], "controls_constant"
    # Add intercept
    X.insert(0, "intercept", 1.0)
    return X.values.astype(float), ["intercept"] + list(X.columns[1:]), "ok"


def daily_ic_table(
    factor_returns: pd.DataFrame,
    horizons: Iterable[int],
    min_cross_section: int = 20,
) -> pd.DataFrame:
    """Compute the per-day cross-sectional IC table.

    Parameters
    ----------
    factor_returns
        Long DataFrame with columns: date, asset, factor_value, fwd_ret_<h>.
    horizons
        Horizons to evaluate.
    min_cross_section
        Minimum non-NaN observations per day; below this, IC is NaN.

    Returns
    -------
    pd.DataFrame
        Columns: date, factor_name, horizon, n_obs, pearson_ic, rank_ic.
    """
    horizons = list(horizons)
    fwd_cols = [f"fwd_ret_{h}" for h in horizons if f"fwd_ret_{h}" in factor_returns.columns]
    if not fwd_cols:
        return pd.DataFrame()

    records: List[dict] = []
    # group by date; ensure factor_value is numeric
    for date, sub in factor_returns.groupby("date"):
        fv = pd.to_numeric(sub["factor_value"], errors="coerce")
        for col in fwd_cols:
            ret = pd.to_numeric(sub[col], errors="coerce")
            mask = fv.notna() & ret.notna()
            n = int(mask.sum())
            if n < min_cross_section:
                rec = {
                    "date": date,
                    "n_obs": n,
                    "pearson_ic": np.nan,
                    "rank_ic": np.nan,
                    "fwd_col": col,
                }
            else:
                x = fv[mask].to_numpy(dtype=float)
                y = ret[mask].to_numpy(dtype=float)
                p, _ = _safe_pearson(x, y)
                s, _ = _safe_spearman(x, y)
                rec = {
                    "date": date,
                    "n_obs": n,
                    "pearson_ic": p,
                    "rank_ic": s,
                    "fwd_col": col,
                }
            records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["horizon"] = out["fwd_col"].str.replace("fwd_ret_", "").astype(int)
    out["factor_name"] = factor_returns["factor_name"].iloc[0] if "factor_name" in factor_returns.columns else "unknown"
    out = out.drop(columns=["fwd_col"])
    return out


def daily_partial_ic(
    factor_returns: pd.DataFrame,
    controls: pd.DataFrame,
    horizons: Iterable[int],
    min_cross_section: int = 20,
    controls_to_use: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute per-day partial IC (theme-controlled).

    This implementation is vectorized: it pivots the per-day panel into a wide
    matrix of (date x asset) and runs a single lstsq per day instead of one
    per (date, horizon). The same residualized factor is correlated with each
    horizon's residualized return, since controls do not depend on the horizon.
    """
    horizons = list(horizons)
    fwd_cols = [f"fwd_ret_{h}" for h in horizons if f"fwd_ret_{h}" in factor_returns.columns]
    if not fwd_cols:
        return pd.DataFrame()
    if controls_to_use is None:
        controls_to_use = [
            "theme_exposure",
            "sub_industry",
            "log_market_cap",
            "beta",
            "liquidity",
            "price_momentum",
        ]
    available = [c for c in controls_to_use if c in controls.columns]
    if not available:
        return pd.DataFrame(
            columns=["date", "horizon", "n_obs", "partial_ic", "controls_used", "reason"]
        )

    fr = factor_returns.copy()
    fr["date"] = pd.to_datetime(fr["date"]).dt.normalize()
    fr["asset"] = fr["asset"].astype(str)
    ctrl = controls.copy()
    ctrl["date"] = pd.to_datetime(ctrl["date"]).dt.normalize()
    ctrl["asset"] = ctrl["asset"].astype(str)

    records: List[dict] = []
    # Group by date once
    fac_by_date = {d: g for d, g in fr.groupby("date")}
    ctrl_by_date = {d: g for d, g in ctrl.groupby("date")}
    for date, fac_sub in fac_by_date.items():
        if date not in ctrl_by_date:
            continue
        csub = ctrl_by_date[date]
        merged = fac_sub.merge(csub, on=["asset"], how="inner", suffixes=("", "_ctrl"))
        if merged.empty:
            continue
        fv = pd.to_numeric(merged["factor_value"], errors="coerce")
        # Build per-day control matrix
        ctrl_df = merged[available].copy()
        # Standardize numeric, one-hot categorical
        parts = []
        used_cols = []
        for c in available:
            s = ctrl_df[c]
            if pd.api.types.is_numeric_dtype(s):
                filled = s.fillna(s.median())
                std = _cross_section_standardize(filled)
                if std.std() > EPS:
                    parts.append(std.rename(c))
                    used_cols.append(c)
            else:
                filled = s.fillna("__missing__").astype(str)
                dummies = pd.get_dummies(filled, prefix=c, drop_first=True, dummy_na=False)
                if dummies.shape[1] > 0:
                    parts.append(dummies)
                    used_cols.extend(list(dummies.columns))
        if not parts:
            for col in fwd_cols:
                records.append(
                    {
                        "date": date,
                        "horizon": int(col.replace("fwd_ret_", "")),
                        "n_obs": len(merged),
                        "partial_ic": np.nan,
                        "controls_used": "",
                        "reason": "no_valid_controls",
                    }
                )
            continue
        X = pd.concat(parts, axis=1)
        keep = X.std() > EPS
        X = X.loc[:, keep]
        if X.shape[1] == 0:
            for col in fwd_cols:
                records.append(
                    {
                        "date": date,
                        "horizon": int(col.replace("fwd_ret_", "")),
                        "n_obs": len(merged),
                        "partial_ic": np.nan,
                        "controls_used": "",
                        "reason": "controls_constant",
                    }
                )
            continue
        X.insert(0, "intercept", 1.0)
        Xv = X.to_numpy(dtype=float)
        n_obs = len(merged)
        if n_obs < min_cross_section or n_obs < Xv.shape[1] + 2:
            for col in fwd_cols:
                records.append(
                    {
                        "date": date,
                        "horizon": int(col.replace("fwd_ret_", "")),
                        "n_obs": n_obs,
                        "partial_ic": np.nan,
                        "controls_used": ",".join(["intercept"] + list(X.columns[1:])),
                        "reason": "insufficient_obs",
                    }
                )
            continue
        fv_arr = fv.to_numpy(dtype=float)
        ok_X = True
        try:
            rank = np.linalg.matrix_rank(Xv, tol=1e-8)
            if rank < Xv.shape[1]:
                ok_X = False
        except np.linalg.LinAlgError:
            ok_X = False
        if not ok_X:
            for col in fwd_cols:
                records.append(
                    {
                        "date": date,
                        "horizon": int(col.replace("fwd_ret_", "")),
                        "n_obs": n_obs,
                        "partial_ic": np.nan,
                        "controls_used": ",".join(["intercept"] + list(X.columns[1:])),
                        "reason": "rank_deficient",
                    }
                )
            continue
        # Residualize factor once
        try:
            beta_f, *_ = np.linalg.lstsq(Xv, fv_arr, rcond=None)
            resid_fac = fv_arr - Xv @ beta_f
        except np.linalg.LinAlgError:
            resid_fac = None
        for col in fwd_cols:
            ret = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(fv_arr) & np.isfinite(ret)
            if mask.sum() < min_cross_section:
                records.append(
                    {
                        "date": date,
                        "horizon": int(col.replace("fwd_ret_", "")),
                        "n_obs": int(mask.sum()),
                        "partial_ic": np.nan,
                        "controls_used": ",".join(["intercept"] + list(X.columns[1:])),
                        "reason": "insufficient_obs",
                    }
                )
                continue
            try:
                beta_r, *_ = np.linalg.lstsq(Xv[mask], ret[mask], rcond=None)
                resid_ret = ret[mask] - Xv[mask] @ beta_r
            except np.linalg.LinAlgError:
                records.append(
                    {
                        "date": date,
                        "horizon": int(col.replace("fwd_ret_", "")),
                        "n_obs": int(mask.sum()),
                        "partial_ic": np.nan,
                        "controls_used": ",".join(["intercept"] + list(X.columns[1:])),
                        "reason": "rank_deficient",
                    }
                )
                continue
            if resid_fac is None:
                records.append(
                    {
                        "date": date,
                        "horizon": int(col.replace("fwd_ret_", "")),
                        "n_obs": int(mask.sum()),
                        "partial_ic": np.nan,
                        "controls_used": ",".join(["intercept"] + list(X.columns[1:])),
                        "reason": "rank_deficient",
                    }
                )
                continue
            rf = resid_fac[mask]
            r, _ = _safe_pearson(rf, resid_ret)
            records.append(
                {
                    "date": date,
                    "horizon": int(col.replace("fwd_ret_", "")),
                    "n_obs": int(mask.sum()),
                    "partial_ic": r,
                    "controls_used": ",".join(["intercept"] + list(X.columns[1:])),
                    "reason": "ok",
                }
            )
    return pd.DataFrame(records)


def aggregate_ic(
    daily: pd.DataFrame,
    partial: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate daily IC into a per-(factor, horizon, period) summary."""
    if daily.empty:
        return pd.DataFrame()
    if "universe" not in daily.columns:
        daily = daily.copy()
        daily["universe"] = "all"
    if partial is not None and not partial.empty and "universe" not in partial.columns:
        partial = partial.copy()
        partial["universe"] = "all"
    grp = daily.groupby(["factor_name", "horizon", "period", "universe"], dropna=False)
    rows: List[dict] = []
    for (fac, hz, period, univ), sub in grp:
        ic = sub["pearson_ic"].dropna()
        ric = sub["rank_ic"].dropna()
        n = len(ic)
        if n < 2:
            rows.append(
                {
                    "factor_name": fac,
                    "horizon": hz,
                    "period": period,
                    "universe": univ,
                    "n_days": n,
                    "ic_mean": np.nan,
                    "ic_std": np.nan,
                    "icir": np.nan,
                    "rank_ic_mean": np.nan,
                    "pos_ic_ratio": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            continue
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=1))
        icir = ic_mean / ic_std if ic_std > EPS else np.nan
        rank_mean = float(ric.mean()) if len(ric) else np.nan
        pos_ratio = float((ic > 0).mean())
        # t-stat for "mean IC != 0" using n observations
        t_stat = ic_mean / (ic_std / math.sqrt(n)) if ic_std > EPS else np.nan
        # two-sided p-value
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))) if np.isfinite(t_stat) else np.nan
        # 95% CI for mean
        se = ic_std / math.sqrt(n)
        ci_low = ic_mean - 1.96 * se
        ci_high = ic_mean + 1.96 * se
        rows.append(
            {
                "factor_name": fac,
                "horizon": hz,
                "period": period,
                "universe": univ,
                "n_days": n,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "rank_ic_mean": rank_mean,
                "pos_ic_ratio": pos_ratio,
                "t_stat": t_stat,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Initialize partial IC columns on out
    for col in [
        "partial_ic_mean", "partial_ic_std", "partial_icir", "partial_t_stat",
        "partial_p_value", "partial_ci_low", "partial_ci_high",
    ]:
        out[col] = np.nan
    if partial is not None and not partial.empty and "universe" not in partial.columns:
        partial = partial.copy()
        partial["universe"] = "all"
    if partial is not None and not partial.empty:
        p_grp = partial.groupby(["factor_name", "horizon", "period", "universe"], dropna=False)
        for (fac, hz, period, univ), sub in p_grp:
            pic = sub["partial_ic"].dropna()
            n = len(pic)
            if n < 2:
                continue
            m = float(pic.mean())
            s = float(pic.std(ddof=1))
            icir = m / s if s > EPS else np.nan
            t = m / (s / math.sqrt(n)) if s > EPS else np.nan
            p_value = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if np.isfinite(t) else np.nan
            se = s / math.sqrt(n)
            rows_idx = out.index[
                (out["factor_name"] == fac)
                & (out["horizon"] == hz)
                & (out["period"] == period)
                & (out["universe"] == univ)
            ]
            ci_l, ci_h = m - 1.96 * se, m + 1.96 * se
            if len(rows_idx):
                out.loc[rows_idx, "partial_ic_mean"] = m
                out.loc[rows_idx, "partial_ic_std"] = s
                out.loc[rows_idx, "partial_icir"] = icir
                out.loc[rows_idx, "partial_t_stat"] = t
                out.loc[rows_idx, "partial_p_value"] = p_value
                out.loc[rows_idx, "partial_ci_low"] = ci_l
                out.loc[rows_idx, "partial_ci_high"] = ci_h
    return out


def run_ic_pipeline(
    factors_long: pd.DataFrame,
    forward_returns: pd.DataFrame,
    controls_table: pd.DataFrame,
    horizons: List[int],
    event_date: Optional[pd.Timestamp],
    post_event_window: int = 120,
    min_cross_section: int = 20,
    universe: pd.DataFrame = None,
    controls_to_use: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the full IC pipeline: daily raw IC, daily partial IC, aggregated summary.

    Returns a dict with keys:
        ``daily_raw``     : per-date per-horizon raw IC (and rank IC)
        ``daily_partial`` : per-date per-horizon theme-controlled partial IC
        ``summary``       : aggregated summary (per factor × horizon × period × universe)
    """
    from .data_loader import get_universe_mask, restrict_to_universe, split_periods

    if universe is None:
        universe = get_universe_mask(forward_returns, mode="all")
    factor_names = sorted(factors_long["factor_name"].dropna().unique().tolist())

    universe_name = "all"
    if universe is not None and not universe.empty and "in_universe" in universe.columns:
        n_total = len(universe)
        n_in = int(universe["in_universe"].sum())
        if n_total > 0 and n_in / n_total < 0.999:
            # Heuristic: if mask is mostly True, name it "all"; otherwise use first non-default
            universe_name = "custom"

    # Map factor values to per-date per-asset frame (long)
    fv = factors_long[["date", "asset", "factor_name", "factor_value"]].copy()
    fv["date"] = pd.to_datetime(fv["date"]).dt.normalize()
    fv["asset"] = fv["asset"].astype(str)
    fr = forward_returns.copy()
    fr["date"] = pd.to_datetime(fr["date"]).dt.normalize()
    fr["asset"] = fr["asset"].astype(str)
    if universe is not None and not universe.empty and "in_universe" in universe.columns:
        u = universe.copy()
        u["date"] = pd.to_datetime(u["date"]).dt.normalize()
        u["asset"] = u["asset"].astype(str)
        fv = fv.merge(u[["date", "asset", "in_universe"]], on=["date", "asset"], how="inner")
        fr = fr.merge(u[["date", "asset", "in_universe"]], on=["date", "asset"], how="inner")

    # Period split
    all_dates = fr["date"].drop_duplicates().sort_values().tolist()
    periods = split_periods(all_dates, event_date, post_event_window=post_event_window)

    # Build per-factor alignment & run
    daily_raw_records: List[pd.DataFrame] = []
    daily_partial_records: List[pd.DataFrame] = []
    for fac in factor_names:
        fac_long = fv.loc[fv["factor_name"] == fac, ["date", "asset", "factor_value"]]
        if fac_long.empty:
            continue
        merged = fac_long.merge(fr, on=["date", "asset"], how="inner")
        # Need at least min_cross_section per day after merge
        counts = merged.groupby("date").size()
        good_dates = counts[counts >= min_cross_section].index
        merged = merged[merged["date"].isin(good_dates)].copy()
        if merged.empty:
            continue
        # daily raw IC
        d = daily_ic_table(
            merged[["date", "asset", "factor_value"] + [c for c in merged.columns if c.startswith("fwd_ret_")]].assign(
                factor_name=fac
            ),
            horizons=horizons,
            min_cross_section=min_cross_section,
        )
        if not d.empty:
            d = d.assign(universe=universe_name)
            daily_raw_records.append(d)
        # daily partial IC
        ctrl_use = controls_to_use if controls_to_use is not None else None
        p = daily_partial_ic(
            merged[["date", "asset", "factor_value"] + [c for c in merged.columns if c.startswith("fwd_ret_")]].assign(
                factor_name=fac
            ),
            controls=controls_table,
            horizons=horizons,
            min_cross_section=min_cross_section,
            controls_to_use=ctrl_use,
        )
        if not p.empty:
            p["factor_name"] = fac
            p["universe"] = universe_name
            daily_partial_records.append(p)

    if daily_raw_records:
        daily_raw = pd.concat(daily_raw_records, ignore_index=True)
    else:
        daily_raw = pd.DataFrame()
    if daily_partial_records:
        daily_partial = pd.concat(daily_partial_records, ignore_index=True)
    else:
        daily_partial = pd.DataFrame()

    # Add period column
    def _period_for(d):
        ts = pd.to_datetime(d)
        if event_date is None:
            return "all"
        ev = pd.to_datetime(event_date).normalize()
        if ts < ev:
            return "pre_event"
        if ts <= ev + pd.tseries.offsets.BDay(post_event_window):
            return "post_event"
        return "post_event_extended"

    if not daily_raw.empty:
        daily_raw["period"] = daily_raw["date"].apply(_period_for)
    if not daily_partial.empty:
        daily_partial["period"] = daily_partial["date"].apply(_period_for)

    summary = aggregate_ic(daily_raw, daily_partial)
    return {"daily_raw": daily_raw, "daily_partial": daily_partial, "summary": summary}
