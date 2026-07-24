"""Rebuild docs/data/results.json from the latest outputs/ CSVs.

Keeps the website's dynamic tables and stat cards in sync with the
pipeline. Reads:
    outputs/factor_horizon_summary_best.csv
    outputs/portfolio_summary.csv
    outputs/factor_half_life.csv
    outputs/data_note  (set in run_demo.py: data_note param)
    outputs/event_date (set in run_demo.py: event_date param)

Writes:
    docs/data/results.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

OUT = Path("outputs")
DEST = Path("docs/data/results.json")
DEST.parent.mkdir(parents=True, exist_ok=True)


def _finite(v):
    """Treat empty/NaN as None so JSON serializes cleanly."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None", "inf", "-inf"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _row_to_json(row: pd.Series, columns: list[str]) -> dict:
    out = {}
    for c in columns:
        v = row.get(c)
        if c in (
            "ic_mean", "ic_std", "rank_ic_mean", "t_stat", "p_value",
            "ci_low", "ci_high", "icir", "pos_ic_ratio",
            "partial_ic_mean", "partial_ic_std", "partial_icir",
            "partial_t_stat", "partial_p_value",
            "partial_ci_low", "partial_ci_high",
            "peak_abs_ic", "half_life_horizon", "plateau_horizon",
            "mean_return", "cum_return", "volatility", "max_drawdown",
            "turnover", "transaction_cost", "net_return", "annualized_return",
            "sharpe", "sharpe_top", "ann_return_gross", "ann_vol", "sharpe_gross",
            "max_drawdown", "top_quantile_annualized",
        ):
            out[c] = _finite(v)
        else:
            out[c] = None if v is None or (isinstance(v, float) and math.isnan(v)) else v
    return out


def main():
    # --- factors (best per factor) ---
    best = pd.read_csv(OUT / "factor_horizon_summary_best.csv")
    # Filter to theme_diffused + post_event (the most informative slice)
    factors_df = best[(best["universe"] == "theme_diffused") & (best["period"] == "post_event")].copy()
    factors_df = factors_df.sort_values("factor_name")
    factor_cols = [
        "factor_name", "horizon", "period", "universe", "n_days",
        "ic_mean", "ic_std", "icir", "rank_ic_mean", "pos_ic_ratio",
        "t_stat", "p_value", "ci_low", "ci_high",
        "partial_ic_mean", "partial_ic_std", "partial_icir",
        "partial_t_stat", "partial_p_value", "partial_ci_low", "partial_ci_high",
    ]
    factors = [_row_to_json(r, factor_cols) for _, r in factors_df.iterrows()]

    # --- portfolios ---
    port = pd.read_csv(OUT / "portfolio_summary.csv")
    port_cols = [
        "portfolio", "factor_name", "n_periods",
        "cum_return_gross", "ann_return_gross", "ann_vol",
        "sharpe_gross", "max_drawdown", "turnover",
        "transaction_cost", "net_return",
    ]
    portfolios = [_row_to_json(r, port_cols) for _, r in port.iterrows()]

    # --- half-life ---
    hl = pd.read_csv(OUT / "factor_half_life.csv")
    hl_cols = [
        "factor_name", "period", "universe",
        "peak_horizon", "peak_abs_ic",
        "half_life_horizon", "plateau_horizon",
        "recommended_rebalance_frequency", "curve_monotonicity", "reliability_note",
    ]
    half_life = [_row_to_json(r, hl_cols) for _, r in hl.iterrows()]

    # --- summary metrics ---
    n_assets = 80
    n_days = 300
    n_factors = int(factors_df["factor_name"].nunique())
    # n_survivors: |partial_ic| > 0.02 AND |partial_t_stat| > 2.5
    def _is_survivor(r):
        p = _finite(r.get("partial_ic_mean"))
        t = _finite(r.get("partial_t_stat"))
        ic = _finite(r.get("ic_mean"))
        if p is None or t is None or ic is None:
            return False
        return abs(ic) > 0.05 and abs(p) > 0.02 and abs(t) > 2.5
    n_survivors = int(factors_df.apply(_is_survivor, axis=1).sum())

    # event_date & data_note from existing JSON
    existing: dict = {}
    if DEST.exists():
        try:
            with DEST.open() as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    payload = {
        "factors": factors,
        "portfolios": portfolios,
        "half_life": half_life,
        "event_date": existing.get("event_date", "2024-07-02"),
        "data_note": existing.get(
            "data_note",
            "synthetic test data (n=80 assets, t=300 days, seed=42)",
        ),
        "summary_metrics": {
            "n_assets": n_assets,
            "n_days": n_days,
            "n_factors": n_factors,
            "core_robot_share": existing.get("summary_metrics", {}).get("core_robot_share", 0.275),
            "diffused_share": existing.get("summary_metrics", {}).get("diffused_share", 1.0),
            "n_survivors": n_survivors,
        },
    }
    with DEST.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✓ wrote {DEST}  ({len(factors)} factors, {len(portfolios)} portfolios, {n_survivors} survivors)")


if __name__ == "__main__":
    main()
