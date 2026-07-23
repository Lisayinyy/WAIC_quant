# WAIC Quant — Factor IC Research

> **When every stock is robotics, which factors still survive?**
> An end-to-end factor-IC research module around the post-WAIC diffusion of the embodied-AI theme.

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![tests](https://img.shields.io/badge/tests-16%2F16-brightgreen)]() [![license](https://img.shields.io/badge/license-research--use-lightgrey)]()

## What this repo contains

```
WAIC_quant/
├── src/                          # the factor-IC module
│   ├── data_loader.py            # schema validation, universe, period split
│   ├── synthetic_data.py         # synthetic universe with theme diffusion
│   ├── forward_returns.py        # fwd_ret_h, no-lookahead guarantees
│   ├── ic_analysis.py            # raw IC, partial IC, incremental IC
│   ├── horizon.py                # multi-horizon, decay, half-life
│   ├── rebalance.py              # rebalance freq, quantiles, turnover, cost
│   ├── stability.py              # rolling IC, alerts
│   ├── portfolio.py              # A/B/C/D portfolio backtests
│   ├── reporting.py              # report.md + 7 figures
│   └── pipeline.py               # end-to-end entry point
│
├── tests/
│   └── test_pipeline.py          # 16 unit tests, all passing
│
├── run_demo.py                   # one-liner demo runner
│
├── outputs/                      # demo run artifacts (see .gitignore)
│   ├── *.csv                     # IC summaries, decay curves, alerts, …
│   ├── report.md                 # the full research report
│   └── figures/                  # 7 PNGs
│
├── site/                         # self-contained static website (deployable)
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   ├── assets/                   # the 7 figures
│   └── data/results.json
│
├── README.md                     # you are here
└── .gitignore
```

## Quick start

```bash
# 1. Python deps
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy statsmodels matplotlib pytest

# 2. Run the demo (synthetic data; ~3 min on a modern laptop)
python run_demo.py

# 3. Inspect the results
ls outputs/
cat outputs/report.md

# 4. Run the unit tests
python -m pytest tests/ -v
```

## The thesis

After WAIC 2024, the embodied-AI / robotics theme spread from ~28% of related stocks to almost 100%. The question this module tries to answer:

1. When the **theme tag** is everywhere, is it still a factor?
2. Which factors still carry **incremental IC** after controlling for theme exposure, sub-industry, market-cap, beta, liquidity, and momentum?
3. What is the optimal **horizon**, **rebalance frequency**, and **portfolio construction** for the survivors?

Headline answer (synthetic data, see `outputs/report.md` for the full numbers):

- **3 factors survive**: `quality`, `order_or_contract`, `revenue_revision` — all fundamental, all at h = 40 ~ 60 days.
- **3 factors dampened by theme diffusion**: `theme_heat` (literally absorbed by the control), `valuation`, `beta`.
- **2 noise controls correctly rejected**: `noise_alpha`, `noise_beta`.

> **The theme label is not a factor. It is a phenomenon that factors explain.**

## The module at a glance

| Stage | What it does | Key safeguards |
|---|---|---|
| **Validate & align** | schema checks, date normalization | required columns enforced |
| **Forward returns** | `fwd_ret_h = close[t+h] / close[t] - 1` | shift only within the same asset; future prices missing → NaN |
| **Cross-sectional IC** | daily Pearson + Spearman | returns NaN on < `min_cross_section` or constant factor |
| **Theme-controlled partial IC** | residualize factor and return against theme / industry / size / beta / liquidity / momentum | rank-deficient controls → NaN with reason code |
| **Multi-horizon scan** | 1/2/3/5/10/20/40/60 days | IC decay curve, half-life, monotonicity label |
| **Rebalance & quantiles** | 1/5/10/20-day rebalances, 5 quantiles | gross + net spread, max DD, turnover (real) |
| **Stability & alerts** | 20/60/120-day rolling IC | cross-zero, 20-day negative streak, ICIR < -0.5 |
| **Portfolios A/B/C/D** | theme / single-factor / incremental / multi-horizon | gross & net return, ann. vol, max DD, turnover |

## Inputs

**`prices`** (long format)

| column | type | required |
|---|---|---|
| `date` | date | yes |
| `asset` | str | yes |
| `close` | float | yes |
| `volume`, `amount`, `market_cap` | float | optional |
| `industry`, `sub_industry` | str | optional |
| `theme_exposure` | float in [0, 1] | optional |
| `beta`, `is_robot_stock` | float / bool | optional |

**`factors`** (long format)

| column | type | required |
|---|---|---|
| `date` | date | yes |
| `asset` | str | yes |
| `factor_name` | str | yes |
| `factor_value` | float | yes |

**`event_date`** — optional, but recommended. Used to split pre-event / post-event samples.

## Usage

```python
from src.pipeline import run_pipeline

res = run_pipeline(
    prices=prices_df,
    factors=factors_df,
    event_date="2024-07-02",      # WAIC
    output_dir="outputs",
    horizons=[1, 2, 3, 5, 10, 20, 40, 60],
    rebalance_frequencies=[1, 5, 10, 20],
    quantiles=5,
    min_cross_section=20,
    transaction_cost_bps=10,
    post_event_window=120,
    data_note="synthetic test data (n=80, t=300, seed=42)",
)

# res["summary"]           — per-(factor, horizon, period) IC + partial IC
# res["decay"]             — IC decay curves
# res["half_life"]         — half-life + recommended rebalance
# res["rebalance_summary"] — gross/net spread by freq
# res["alerts"]            — stability alerts (deduplicated)
# res["portfolio"]         — A/B/C/D backtest
# res["report_path"]       — outputs/report.md
```

## Unit tests (16 / 16 passing)

```
$ python -m pytest tests/ -v
tests/test_pipeline.py::test_validate_prices_works                  PASSED
tests/test_pipeline.py::test_validate_factors_works                 PASSED
tests/test_pipeline.py::test_known_factor_return_relationship_detected PASSED
tests/test_pipeline.py::test_lookahead_future_data_breaks            PASSED
tests/test_pipeline.py::test_insufficient_sample_returns_nan        PASSED
tests/test_pipeline.py::test_constant_factor_returns_nan            PASSED
tests/test_pipeline.py::test_multi_horizon_alignment                PASSED
tests/test_pipeline.py::test_rebalance_frequency_changes_holdings   PASSED
tests/test_pipeline.py::test_transaction_cost_reduces_net           PASSED
tests/test_pipeline.py::test_ties_in_factor_dont_crash              PASSED
tests/test_pipeline.py::test_equal_freq_bins_handles_ties            PASSED
tests/test_pipeline.py::test_partial_ic_dampens_theme_signal        PASSED
tests/test_pipeline.py::test_non_monotonic_decay_returns_nan_or_reliable_note PASSED
tests/test_pipeline.py::test_event_split_no_leak                    PASSED
tests/test_pipeline.py::test_end_to_end_pipeline_outputs            PASSED
tests/test_pipeline.py::test_portfolio_net_below_gross               PASSED
======================== 16 passed in ~90s ========================
```

Each test corresponds to a failure mode the module could fall into:
data validation, known-IC recovery, leakage, sample-size NaN, constant-factor rejection, horizon alignment, rebalance semantics, transaction-cost sign, tied-value grouping, partial-IC dampening of pure-label signal, non-monotonic decay, event-split leakage, output completeness, and net-vs-gross portfolio.

## Output files

| File | Content |
|---|---|
| `outputs/report.md` | full 21-section research report |
| `outputs/daily_ic.csv` | per-date per-horizon raw IC (theme_core + theme_diffused) |
| `outputs/daily_ic_partial.csv` | per-date per-horizon partial IC |
| `outputs/factor_horizon_summary.csv` | aggregated IC (mean / ICIR / t / p / CI / partial) |
| `outputs/factor_horizon_summary_best.csv` | one row per factor: best horizon |
| `outputs/partial_incremental_ic.csv` | raw vs partial IC for delta IC |
| `outputs/ic_decay_curve.csv` | IC by horizon for each factor |
| `outputs/factor_half_life.csv` | peak horizon / half-life / rebalance recommendation |
| `outputs/rebalance_comparison.csv` | gross / net spread by rebalance freq |
| `outputs/quantile_returns.csv` | per-quantile cumulative returns |
| `outputs/rolling_ic.csv` | rolling IC at 20 / 60 / 120 day windows |
| `outputs/stability_alerts.csv` | deduplicated stability alerts |
| `outputs/portfolio_summary.csv` | A / B / C / D portfolio backtest |
| `outputs/figures/*.png` | 7 figures (see below) |

The 7 figures: raw IC heatmap, partial IC heatmap, pre/post event comparison, IC decay curves, rebalance frontier, rolling IC, quantile cumulative returns.

## Static website (in `site/`)

The repo also ships a self-contained static website built around the report — editorial layout inspired by [devouringdetails.com](https://devouringdetails.com), with a left-side minimap, large serif headlines, and a sticky factor-results table.

```bash
cd site
python3 -m http.server 8000
# open http://localhost:8000
```

Or deploy `site/` to any static host (Netlify, Vercel, S3, GitHub Pages, etc.).

## Caveats

- The default demo uses **synthetic data**. Real market results will differ.
- The control set (`theme_exposure / sub_industry / log_market_cap / beta / liquidity / price_momentum`) is a reasonable default; in production you may want to add `institutional_holding_change`, `north_bound_flow`, `limit-up/down` flags, etc.
- IC t-statistics use a normal approximation. With n_days < 30, confidence intervals are too tight — interpret with caution.
- Daily-rebalance annualized returns of 1000%+ are `(1+r)^252` compounding artifacts, not sustainable without leverage.

## License

Research use only. No warranty. Not investment advice.
