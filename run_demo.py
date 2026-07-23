"""End-to-end demo runner for the WAIC factor IC research module.

Generates synthetic data (clearly marked) and runs the full pipeline.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.synthetic_data import generate_synthetic_universe
from src.pipeline import run_pipeline


def main():
    out_dir = str(Path(__file__).resolve().parent / "outputs")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WAIC 之后具身智能股票池因子 IC 分析 - Demo")
    print("=" * 70)
    print()
    print("生成合成数据(n=80 assets, t=300 days, seed=42)...")
    t0 = time.time()
    prices, factors, event = generate_synthetic_universe(
        n_assets=80, n_days=300, seed=42
    )
    print(f"  数据生成耗时 {time.time()-t0:.1f}s")
    print(f"  价格表: {prices.shape}")
    print(f"  因子表: {factors.shape}")
    print(f"  事件日: {event.date()}")
    print(f"  因子列表: {sorted(factors['factor_name'].unique().tolist())}")
    print()

    print("运行因子 IC pipeline...")
    t1 = time.time()
    res = run_pipeline(
        prices=prices,
        factors=factors,
        event_date=event,
        output_dir=out_dir,
        horizons=[1, 2, 3, 5, 10, 20, 40, 60],
        rebalance_frequencies=[1, 5, 10, 20],
        quantiles=5,
        min_cross_section=20,
        transaction_cost_bps=10,
        post_event_window=120,
        data_note="synthetic test data (n=80 assets, t=300 days, seed=42)",
        verbose=False,
    )
    print(f"  Pipeline 耗时 {time.time()-t1:.1f}s")
    print()
    print("=" * 70)
    print("输出文件:")
    print("=" * 70)
    for f in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, f)
        if os.path.isfile(full):
            sz = os.path.getsize(full)
            print(f"  {f}: {sz} bytes")
        else:
            print(f"  {f}/")
            for sub in sorted(os.listdir(full)):
                print(f"    {sub}")
    print()
    print(f"完整报告: {res['report_path']}")


if __name__ == "__main__":
    main()
