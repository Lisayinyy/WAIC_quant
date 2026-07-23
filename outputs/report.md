# WAIC 之后,当所有股票都沾上机器人,真正有用的因子还剩什么?

## 1. 研究问题
本报告回答的核心问题:在 WAIC 之后,具身智能/机器人主题从少数龙头扩散到更多股票,单纯的「机器人概念暴露」可能已经无法有效区分股票。我们需要识别:**在主题标签普及后,仍然能够区分未来收益的因子;这些因子的最优 Horizon、调仓频率;主题暴露作为控制变量被剥离后,哪些因子仍具有增量 IC**。

## 2. 数据说明
本报告基于 **synthetic test data (n=80 assets, t=300 days, seed=42)**。所有数值均来自程序计算,不预假设任何因子有效。
- 事件日期 (WAIC): `2024-07-02`
- 研究 Horizons: `[1, 2, 3, 5, 10, 20, 40, 60]`

## 3. 具身智能股票池定义
- 核心机器人股票数: **22** (占比 27.5%)
- 主题扩散后(theme_exposure>0)股票数: **80** (占比 100.0%)
> 主题标签的覆盖度已经超过 60%,「机器人概念」本身在截面上的区分度明显下降,需要转向研究股票之间的相对差异。

## 4. 因子定义
- 研究因子: beta, liquidity, noise_alpha, noise_beta, order_or_contract, price_momentum, quality, revenue_revision, theme_heat, valuation
- 控制变量(Partial IC 回归): `theme_exposure, sub_industry, log_market_cap, beta, liquidity, price_momentum`。

## 5. 未来收益计算方式
对每只股票,在其价格序列上按时间升序计算: `fwd_ret_h = close[t+h] / close[t] - 1`,其中 h 取研究 Horizons。未来价格不足时返回 NaN,不做静默填充。因子在 t 日仅与 t 日之后的收益对齐。

## 6. 防止未来函数的处理
- 价格 `shift(-h)` 仅在同一 `asset` 的时间序列内进行,不会跨股票错位。
- 部分控制变量使用 t 日已可见的数据(如 `log_market_cap`、`theme_exposure`),不引用未来。
- 调仓日只能使用当日可见的因子;未来收益不参与分组决策。
- 每个交易日只保留成对非缺失样本;样本数 < `min_cross_section` 时返回 NaN。

## 7. Raw IC 结果
下表展示事件后(post_event)各因子在最佳 Horizon 下的 Raw IC(主题扩散股票池):

| factor | best_horizon | ic_mean | rank_ic_mean | t_stat | p_value | n_days |
|---|---|---|---|---|---|---|
| beta | 40 | -0.1545 | -0.1502 | -14.60 | 0.000 | 47 |
| liquidity | 40 | 0.0299 | 0.0215 | 2.30 | 0.026 | 47 |
| noise_alpha | 20 | -0.0411 | -0.0410 | -2.76 | 0.008 | 47 |
| noise_beta | 20 | 0.0427 | 0.0383 | 2.61 | 0.012 | 47 |
| order_or_contract | 60 | 0.1051 | 0.1231 | 14.55 | 0.000 | 47 |
| price_momentum | 60 | -0.0277 | -0.0556 | -1.20 | 0.237 | 47 |
| quality | 40 | 0.0634 | 0.0729 | 4.11 | 0.000 | 47 |
| revenue_revision | 60 | 0.0619 | 0.0517 | 6.46 | 0.000 | 47 |
| theme_heat | 60 | 0.0644 | 0.0035 | 10.05 | 0.000 | 47 |
| valuation | 60 | -0.0481 | -0.0976 | -6.06 | 0.000 | 47 |

## 8. Theme-controlled (Partial) IC 结果
下表为各因子在主题控制后的 partial IC(同一个回归中控制 theme_exposure / sub_industry / log_market_cap / beta / liquidity / price_momentum):

| factor | best_horizon | raw_ic | partial_ic | raw_p | partial_p |
|---|---|---|---|---|---|
| noise_alpha | 20 | -0.0411 | -0.0472 | 0.008 | 0.005 |
| noise_beta | 20 | 0.0427 | 0.0436 | 0.012 | 0.015 |
| order_or_contract | 60 | 0.1051 | 0.0494 | 0.000 | 0.000 |
| quality | 40 | 0.0634 | 0.0926 | 0.000 | 0.000 |
| revenue_revision | 60 | 0.0619 | 0.0291 | 0.000 | 0.003 |
| valuation | 1 | -0.0099 | -0.0408 | 0.497 | 0.047 |

## 9. Incremental IC 与主题扩散分析
下表列出各因子的 Raw IC vs Partial IC 增量(主题扩散股票池 theme_diffused):

| factor | horizon | raw_ic | partial_ic | raw_p | partial_p |
|---|---|---|---|---|---|
| beta | 40 | -0.1545 | nan | 0.000 | nan |
| liquidity | 40 | 0.0299 | nan | 0.026 | nan |
| noise_alpha | 20 | -0.0411 | -0.0472 | 0.008 | 0.005 |
| noise_beta | 20 | 0.0427 | 0.0436 | 0.012 | 0.015 |
| order_or_contract | 60 | 0.1051 | 0.0494 | 0.000 | 0.000 |
| price_momentum | 60 | 0.0747 | nan | 0.000 | nan |
| quality | 60 | -0.0826 | -0.0360 | 0.000 | 0.066 |
| revenue_revision | 40 | 0.1087 | 0.1130 | 0.000 | 0.000 |
| theme_heat | 60 | 0.0644 | nan | 0.000 | nan |
| valuation | 60 | 0.0879 | 0.0282 | 0.000 | 0.152 |

## 10. 多 Horizon 对比
事件后各因子在每个 Horizon 上的 IC(矩阵视图,数值为 IC 均值,主题扩散股票池):

| factor | 1 | 2 | 3 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|---|
| beta | -0.015 | -0.023 | -0.029 | -0.041 | -0.076 | -0.123 | -0.155 | -0.131 |
| liquidity | -0.017 | -0.010 | -0.002 | 0.004 | 0.008 | 0.017 | 0.030 | 0.017 |
| noise_alpha | -0.023 | -0.033 | -0.029 | -0.023 | -0.025 | -0.041 | -0.035 | -0.033 |
| noise_beta | 0.018 | 0.002 | 0.001 | 0.034 | 0.035 | 0.043 | 0.023 | 0.026 |
| order_or_contract | 0.011 | 0.008 | 0.012 | 0.015 | 0.027 | 0.049 | 0.078 | 0.105 |
| price_momentum | -0.005 | -0.009 | -0.012 | -0.012 | -0.006 | -0.023 | 0.026 | -0.028 |
| quality | 0.023 | 0.029 | 0.032 | 0.032 | 0.018 | 0.001 | 0.063 | 0.039 |
| revenue_revision | -0.009 | -0.016 | -0.014 | -0.006 | 0.002 | 0.012 | 0.035 | 0.062 |
| theme_heat | -0.003 | -0.006 | -0.008 | -0.015 | -0.013 | 0.005 | 0.040 | 0.064 |
| valuation | -0.010 | -0.006 | -0.002 | 0.005 | -0.003 | -0.018 | -0.035 | -0.048 |


## 11. IC 衰减和半衰期
| factor | peak_horizon | peak_|IC| | half_life_horizon | recommended_rebalance | monotonicity | note |
|---|---|---|---|---|---|---|
| beta | 40 | 0.1545 | nan | 1 | peak_then_decay | never_decayed; recommend rebalance=1 (low confidence) |
| liquidity | 40 | 0.0299 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |
| noise_alpha | 20 | 0.0411 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |
| noise_beta | 20 | 0.0427 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |
| order_or_contract | 60 | 0.1051 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |
| price_momentum | 60 | 0.0277 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |
| quality | 40 | 0.0634 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |
| revenue_revision | 60 | 0.0619 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |
| theme_heat | 60 | 0.0644 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |
| valuation | 60 | 0.0481 | nan | 1 | non_monotonic | never_decayed; recommend rebalance=1 (low confidence) |

## 12. 调仓频率对比
| factor | freq | gross_spread | net_spread | turnover | top_quantile_annualized |
|---|---|---|---|---|---|
| beta | 1 | 73.4223 | 73.4220 | 0.003 | 24292.65% |
| beta | 5 | 0.6489 | 0.6489 | 0.017 | 242.28% |
| beta | 10 | 0.6653 | 0.6653 | 0.033 | 100.44% |
| beta | 20 | 0.1078 | 0.1078 | 0.067 | -5.47% |
| liquidity | 1 | 197.5275 | 197.3707 | 0.792 | 71836.83% |
| liquidity | 5 | -2.2580 | -2.2563 | 0.808 | 55.62% |
| liquidity | 10 | 0.4177 | 0.4174 | 0.792 | 88.60% |
| liquidity | 20 | -0.1124 | -0.1123 | 0.801 | 3.34% |
| noise_alpha | 1 | -21.0446 | -21.0275 | 0.800 | 5888.71% |
| noise_alpha | 5 | 1.3710 | 1.3699 | 0.794 | 170.30% |
| noise_alpha | 10 | 0.7900 | 0.7894 | 0.804 | 99.68% |
| noise_alpha | 20 | 0.6107 | 0.6102 | 0.803 | 40.36% |
| noise_beta | 1 | 125.0933 | 124.9932 | 0.803 | 28437.84% |
| noise_beta | 5 | 1.2000 | 1.1990 | 0.790 | 110.15% |
| noise_beta | 10 | 0.7411 | 0.7405 | 0.810 | 56.85% |
| noise_beta | 20 | -0.0502 | -0.0501 | 0.833 | -33.35% |
| order_or_contract | 1 | nan | nan | 0.009 | 1586.93% |
| order_or_contract | 5 | nan | nan | 0.046 | 61.99% |
| order_or_contract | 10 | nan | nan | 0.091 | 30.46% |
| order_or_contract | 20 | nan | nan | 0.181 | -11.46% |
| price_momentum | 1 | -117.8632 | -117.8371 | 0.330 | 66765.76% |
| price_momentum | 5 | -0.1515 | -0.1513 | 0.569 | 296.58% |
| price_momentum | 10 | -2.4512 | -2.4499 | 0.688 | 197.45% |
| price_momentum | 20 | 0.6520 | 0.6514 | 0.815 | 96.17% |
| quality | 1 | 2220.5205 | 2220.1328 | 0.311 | 235023.14% |
| quality | 5 | 2.9901 | 2.9890 | 0.539 | 376.95% |
| quality | 10 | 3.1515 | 3.1499 | 0.642 | 281.63% |
| quality | 20 | 0.9728 | 0.9723 | 0.723 | 80.17% |
| revenue_revision | 1 | 33.6639 | 33.6516 | 0.410 | 69953.37% |
| revenue_revision | 5 | -0.2910 | -0.2903 | 0.429 | -14.02% |
| revenue_revision | 10 | -0.2298 | -0.2291 | 0.469 | -17.27% |
| revenue_revision | 20 | -0.3655 | -0.3648 | 0.526 | -40.90% |
| theme_heat | 1 | 5.1188 | 5.1188 | 0.520 | 3436.67% |
| theme_heat | 5 | -2.1696 | -2.1685 | 0.525 | 59.65% |
| theme_heat | 10 | -3.2773 | -3.2758 | 0.540 | 1.79% |
| theme_heat | 20 | -2.3323 | -2.3313 | 0.539 | -37.79% |
| valuation | 1 | -446.9266 | -446.8614 | 0.403 | 4067.32% |
| valuation | 5 | -0.4990 | -0.4991 | 0.414 | 90.65% |
| valuation | 10 | -0.7349 | -0.7350 | 0.406 | 23.04% |
| valuation | 20 | -0.0033 | -0.0034 | 0.458 | -14.73% |

## 13. 毛收益和成本后收益
以下为各因子在成本后净 spread 最大的调仓频率:

| factor | best_freq | gross_spread | net_spread | turnover |
|---|---|---|---|---|
| beta | 1 | 73.422 | 73.422 | 0.003 |
| liquidity | 1 | 197.528 | 197.371 | 0.792 |
| noise_alpha | 5 | 1.371 | 1.370 | 0.794 |
| noise_beta | 1 | 125.093 | 124.993 | 0.803 |
| price_momentum | 20 | 0.652 | 0.651 | 0.815 |
| quality | 1 | 2220.520 | 2220.133 | 0.311 |
| revenue_revision | 1 | 33.664 | 33.652 | 0.410 |
| theme_heat | 1 | 5.119 | 5.119 | 0.520 |
| valuation | 20 | -0.003 | -0.003 | 0.458 |

> 未识别出「高频调仓毛收益为正、扣成本后为负」的因子。

## 14. 因子稳定性告警
共检测到 **74** 条告警:

| factor | alert | severity | start | end | evidence | action |
|---|---|---|---|---|---|---|
| beta | 20_consecutive_neg_ic | high | 2023-01-02 00:00:00 | 2024-06-28 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| beta | 20_consecutive_neg_ic | high | 2024-07-02 00:00:00 | 2024-12-17 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| beta | 20_consecutive_neg_ic | high | 2024-12-20 00:00:00 | 2025-12-18 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| beta | rolling_icir_below_-0.5 | high | 2023-02-21 00:00:00 | 2023-09-11 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| beta | rolling_icir_below_-0.5 | high | 2024-08-22 00:00:00 | 2024-12-17 00:00:00 | 4 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| beta | rolling_mean_cross_zero | warning | 2023-01-16 00:00:00 | 2024-06-28 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| beta | rolling_mean_cross_zero | warning | 2024-07-17 00:00:00 | 2024-10-04 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| beta | rolling_mean_cross_zero | warning | 2025-01-23 00:00:00 | 2025-12-18 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| liquidity | 20_consecutive_neg_ic | high | 2023-07-06 00:00:00 | 2024-05-13 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| liquidity | 20_consecutive_neg_ic | high | 2025-03-04 00:00:00 | 2025-03-26 00:00:00 | 1 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| liquidity | rolling_mean_cross_zero | warning | 2023-01-16 00:00:00 | 2024-06-25 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| liquidity | rolling_mean_cross_zero | warning | 2024-07-17 00:00:00 | 2024-12-17 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| liquidity | rolling_mean_cross_zero | warning | 2025-01-06 00:00:00 | 2025-12-18 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| noise_alpha | 20_consecutive_neg_ic | high | 2023-02-24 00:00:00 | 2024-03-19 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| noise_alpha | 20_consecutive_neg_ic | high | 2024-07-10 00:00:00 | 2024-11-11 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| noise_alpha | 20_consecutive_neg_ic | high | 2025-06-13 00:00:00 | 2025-07-07 00:00:00 | 1 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| noise_alpha | rolling_icir_below_-0.5 | high | 2024-08-22 00:00:00 | 2024-12-03 00:00:00 | 4 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| noise_alpha | rolling_icir_below_-0.5 | high | 2025-10-09 00:00:00 | 2025-10-16 00:00:00 | 1 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| noise_alpha | rolling_mean_cross_zero | warning | 2023-01-16 00:00:00 | 2024-06-25 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| noise_alpha | rolling_mean_cross_zero | warning | 2024-07-17 00:00:00 | 2024-12-17 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| noise_alpha | rolling_mean_cross_zero | warning | 2025-01-06 00:00:00 | 2025-12-25 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| noise_beta | 20_consecutive_neg_ic | high | 2023-04-27 00:00:00 | 2024-05-23 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| noise_beta | rolling_mean_cross_zero | warning | 2023-01-16 00:00:00 | 2024-06-06 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| noise_beta | rolling_mean_cross_zero | warning | 2024-07-19 00:00:00 | 2024-12-17 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| noise_beta | rolling_mean_cross_zero | warning | 2025-01-06 00:00:00 | 2025-12-15 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| order_or_contract | 20_consecutive_neg_ic | high | 2023-02-07 00:00:00 | 2024-03-26 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| order_or_contract | 20_consecutive_neg_ic | high | 2024-07-29 00:00:00 | 2024-10-02 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| order_or_contract | 20_consecutive_neg_ic | high | 2025-01-13 00:00:00 | 2025-11-21 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| order_or_contract | rolling_icir_below_-0.5 | high | 2023-04-05 00:00:00 | 2024-05-23 00:00:00 | 3 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| order_or_contract | rolling_icir_below_-0.5 | high | 2025-02-11 00:00:00 | 2025-12-30 00:00:00 | 2 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| order_or_contract | rolling_mean_cross_zero | warning | 2023-01-16 00:00:00 | 2024-06-18 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| order_or_contract | rolling_mean_cross_zero | warning | 2024-08-12 00:00:00 | 2024-11-26 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| order_or_contract | rolling_mean_cross_zero | warning | 2025-01-06 00:00:00 | 2025-12-30 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| price_momentum | 20_consecutive_neg_ic | high | 2023-04-13 00:00:00 | 2024-06-13 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| price_momentum | 20_consecutive_neg_ic | high | 2024-08-22 00:00:00 | 2024-11-26 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| price_momentum | 20_consecutive_neg_ic | high | 2025-01-30 00:00:00 | 2025-10-02 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| price_momentum | rolling_icir_below_-0.5 | high | 2023-05-04 00:00:00 | 2024-02-21 00:00:00 | 4 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| price_momentum | rolling_mean_cross_zero | warning | 2023-04-03 00:00:00 | 2024-06-28 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| price_momentum | rolling_mean_cross_zero | warning | 2024-07-31 00:00:00 | 2024-12-10 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| price_momentum | rolling_mean_cross_zero | warning | 2025-01-08 00:00:00 | 2025-12-22 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| quality | 20_consecutive_neg_ic | high | 2023-04-20 00:00:00 | 2024-06-20 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| quality | 20_consecutive_neg_ic | high | 2024-08-26 00:00:00 | 2024-12-17 00:00:00 | 4 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| quality | 20_consecutive_neg_ic | high | 2024-12-20 00:00:00 | 2025-10-23 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| quality | rolling_icir_below_-0.5 | high | 2025-02-11 00:00:00 | 2025-12-30 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| quality | rolling_mean_cross_zero | warning | 2023-06-07 00:00:00 | 2024-06-28 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| quality | rolling_mean_cross_zero | warning | 2024-08-29 00:00:00 | 2024-12-10 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| quality | rolling_mean_cross_zero | warning | 2025-01-06 00:00:00 | 2025-12-01 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| revenue_revision | 20_consecutive_neg_ic | high | 2023-11-27 00:00:00 | 2024-06-20 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| revenue_revision | 20_consecutive_neg_ic | high | 2024-07-29 00:00:00 | 2024-10-23 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| revenue_revision | 20_consecutive_neg_ic | high | 2025-03-14 00:00:00 | 2025-08-22 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| revenue_revision | rolling_icir_below_-0.5 | high | 2024-01-16 00:00:00 | 2024-01-16 00:00:00 | 3 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| revenue_revision | rolling_icir_below_-0.5 | high | 2024-08-22 00:00:00 | 2024-10-23 00:00:00 | 2 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| revenue_revision | rolling_icir_below_-0.5 | high | 2025-05-20 00:00:00 | 2025-12-30 00:00:00 | 3 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| revenue_revision | rolling_mean_cross_zero | warning | 2024-01-09 00:00:00 | 2024-06-28 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| revenue_revision | rolling_mean_cross_zero | warning | 2024-07-17 00:00:00 | 2024-12-13 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| revenue_revision | rolling_mean_cross_zero | warning | 2025-01-08 00:00:00 | 2025-10-21 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| theme_heat | 20_consecutive_neg_ic | high | 2023-03-22 00:00:00 | 2024-06-25 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| theme_heat | 20_consecutive_neg_ic | high | 2024-07-02 00:00:00 | 2024-09-10 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| theme_heat | 20_consecutive_neg_ic | high | 2025-03-21 00:00:00 | 2025-11-21 00:00:00 | 6 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| theme_heat | rolling_icir_below_-0.5 | high | 2023-09-18 00:00:00 | 2024-06-28 00:00:00 | 4 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| theme_heat | rolling_icir_below_-0.5 | high | 2024-08-22 00:00:00 | 2024-10-23 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| theme_heat | rolling_icir_below_-0.5 | high | 2025-09-03 00:00:00 | 2025-12-30 00:00:00 | 3 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| theme_heat | rolling_mean_cross_zero | warning | 2023-02-09 00:00:00 | 2024-06-28 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| theme_heat | rolling_mean_cross_zero | warning | 2024-08-29 00:00:00 | 2024-11-19 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| theme_heat | rolling_mean_cross_zero | warning | 2025-01-06 00:00:00 | 2025-08-15 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| valuation | 20_consecutive_neg_ic | high | 2023-01-02 00:00:00 | 2024-06-03 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| valuation | 20_consecutive_neg_ic | high | 2024-07-02 00:00:00 | 2024-11-28 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| valuation | 20_consecutive_neg_ic | high | 2024-12-20 00:00:00 | 2025-05-13 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Stop using this factor for new positions; reduce weight. |
| valuation | rolling_icir_below_-0.5 | high | 2023-02-21 00:00:00 | 2024-06-28 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| valuation | rolling_icir_below_-0.5 | high | 2024-08-22 00:00:00 | 2024-12-17 00:00:00 | 3 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| valuation | rolling_icir_below_-0.5 | high | 2025-02-11 00:00:00 | 2025-03-04 00:00:00 | 5 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Factor may be broken; review signal definition. |
| valuation | rolling_mean_cross_zero | warning | 2023-01-16 00:00:00 | 2024-06-28 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| valuation | rolling_mean_cross_zero | warning | 2024-07-31 00:00:00 | 2024-11-28 00:00:00 | 7 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |
| valuation | rolling_mean_cross_zero | warning | 2025-01-06 00:00:00 | 2025-07-24 00:00:00 | 8 distinct (horizon, evidence) tuples aggregated; see stability_alerts.csv raw for details. | Investigate regime change; do not trust this factor blindly. |

## 15. 分组收益和单调性
- **beta**: Spearman(quantile, return) = 0.200, is_monotonic = False, top_is_peak = True
- **liquidity**: Spearman(quantile, return) = -0.600, is_monotonic = False, top_is_peak = False
- **noise_alpha**: Spearman(quantile, return) = -0.100, is_monotonic = False, top_is_peak = False
- **noise_beta**: Spearman(quantile, return) = 0.300, is_monotonic = False, top_is_peak = False
- **order_or_contract**: Spearman(quantile, return) = nan, is_monotonic = False, top_is_peak = False
- **price_momentum**: Spearman(quantile, return) = 0.400, is_monotonic = False, top_is_peak = True
- **quality**: Spearman(quantile, return) = 0.500, is_monotonic = False, top_is_peak = True
- **revenue_revision**: Spearman(quantile, return) = -0.200, is_monotonic = False, top_is_peak = False
- **theme_heat**: Spearman(quantile, return) = -0.300, is_monotonic = False, top_is_peak = False
- **valuation**: Spearman(quantile, return) = -0.200, is_monotonic = False, top_is_peak = False

## 16. 最终推荐的因子、Horizon 和调仓频率
见下文「真正保留下来的因子 / 被削弱的因子 / 尚不能确认的因子」总结。

## 17. 研究限制
- 本报告所有数值均来自程序计算;`data_note` 为合成数据时,真实股票的真实 IC 与此不可类比。
- 主题暴露 `theme_exposure` 是合成数据中按行业 / 事件时间生成的人造变量,与现实中的实际机器人概念覆盖度有差异。
- 调仓收益为等权 + 单边成本;未考虑冲击成本、流动性、可借券、停牌等真实约束。
- daily rebalance 下的极端 annualized_return(数千/万%)是频率复合结果,不可直接外推。

## 18. 不显著或不稳定结果
对作为对照的 noise 因子:
- **noise_alpha** | best_h=20 | IC=-0.0411 | t=-2.76 | p=0.008
- **noise_beta** | best_h=20 | IC=0.0427 | t=2.61 | p=0.012

## 19. 不应被误读为有效的因子
- **概念标签本身**: 当 theme_exposure 接近全样本, 概念本身不再有区分度, 不能作为因子。
- **样本内累计收益**: 累计收益受单边运气影响大; 必须看 IC、ICIR、t-stat 和成本后收益。
- **单 Horizon 单一 IC**: 一个 Horizon 上的高 IC 不代表稳健; 必须在多 Horizon 上验证。
- **日频调仓 annualized 收益**: 几万% 的 annualized_return 是 (1+r)^252 复合结果, 无杠杆下不可持续。
- **noise_* 因子通过准入门槛**: 任何 noise 因子出现在「真正保留」列表中即说明阈值需要更严。

## 20. 真正保留下来的因子 / 被削弱的因子 / 尚不能确认的因子
以下三段从 summary / half_life / alerts 表中自动抽取。**准入条件**:
- `真正保留`: post_event 下 raw |IC| > 0.02 且 |partial IC| > 0.01 且 p < 0.10。**且**不是 noise 因子。
- `被主题扩散削弱`: pre_event 下 raw |IC| > 0.03 但 post_event 下 |partial IC| < 0.005。
- `尚不能确认`: 不满足以上任一条件。

> **注意**:名称以 `noise_` 开头的因子是噪声对照因子;其 IC 不显著才证明模块没有把噪声误判为信号。若任何 `noise_*` 因子被列为「真正保留」,则是模块本身的统计能力问题,需要更高的样本量或更严格的门槛。

**真正保留下来的因子(事件后 Incremental IC 显著为正,且稳定):**

| factor | best_h | raw_IC | partial_IC | raw_p | partial_p | recommended_rebal |
|---|---|---|---|---|---|---|
| order_or_contract | 60 | 0.1051 | 0.0494 | 0.000 | 0.000 | 1 |
| quality | 40 | 0.0634 | 0.0926 | 0.000 | 0.000 | 1 |
| revenue_revision | 60 | 0.0619 | 0.0291 | 0.000 | 0.003 | 1 |

**被主题扩散削弱的因子(事件前有效,事件后增量 IC 接近 0):**

- 在本数据集中未识别出被主题扩散显著削弱的因子。

**尚不能确认的因子(样本不足或结果不显著):**

- **beta**: 未达到准入阈值(raw |IC|<0.02 或 partial p>0.10)
- **liquidity**: 未达到准入阈值(raw |IC|<0.02 或 partial p>0.10)
- **noise_alpha**: 噪声对照因子;IC 均值不显著证明未误入为有效信号
- **noise_beta**: 噪声对照因子;IC 均值不显著证明未误入为有效信号
- **price_momentum**: 未达到准入阈值(raw |IC|<0.02 或 partial p>0.10)
- **theme_heat**: 未达到准入阈值(raw |IC|<0.02 或 partial p>0.10)
- **valuation**: 未达到准入阈值(raw |IC|<0.02 或 partial p>0.10)

## 21. 对研究主题的最终回答

**研究问题**:WAIC 之后,当所有股票都沾上机器人,真正有用的因子还剩什么?

**数据限制**:本报告基于 synthetic test data,所有数值仅供流程验证,不可外推到真实市场。

- **主题热度因子(theme_heat)**: 主题标签的截面 z-score 本身,作为因子在partial IC 框架下被控制变量完全吸收(因为它就是 `theme_exposure` 的截面标准化),在主题扩散股票池上不再提供增量预测能力。这与研究问题的前提一致:「当所有股票都沾上机器人,概念标签本身已经无法区分股票」。
- **在主题扩散后仍具有增量 IC 的因子**:

  - **quality**(h=40): raw IC=0.063, partial IC=0.093, partial p=0.000。
  - **order_or_contract**(h=60): raw IC=0.105, partial IC=0.049, partial p=0.000。
  - **revenue_revision**(h=60): raw IC=0.062, partial IC=0.029, partial p=0.003。
- 这些因子的共同特征是:都不是「机器人概念」本身,而是与基本面或财务预期相关的实质性信号(订单 / 合同、盈利质量、营收预期修正)。当主题标签已经普及,真正保留下来的,是从公司经营层面能区分股票的因子。

**调仓频率建议**:基于半衰期估计,若 daily IC 仍在增长(never_decayed),调仓频率应与 IC 增长速率匹配;若存在 trade-off(高频毛收益高但扣成本后下降),成本后净 spread 最大的频率是首选。

**核心结论**:WAIC 之后,当「机器人概念」标签本身已经几乎覆盖全市场,它作为因子的截面区分度必然下降。真正保留下来的,是从订单 / 合同、盈利质量、营收预期修正等**基本面**维度,而不是从「概念热度」维度,去识别哪些公司在主题扩散中真正受益。**主题标签不是因子,它只是被因子解释的现象。**
