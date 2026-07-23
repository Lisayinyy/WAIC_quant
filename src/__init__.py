"""
WAIC 之后具身智能股票池因子 IC 分析模块
========================================

当所有股票都沾上机器人之后,真正有用的因子还剩什么?

核心模块:
    - data_loader: 数据校验与对齐
    - synthetic_data: 合成数据生成器(仅用于测试,带主题扩散机制)
    - forward_returns: 未来收益计算(防未来函数)
    - ic_analysis: 基础IC、Partial IC、Incremental IC
    - horizon: 多horizon分析、IC衰减、半衰期
    - rebalance: 调仓频率分析与分组组合
    - stability: 因子稳定性监控
    - portfolio: 组合构建与回测
    - reporting: 报告生成
"""
__version__ = "0.1.0"
