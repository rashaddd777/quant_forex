## Project Overview

Quant FX Indicator Pipeline is an end-to-end Python framework for systematic FX mean-reversion strategies. It downloads EUR/USD and GBP/USD data, computes rolling‐OLS factor models, trains an anomaly autoencoder plus Random Forest and feed-forward NN overlays, generates trading signals (anomaly, RF-blend, NN-blend, hybrid), backtests each variant, and outputs performance and per‐trade analytics.

## Key Results

| Strategy    | Total Return | CAGR  | Sharpe | Max Drawdown |
|-------------|--------------|-------|--------|--------------|
| Baseline    | 55.4%        | 9.4%  | 0.55   | –39.6%       |
| RF-Blend    | 63.6%        | 10.5% | 0.60   | –37.4%       |
| NN-Blend    | 59.9%        | 10.0% | 0.58   | –38.6%       |
| **Hybrid**  | **64.9%**    | **10.7%** | **0.61** | **–37.3%** |

## Insights

- **Blending Models** raised returns by ~9 pp and improved Sharpe from 0.55 to 0.61.  
- **Random Forest** outperformed the basic NN, suggesting tree-based methods handle these tabular features better in this data regime.  
- **Drawdowns (~37%)** remain high—next steps include volatility targeting, dynamic sizing, and stop-loss rules.  
- **Sharpe <1**: adding orthogonal features (macro factors, order-flow, regime indicators) or more advanced DL architectures (LSTM/CNN/attention) could lift risk-adjusted performance.
