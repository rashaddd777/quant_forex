import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_sharpe(pnl, freq: int = 252) -> float:
    """
    Annualized Sharpe ratio assuming risk-free = 0.
    """
    mean_ret = pnl.mean() * freq
    std_ret  = pnl.std() * np.sqrt(freq)
    sharpe   = mean_ret / std_ret if std_ret != 0 else float("nan")
    logger.debug(f"Sharpe: {sharpe:.4f}")
    return sharpe

def calculate_max_drawdown(equity) -> float:
    """
    Maximum drawdown on equity curve.
    """
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd   = drawdown.min()
    logger.debug(f"Max Drawdown: {max_dd:.4f}")
    return max_dd

def calculate_cagr(equity, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.
    """
    n_periods    = len(equity) - 1
    if n_periods <= 0:
        return float("nan")
    total_return = equity.iloc[-1]
    cagr         = total_return ** (periods_per_year / n_periods) - 1
    logger.debug(f"CAGR: {cagr:.4f}")
    return cagr

def performance_summary(results):
    """
    Given a DataFrame with 'pnl' and 'equity' columns, returns key metrics.
    """
    pnl    = results["pnl"]
    equity = results["equity"]
    summary = {
        "total_return": equity.iloc[-1] - 1,
        "cagr":          calculate_cagr(equity),
        "sharpe":        calculate_sharpe(pnl),
        "max_drawdown":  calculate_max_drawdown(equity),
    }
    logger.info(f"Performance summary: {summary}")
    return summary
