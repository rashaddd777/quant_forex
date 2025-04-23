import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

def create_ml_dataset(sym: str):
    """
    Build X, y for a given FX pair:
      - X: lagged residual, rolling mean & std of residual, plus factor values
      - y: next-day residual (regression target)

    Returns:
      X (pd.DataFrame), y (pd.Series)
    """
    cfg = Config.load("ml_config")
    lag_w   = cfg["lag_window"]
    mean_w  = cfg["roll_mean_window"]
    std_w   = cfg["roll_std_window"]

    # Load residuals & factors
    resid   = pd.read_csv(PROCESSED_DIR / "residuals.csv", index_col="Date", parse_dates=True)
    factors = pd.read_csv(PROCESSED_DIR / "factors.csv",   index_col="Date", parse_dates=True)

    if sym not in resid.columns:
        raise KeyError(f"{sym} not in residuals.csv")

    df = pd.DataFrame(index=resid.index)
    df["resid"] = resid[sym]
    # lagged residual
    df[f"resid_lag{lag_w}"] = df["resid"].shift(lag_w)
    # rolling mean / std on residual
    df[f"resid_rollmean{mean_w}"] = df["resid"].rolling(mean_w).mean().shift(1)
    df[f"resid_rollstd{std_w}"]  = df["resid"].rolling(std_w).std().shift(1)
    # factor features
    # factors columns are like "EUR/USD_mom" and "EUR/USD_rv"
    mom_col = f"{sym}_mom"
    rv_col  = f"{sym}_rv"
    df["mom"] = factors[mom_col]
    df["rv"]  = factors[rv_col]

    # target: next-day residual
    df["target"] = df["resid"].shift(-1)

    # drop rows with any NaN
    df = df.dropna()

    # split features vs. target
    X = df.drop(columns=["resid", "target"])
    y = df["target"]
    logger.info(f"Built ML dataset for {sym}: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y
