# src/backtest/backtest_engine.py

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import Config
from src.utils.logger import get_logger
from src.features.residuals import load_residuals
from src.features.ml_features import create_ml_dataset
from src.models.ml_model import MLModel

logger = get_logger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

class BacktestEngine:
    """
    Generates signals by blending anomaly z-scores with ML forecasts,
    then runs a backtest.
    """

    def __init__(self):
        cfg = Config.load("backtest_config")
        self.entry_z          = cfg["entry_z"]
        self.exit_z           = cfg["exit_z"]
        self.transaction_cost = cfg.get("transaction_cost", 0.0001)
        self.leverage_vol     = cfg.get("leverage_vol", 0.10)

        # load data
        self.prices = pd.read_csv(
            PROCESSED_DIR / "prices.csv", index_col="Date", parse_dates=True
        )
        self.rets   = pd.read_csv(
            PROCESSED_DIR / "returns.csv", index_col="Date", parse_dates=True
        )
        self.resid  = load_residuals()

        # build historical ML forecasts
        self.ml = MLModel()
        ml_series = {}
        for sym in self.resid.columns:
            X, _ = create_ml_dataset(sym)
            ml_series[sym] = self.ml.predict(sym, X)
        self.ml_preds = pd.DataFrame(ml_series)

        # align indices
        idx = self.resid.index.intersection(self.ml_preds.index)
        self.resid    = self.resid.loc[idx]
        self.ml_preds = self.ml_preds.loc[idx]

    def compute_anomaly_score(self):
        # L2 norm of residuals as pandas Series
        errors = pd.Series(
            np.sqrt((self.resid ** 2).sum(axis=1)),
            index=self.resid.index,
            name="error"
        )
        mu    = errors.rolling(window=60).mean()
        sigma = errors.rolling(window=60).std()
        self.anomaly = (errors - mu) / sigma
        logger.info("Computed anomaly z-score series")

    def generate_signals(self):
        """
        Blended signals:
          - base anomaly signal (±1)
          - ml_weight = tanh(ml_pred / std(ml_pred))
          - hybrid = base * |ml_weight|
        Fallback to pure anomaly on days with no hybrid signal.
        """
        idx  = self.anomaly.index
        syms = self.resid.columns

        # 1) Base anomaly signals
        base_long  = (self.anomaly < -self.entry_z)
        base_short = (self.anomaly >  self.entry_z)
        base_sig   = pd.DataFrame(0.0, index=idx, columns=syms)
        base_sig[base_long]  =  1.0
        base_sig[base_short] = -1.0

        # 2) ML weight: normalize across history
        ml_std = self.ml_preds.std()
        ml_weight = np.tanh(self.ml_preds.divide(ml_std, axis=1))

        # 3) Hybrid signal: base * magnitude of ml_weight
        hybrid = base_sig * ml_weight.abs()

        # 4) Exit: when |anomaly| < exit_z, set signal to 0
        exit_mask = (self.anomaly.abs() < self.exit_z)
        hybrid[exit_mask] = 0.0

        # 5) Fallback: if hybrid is zero across all symbols, use base anomaly
        no_hybrid = (hybrid.abs().sum(axis=1) == 0)
        hybrid.loc[no_hybrid, :] = base_sig.loc[no_hybrid, :]

        # 6) Shift for next-day execution
        self.signals = hybrid.shift(1).fillna(0.0)
        logger.info("Generated blended ML+anomaly trading signals")

    def run_backtest(self):
        ret  = self.rets.loc[self.signals.index]
        cost = self.transaction_cost * self.signals.diff().abs()
        pnl  = (self.signals * ret).sum(axis=1) - cost.sum(axis=1)

        # if pnl is constant zero, skip scaling
        if pnl.std() == 0:
            scale = 0.0
        else:
            scale = self.leverage_vol / pnl.std() / np.sqrt(252)

        self.pnl    = pnl * scale
        self.equity = (1 + self.pnl).cumprod()
        logger.info("Backtest complete")
        return pd.DataFrame({"pnl": self.pnl, "equity": self.equity})
