import pandas as pd
from pathlib import Path

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

def compute_factors() -> None:
    """
    Reads data/processed/returns.csv and computes:
      - 6-month momentum (configurable, default 126 days)
      - 1-month realized volatility (configurable, default 21 days)
    Writes:
      - data/processed/factors.csv
    """
    rets = pd.read_csv(PROCESSED_DIR / "returns.csv", index_col="Date", parse_dates=True)

    cfg = Config.load("model_config")
    mom_window = cfg.get("momentum_window", 126)
    rv_window  = cfg.get("rv_window", 21)

    # Compute momentum & realized vol
    momentum    = rets.rolling(window=mom_window).sum().add_suffix("_mom")
    realized_vol = rets.rolling(window=rv_window).std().add_suffix("_rv")

    factors = pd.concat([momentum, realized_vol], axis=1).dropna()
    factors.to_csv(PROCESSED_DIR / "factors.csv", index_label="Date")
    logger.info(f"Saved factors to {PROCESSED_DIR / 'factors.csv'}")
