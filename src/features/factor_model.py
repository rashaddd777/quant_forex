# src/features/factor_model.py

import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

def run_factor_model() -> None:
    """
    Loads returns and factor time series, runs a rolling OLS of each FX pair's returns on the factors,
    then writes out:
      - data/processed/betas.csv
      - data/processed/residuals.csv
    """
    # Load configuration
    cfg    = Config.load("model_config")
    window = cfg.get("rolling_window", 252)

    # Load processed data
    rets    = pd.read_csv(PROCESSED_DIR / "returns.csv", index_col="Date", parse_dates=True)
    factors = pd.read_csv(PROCESSED_DIR / "factors.csv", index_col="Date", parse_dates=True)
    df      = rets.join(factors, how="inner").dropna()

    betas_list  = []
    resids_list = []

    for sym in rets.columns:
        y = df[sym]
        X = sm.add_constant(df[factors.columns])

        logger.info(f"Rolling OLS for {sym} (window={window})")
        model = RollingOLS(y, X, window=window)
        res   = model.fit()

        # --- Betas ---
        # res.params is a DataFrame: index=Date, cols=['const', factor1, ...]
        beta = res.params.add_suffix(f"_{sym}")
        betas_list.append(beta)

        # --- Residuals (computed manually) ---
        # Align X to the dates where betas are available
        X_sub = X.loc[beta.index]
        # Predicted values: elementwise multiply and sum across columns
        pred = (X_sub * res.params).sum(axis=1)
        resid = (y.loc[beta.index] - pred).to_frame(name=sym)
        resids_list.append(resid)

    # Concatenate all symbols
    betas_df = pd.concat(betas_list, axis=1)
    resid_df = pd.concat(resids_list, axis=1)

    # Save to CSV
    betas_df.to_csv(PROCESSED_DIR / "betas.csv", index_label="Date")
    resid_df.to_csv(PROCESSED_DIR / "residuals.csv", index_label="Date")

    logger.info("Saved betas.csv and residuals.csv")
