# src/main.py

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import Config

from src.data.download_data import download_fx_data
from src.data.preprocess import preprocess_data

from src.features.compute_factors import compute_factors
from src.features.factor_model import run_factor_model
from src.features.residuals import load_residuals
from src.features.ml_features import create_ml_dataset

from src.models.train_ml import train_ml_models
from src.models.train_nn import train_nn_models
from src.models.ml_model import MLModel
from src.models.predict_nn import NNModel
from src.models.train import train_autoencoder

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.performance import performance_summary


def main():
    logger = get_logger(__name__)
    logger.info("=== Starting Quant FX Indicator Pipeline ===")

    # 1. Download & preprocess data
    download_fx_data()
    preprocess_data()

    # 2. Compute factor series & residuals
    compute_factors()
    run_factor_model()

    # 3. Train ML (RF) and NN models on residuals
    logger.info("=== Training ML Models ===")
    train_ml_models()
    logger.info("=== Training NN Models ===")
    train_nn_models()

    # 4. Train the autoencoder (not used in ablation but part of pipeline)
    train_autoencoder()

    # 5. Build historical RF and NN forecasts
    symbols = Config.load("data_config")["symbols"]
    ml = MLModel()
    nn = NNModel()

    # load residuals
    resid = load_residuals()

    # build per-symbol DataFrames of predictions
    ml_preds = pd.DataFrame({sym: ml.predict(sym, create_ml_dataset(sym)[0])
                              for sym in symbols})
    nn_preds = pd.DataFrame({sym: nn.predict(sym, create_ml_dataset(sym)[0])
                              for sym in symbols})

    # align indices
    idx = resid.index.intersection(ml_preds.index).intersection(nn_preds.index)
    resid    = resid.loc[idx]
    ml_preds = ml_preds.loc[idx]
    nn_preds = nn_preds.loc[idx]

    # 6. Compute anomaly z-score via engine
    engine = BacktestEngine()
    engine.compute_anomaly_score()
    anomaly = engine.anomaly

    entry_z = engine.entry_z
    exit_z  = engine.exit_z

    # 7. Build base anomaly signals (±1 mean-revert)
    base = pd.DataFrame(0.0, index=idx, columns=symbols)
    base.loc[anomaly < -entry_z, symbols] =  1.0
    base.loc[anomaly >  entry_z, symbols] = -1.0
    # exit when back inside threshold
    inside = anomaly.abs() < exit_z
    base.loc[inside, symbols] = 0.0

    # 8. Helper to build weighted signals
    def weighted_sig(base_sig: pd.DataFrame, weight: pd.DataFrame):
        sig = base_sig * weight.abs()
        sig.loc[inside, :] = 0.0
        # fallback to base if weight zero
        no_trade = (sig.abs().sum(axis=1) == 0)
        sig.loc[no_trade, :] = base_sig.loc[no_trade, :]
        return sig.shift(1).fillna(0.0)

    # normalize and tanh-clip weights
    ml_weight     = np.tanh( ml_preds.div( ml_preds.std()     ))
    nn_weight     = np.tanh( nn_preds.div( nn_preds.std()     ))
    hybrid_weight = (ml_weight.abs() + nn_weight.abs()) / 2

    # 9. Prepare each strategy's signals
    strategies = [
        ("Baseline", base.shift(1).fillna(0.0)),
        ("RF-blend", weighted_sig(base, ml_weight)),
        ("NN-blend", weighted_sig(base, nn_weight)),
        ("Hybrid" , weighted_sig(base, hybrid_weight)),
    ]

    # 10. Run backtests and compare
    for name, signals in strategies:
        engine.signals = signals
        results = engine.run_backtest()
        summary = performance_summary(results)
        logger.info(f"{name} summary: {summary}")

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()


# python -m src.main
