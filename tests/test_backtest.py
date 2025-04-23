import numpy as np
import pandas as pd
import pytest

from src.utils.config import Config
from src.backtest.backtest_engine import BacktestEngine

@pytest.fixture(autouse=True)
def setup_backtest_env(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"EUR/USD": np.linspace(1.1, 1.2, 5)}, index=dates)
    rets   = np.log(prices / prices.shift(1)).fillna(0)
    resid  = pd.DataFrame(np.random.randn(5, 1), index=dates, columns=["EUR/USD"])

    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    prices.to_csv(proc / "prices.csv")
    rets.to_csv(proc / "returns.csv")
    resid.to_csv(proc / "residuals.csv")

    monkeypatch.setattr("src.backtest.backtest_engine.PROCESSED_DIR", proc)
    cfg = {
        "entry_z": 0.0,
        "exit_z": 0.0,
        "transaction_cost": 0.0,
        "leverage_vol": 0.10
    }
    monkeypatch.setattr(Config, "load", staticmethod(lambda name: cfg if name == "backtest_config" else {}))

def test_backtest_engine_flow():
    engine = BacktestEngine()
    engine.compute_anomaly_score()
    engine.generate_signals()
    results = engine.run_backtest()

    assert "pnl" in results.columns and "equity" in results.columns
    assert np.isclose(results["equity"].iloc[0], 1 + results["pnl"].iloc[0])
    assert (results["equity"] >= 0).all()
