import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.utils.config import Config
from src.data.preprocess import preprocess_data, RAW_DATA_DIR, PROCESSED_DATA_DIR

@pytest.fixture(autouse=True)
def setup_env(tmp_path, monkeypatch):
    raw  = tmp_path / "data" / "raw"
    proc = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)

    monkeypatch.setattr("src.data.preprocess.RAW_DATA_DIR", raw)
    monkeypatch.setattr("src.data.preprocess.PROCESSED_DATA_DIR", proc)
    monkeypatch.setattr(Config, "load", staticmethod(lambda name: {"symbols": ["EUR/USD"]}))
    return raw, proc

def test_preprocess_creates_price_and_return_csvs(setup_env):
    raw_dir, proc_dir = setup_env
    df = pd.DataFrame({
        "2020-01-01": [1.10, 1.20, 1.15]
    }).T
    df.index.name = None
    df.columns = ["Close"]
    df.to_csv(raw_dir / "EUR_USD.csv", header=True)

    preprocess_data()

    prices_fp  = proc_dir / "prices.csv"
    returns_fp = proc_dir / "returns.csv"
    assert prices_fp.exists()
    assert returns_fp.exists()

    prices  = pd.read_csv(prices_fp)
    returns = pd.read_csv(returns_fp)
    assert "EUR/USD" in prices.columns
    exp = pytest.approx(np.log(1.20/1.10), rel=1e-5)
    assert returns["EUR/USD"].iloc[0] == exp
