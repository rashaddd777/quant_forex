import numpy as np
import pytest

from src.utils.config import Config
import src.models.autoencoder as ae

@pytest.fixture(autouse=True)
def mock_model_config(monkeypatch):
    cfg = {
        "autoencoder": {
            "latent_dim": 2,
            "hidden_layers": [4],
            "l2_reg": 0.0
        },
        "training": {
            "learning_rate": 0.001,
            "loss": "mse",
            "metrics": ["mse"],
            "epochs": 1,
            "batch_size": 2,
            "validation_split": 0.5,
            "patience": 1
        }
    }
    monkeypatch.setattr(Config, "load", staticmethod(lambda name: cfg if name == "model_config" else {}))
    return cfg

def test_build_autoencoder_shapes(mock_model_config):
    input_dim = 3
    autoencoder, encoder, decoder = ae.build_autoencoder(input_dim)
    assert autoencoder.input_shape  == (None, input_dim)
    assert autoencoder.output_shape == (None, input_dim)
    assert encoder.output_shape     == (None, mock_model_config["autoencoder"]["latent_dim"])

    x = np.random.randn(1, input_dim).astype(np.float32)
    recon = autoencoder.predict(x, verbose=0)
    assert recon.shape == (1, input_dim)
