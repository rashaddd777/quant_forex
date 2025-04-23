import yaml
from pathlib import Path

class Config:
    """
    Simple loader for YAML configuration files.
    Caches loaded configs in-memory to avoid repeated I/O.
    """
    _cache = {}

    @staticmethod
    def load(name: str) -> dict:
        """
        Load a YAML config by name (e.g. 'data_config', 'model_config', 'backtest_config').
        Raises FileNotFoundError if the file does not exist.
        """
        if name in Config._cache:
            return Config._cache[name]

        # Locate project root → configs/
        config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
        path = config_dir / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} did not produce a dict")

        Config._cache[name] = data
        return data
