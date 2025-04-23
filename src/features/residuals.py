import pandas as pd
from pathlib import Path

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

def load_residuals() -> pd.DataFrame:
    """
    Loads the residuals DataFrame (dates × FX symbols) for downstream modeling.
    """
    path = PROCESSED_DIR / "residuals.csv"
    if not path.is_file():
        logger.error(f"Residuals file not found at {path}")
        raise FileNotFoundError(f"Expected residuals.csv at {path}")

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    logger.info(f"Loaded residuals: {df.shape[0]} rows × {df.shape[1]} symbols")
    return df
