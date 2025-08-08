"""Global configuration for the Fraud Detector project."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    """Centralized project paths to avoid hard-coded strings across the codebase."""
    DATA_RAW: Path = Path("data/raw/creditcard.csv")
    MODELS_DIR: Path = Path("models")
    REPORTS_DIR: Path = Path("reports")
    MODEL_FILE: Path = MODELS_DIR / "fraud_pipeline.joblib"
    METRICS_FILE: Path = REPORTS_DIR / "metrics.json"

@dataclass(frozen=True)
class Config:
    """Experiment-level constants and default splits."""
    RANDOM_SEED: int = 42
    TARGET_COL: str = "Class"     # 0: legitimate, 1: fraud
    TEST_SIZE: float = 0.20       # 20% test
    VAL_SIZE: float = 0.10        # 10% of (train+val) reserved for val
