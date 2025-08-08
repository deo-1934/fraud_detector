"""Generic utilities shared across modules."""
import os
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def set_seed(seed: int = 42) -> None:
    """Make results reproducible across runs where possible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dirs(*paths: Iterable[Path]) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def split_train_val(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    val_size: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/val/test while preserving class distribution (stratify).
    val_size is interpreted relative to the remaining data after test split.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed
    )
    # Adjust val size to be a fraction of the train portion only
    rel_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=rel_val, stratify=train_df[target], random_state=seed
    )
    return train_df, val_df, test_df
