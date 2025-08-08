"""Data loading and basic hygiene for the raw CSV."""
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """
    Load the credit card dataset.
    The dataset contains columns: Time, V1..V28 (PCA features), Amount, Class.
    """
    df = pd.read_csv(path)
    # Basic hygiene: drop duplicated rows (rare, but safe), reset index
    df = df.drop_duplicates().reset_index(drop=True)
    # Ensure target has proper dtype
    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)
    return df
