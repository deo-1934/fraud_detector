# src/data_loader/preprocessing.py

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# === Paths ===
RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load raw dataset and remove duplicates"""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_DATA_PATH}")
    
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"[INFO] Original shape: {df.shape}")
    
    df = df.drop_duplicates()
    print(f"[INFO] After removing duplicates: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame, drop_time: bool = True) -> pd.DataFrame:
    """Scale numeric features and optionally drop 'Time'"""
    scaler = RobustScaler()

    if "Amount" in df.columns:
        df["Amount"] = scaler.fit_transform(df[["Amount"]])
        print("[INFO] Scaled 'Amount' column.")

    if drop_time and "Time" in df.columns:
        df = df.drop(columns=["Time"])
        print("[INFO] Dropped 'Time' column.")
    
    return df

def handle_imbalance(df: pd.DataFrame, method: str = "smote") -> pd.DataFrame:
    """
    Handle class imbalance:
    - smote: SMOTE oversampling
    - undersample: Random undersampling
    - none: No change
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    if method == "smote":
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print("[INFO] Applied SMOTE oversampling.")
    elif method == "undersample":
        legit = df[df["Class"] == 0]
        fraud = df[df["Class"] == 1]
        legit_sample = legit.sample(len(fraud), random_state=42)
        df_res = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)
        X_res = df_res.drop(columns=["Class"])
        y_res = df_res["Class"]
        print("[INFO] Applied random undersampling.")
    else:
        X_res, y_res = X, y
        print("[INFO] No imbalance handling applied.")

    df_balanced = pd.concat([X_res, y_res], axis=1)
    print(f"[INFO] Final balanced shape: {df_balanced.shape}")
    return df_balanced

def save_data(df: pd.DataFrame, filename: str):
    """Save processed dataset to processed folder"""
    save_path = PROCESSED_DIR / filename
    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved processed data to: {save_path}")

if __name__ == "__main__":
    print("[START] Data preprocessing pipeline...")
    df_raw = load_data()
    df_clean = preprocess_data(df_raw, drop_time=True)
    df_balanced = handle_imbalance(df_clean, method="smote")
    save_data(df_balanced, "train_ready.csv")
    print("[END] Data preprocessing complete.")
