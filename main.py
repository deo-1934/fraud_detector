"""Command-line entrypoint to train, evaluate, and save the fraud detection pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.config.config import Config, Paths
from src.utils.utils import set_seed, ensure_dirs, split_train_val
from src.data_loader.data_loader import load_csv
from src.model_builder.model_builder import build_baseline
from src.trainer.trainer import train, evaluate
from src.evaluator.evaluator import save_metrics

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments to override defaults when needed."""
    p = argparse.ArgumentParser(description="Train and evaluate the Fraud Detector pipeline.")
    p.add_argument("--data", type=str, default=str(Paths.DATA_RAW), help="Path to creditcard.csv")
    p.add_argument("--model-out", type=str, default=str(Paths.MODEL_FILE), help="Path to save trained pipeline")
    p.add_argument("--metrics-out", type=str, default=str(Paths.METRICS_FILE), help="Path to save metrics JSON")
    p.add_argument("--test-size", type=float, default=Config.TEST_SIZE, help="Test split size (0-1)")
    p.add_argument("--val-size", type=float, default=Config.VAL_SIZE, help="Validation split size (0-1 of total)")
    p.add_argument("--seed", type=int, default=Config.RANDOM_SEED, help="Random seed")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    cfg = Config()
    paths = Paths()

    set_seed(args.seed)
    ensure_dirs(Path(paths.MODELS_DIR), Path(paths.REPORTS_DIR))

    # --- 1) Load
    csv_path = Path(args.data)
    assert csv_path.exists(), f"CSV not found at: {csv_path}"
    df = load_csv(str(csv_path))

    # --- 2) Feature/target split
    target = cfg.TARGET_COL
    assert target in df.columns, f"Target column '{target}' not found in CSV."
    feature_cols = [c for c in df.columns if c != target]  # Time, Amount, V1..V28
    X = df[feature_cols]
    y = df[target]

    # --- 3) Split into train/val/test
    train_df, val_df, test_df = split_train_val(df, target, args.test_size, args.val_size, args.seed)
    Xtr, ytr = train_df[feature_cols], train_df[target]
    Xva, yva = val_df[feature_cols], val_df[target]
    Xte, yte = test_df[feature_cols], test_df[target]

    # --- 4) Build model
    pipe = build_baseline(feature_cols)

    # --- 5) Train on train split
    pipe = train(pipe, Xtr, ytr)

    # --- 6) Validate
    val_report, val_roc, val_pr, val_extra = evaluate(pipe, Xva, yva)
    print(f"[VAL] ROC-AUC={val_roc:.4f} | PR-AUC={val_pr:.4f}")

    # Save interim metrics
    save_metrics(
        report={"val": val_report},
        roc_auc=val_roc,
        pr_auc=val_pr,
        extra={"phase": "val", **val_extra},
        path=str(args.metrics_out)
    )

    # --- 7) Retrain on train+val; final test
    trval_df = pd.concat([train_df, val_df], axis=0)
    Xtrval, ytrval = trval_df[feature_cols], trval_df[target]

    final_pipe = build_baseline(feature_cols)
    final_pipe = train(final_pipe, Xtrval, ytrval)

    te_report, te_roc, te_pr, te_extra = evaluate(final_pipe, Xte, yte)
    print(f"[TEST] ROC-AUC={te_roc:.4f} | PR-AUC={te_pr:.4f}")

    # --- 8) Persist artifacts
    joblib.dump(final_pipe, args.model_out)
    save_metrics(
        report={"val": val_report, "test": te_report},
        roc_auc=te_roc,
        pr_auc=te_pr,
        extra={"phase": "test", **te_extra},
        path=str(args.metrics_out)
    )
    print(f"Artifacts saved -> model: {args.model_out} | metrics: {args.metrics_out}")

if __name__ == "__main__":
    main()
