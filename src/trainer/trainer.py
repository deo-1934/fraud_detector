# src/trainer/trainer.py
# ---------------------------------------------
# Train a fraud detection model with clean APIs
# - Loads processed data (e.g., data/processed/train_ready.csv)
# - Stratified Split, optional GridSearchCV
# - Supports Logistic Regression and RandomForest
# - Saves: best model (pkl), metrics (json), plots (png)
# ---------------------------------------------

from pathlib import Path
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------
# Paths & I/O Utilities
# ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # go up from src/trainer/
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read processed CSV and basic sanity checks."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column (0: legit, 1: fraud).")
    return df


def build_pipeline(model_name: str):
    """
    Build a sklearn pipeline based on the chosen model.
    - 'logreg': StandardScaler + LogisticRegression
    - 'rf': RandomForest (no scaler needed)
    """
    if model_name == "logreg":
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # features to similar scale for LR
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear", max_iter=1000, random_state=42
                    ),
                ),
            ]
        )
        param_grid = {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
            "clf__penalty": ["l2"],
        }
    elif model_name == "rf":
        pipe = Pipeline(
            steps=[
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,     # default; grid can override
                        random_state=42,
                        n_jobs=-1,
                        class_weight=None,
                    ),
                )
            ]
        )
        # 🔽 Light grid to cut total fits from 120 → 12 (with cv=3)
        param_grid = {
            "clf__n_estimators": [200],
            "clf__max_depth": [None, 16],
            "clf__min_samples_split": [2],
            "clf__class_weight": [None, "balanced"],
        }
    else:
        raise ValueError("Unsupported model. Use 'logreg' or 'rf'.")
    return pipe, param_grid


def tune_and_fit(pipe, param_grid, X_train, y_train, cv_folds=3, do_tune=True):
    """Optionally run GridSearchCV and fit the best estimator."""
    if not do_tune:
        pipe.fit(X_train, y_train)
        return pipe, {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,   # set to 1 if you want progress logs
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def evaluate(model, X_test, y_test, prefix: str = "") -> dict:
    """Compute key metrics and plot ROC/PR/Confusion Matrix."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])

    # Save metrics to JSON
    metrics = {
        "roc_auc": float(roc_auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    (REPORTS_DIR / f"{prefix}metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}roc_curve.png", dpi=150)
    plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}pr_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "roc_auc": roc_auc,
        "precision_at_default": float(np.nan_to_num(prec[np.argmax(rec >= 0.5)] if np.any(rec >= 0.5) else 0.0)),
        "report": report,
    }


def save_model(model, name: str):
    """Persist the trained model pipeline."""
    path = MODELS_DIR / f"{name}.pkl"
    dump(model, path)
    print(f"[INFO] Saved model → {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train fraud detection model")
    p.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "train_ready.csv"),
        help="Path to processed CSV",
    )
    p.add_argument(
        "--model",
        type=str,
        choices=["logreg", "rf"],
        default="rf",
        help="Model to train",
    )
    p.add_argument("--test_size", type=float, default=0.2, help="Test size ratio")
    p.add_argument("--no_tune", action="store_true", help="Skip GridSearchCV")
    p.add_argument("--cv", type=int, default=3, help="CV folds for tuning")  # default 3
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    # Load data
    df = load_dataset(Path(args.data))
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Build model & tune
    pipe, grid = build_pipeline(args.model)
    model, best_params = tune_and_fit(
        pipe, grid, X_train, y_train, cv_folds=args.cv, do_tune=not args.no_tune
    )
    if best_params:
        print(f"[INFO] Best params: {best_params}")

    # Evaluate
    metrics = evaluate(model, X_test, y_test, prefix=f"{args.model}_")

    # Save artifacts
    save_model(model, name=f"{args.model}_best")
    print(f"[INFO] ROC-AUC: {metrics['roc_auc']:.4f}")
    print("[DONE] Training complete. Artifacts saved in /models and /reports.")


if __name__ == "__main__":
    main()
# --- Wrappers for main.py compatibility ---
def train(pipe, X, y):
    """
    Wrapper so main.py can call `train(...)`.
    Adjust the inside to use your existing training code.
    """
    # اگر کد آموزش شما اسمش مثلاً train_model هست، از اون استفاده کن:
    # return train_model(pipe, X, y)
    pipe.fit(X, y)
    return pipe

def evaluate(pipe, X, y):
    """
    Wrapper so main.py can call `evaluate(...)`.
    Must return: (report_dict, roc_auc, pr_auc, extra_dict)
    """
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    import numpy as np

    if hasattr(pipe, "predict_proba"):
        y_scores = pipe.predict_proba(X)[:, 1]
    else:
        y_scores = pipe.decision_function(X)

    roc = roc_auc_score(y, y_scores)
    pr = average_precision_score(y, y_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
    extra = {"n_samples": len(y), "pos_rate": float(np.mean(y))}
    return rep, float(roc), float(pr), extra
