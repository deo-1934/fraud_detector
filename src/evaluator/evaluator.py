# src/evaluator/evaluator.py
# ------------------------------------------------------
# Minimal evaluator utilities for fraud detection project
# - Computes ROC/PR curves, confusion matrix
# - Finds best threshold by F1 (you can change to F-beta)
# - Saves metrics.json (merging with existing content if present)
# - CLI to generate plots after training
# ------------------------------------------------------

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

# ---------- Core helpers ----------

def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray, beta: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """
    Find threshold maximizing F-beta on precision-recall curve.
    - y_true: binary {0,1}
    - y_scores: probabilities for class 1
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # avoid division by zero
    beta2 = beta * beta
    fbeta = (1 + beta2) * (precision * recall) / np.clip((beta2 * precision + recall), 1e-12, None)
    # thresholds has len = len(precision)-1
    idx = np.nanargmax(fbeta[:-1])
    best_thr = float(thresholds[idx])
    return best_thr, {
        "best_fbeta": float(fbeta[idx]),
        "best_precision": float(precision[idx]),
        "best_recall": float(recall[idx]),
    }

def plot_roc_pr(y_true: np.ndarray, y_scores: np.ndarray, outdir: str) -> Dict[str, str]:
    """
    Save ROC and PR curves to outdir. Returns file paths.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    roc_path = str(Path(outdir) / "roc.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_path = str(Path(outdir) / "pr.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    return {"roc": roc_path, "pr": pr_path}

def conf_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, outdir: str, name: str = "cm.png") -> str:
    """
    Save a simple confusion matrix heatmap.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2), yticks=np.arange(2),
        xticklabels=[0, 1], yticklabels=[0, 1],
        xlabel="Predicted", ylabel="True", title="Confusion Matrix"
    )
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    cm_path = str(Path(outdir) / name)
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    return cm_path

def save_metrics(report: Dict, roc_auc: float, pr_auc: float, extra: Dict, path: str) -> None:
    """
    Merge & save metrics JSON.
    - report: dict like {"val": classification_report_dict, "test": ...}
    - roc_auc, pr_auc: floats
    - extra: any extra info (phase, threshold, etc.)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    data.update({
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "report": report,
        "extra": extra,
    })
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

# ---------- Lightweight CLI ----------
# Use this after training to:
# - load model + raw CSV
# - compute probs
# - find best threshold (by F1)
# - save curves & confusion matrix
# - update reports/metrics.json with best_threshold

def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    # Works for sklearn Pipeline with predict_proba
    proba = model.predict_proba(X)[:, 1]
    return np.asarray(proba)

def run_cli(
    model_path: str,
    csv_path: str,
    target_col: str,
    outdir: str,
    metrics_path: str,
    beta: float = 1.0,
) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)

    assert target_col in df.columns, f"Target '{target_col}' not in CSV columns."
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col].values.astype(int)

    y_scores = _predict_proba(model, X)
    roc_auc = roc_auc_score(y, y_scores)
    pr_auc = average_precision_score(y, y_scores)

    best_thr, thr_stats = find_best_threshold(y, y_scores, beta=beta)
    y_pred = (y_scores >= best_thr).astype(int)

    # Plots
    paths = plot_roc_pr(y, y_scores, outdir=outdir)
    cm_path = conf_matrix_plot(y, y_pred, outdir=outdir, name="cm.png")

    # Reports
    clf_rep = classification_report(y, y_pred, output_dict=True)

    # Save merged metrics (append best_threshold & assets)
    extra = {
        "phase": "test_like_fullcsv",
        "best_threshold": best_thr,
        "threshold_stats": thr_stats,
        "assets": {"roc": paths["roc"], "pr": paths["pr"], "cm": cm_path},
    }
    save_metrics(report={"fullcsv": clf_rep}, roc_auc=roc_auc, pr_auc=pr_auc, extra=extra, path=metrics_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Post-train evaluator for fraud detection.")
    parser.add_argument("--model", required=True, type=str, help="Path to trained pipeline .joblib")
    parser.add_argument("--data", required=True, type=str, help="Path to CSV (e.g., data/raw/creditcard.csv)")
    parser.add_argument("--target", default="Class", type=str, help="Target column name")
    parser.add_argument("--outdir", default="reports/figures", type=str, help="Directory to save plots")
    parser.add_argument("--metrics", default="reports/metrics.json", type=str, help="Path to metrics JSON")
    parser.add_argument("--beta", default=1.0, type=float, help="Beta for F-beta (1.0 = F1)")
    args = parser.parse_args()

    run_cli(
        model_path=args.model,
        csv_path=args.data,
        target_col=args.target,
        outdir=args.outdir,
        metrics_path=args.metrics,
        beta=args.beta,
    )
