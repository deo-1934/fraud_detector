"""Training and evaluation helpers."""
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix

def train(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit the pipeline on training data."""
    model.fit(X_train, y_train)
    return model

def evaluate(model, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], float, float, Dict[str, Any]]:
    """
    Evaluate a fitted model on a given split.
    Returns:
      - classification_report (dict)
      - ROC-AUC (float)
      - PR-AUC / Average Precision (float)
      - extra dict: confusion matrix and any other useful diagnostics
    """
    # Predicted class labels
    y_pred = model.predict(X)

    # For AUC metrics we need a probability or decision score
    proba_fn = getattr(model, "predict_proba", None)
    y_score = proba_fn(X)[:, 1] if proba_fn else y_pred

    roc = roc_auc_score(y, y_score)
    pr_auc = average_precision_score(y, y_score)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    extra = {
        "confusion_matrix": cm.tolist(),
        "threshold": 0.5  # default threshold used for y_pred above
    }
    return report, float(roc), float(pr_auc), extra
