"""Persisting metrics and evaluation artifacts."""
import json
from typing import Dict, Any

def save_metrics(report: Dict[str, Any], roc_auc: float, pr_auc: float, extra: Dict[str, Any], path: str) -> None:
    """Save evaluation results to a JSON file for later inspection."""
    payload = {
        "classification_report": report,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        **extra
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
