"""Model factories and preprocessing pipelines."""
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_baseline(numeric_cols: List[str]) -> Pipeline:
    """
    Baseline fraud detector:
      - Scale numeric features with StandardScaler
      - Train a Logistic Regression with class_weight="balanced"
    This is a strong, interpretable baseline for imbalanced binary classification.
    """
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=True), numeric_cols)],
        remainder="drop"  # keep the pipeline explicit; drop any unexpected columns
    )
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",  # counteract class imbalance
        n_jobs=None               # keep default; LR parallelism depends on solver
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe
