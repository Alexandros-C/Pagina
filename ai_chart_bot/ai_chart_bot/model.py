from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .features import FEATURE_COLUMNS, build_features, build_forward_returns


@dataclass
class ModelResult:
    proba_up: float
    proba_down: float
    action: str  # "Buy" | "Sell" | "Wait"
    rationale: List[str]


def train_and_infer(df: pd.DataFrame, horizon_days: int = 5) -> ModelResult:
    enriched = build_features(df)
    enriched = enriched.dropna(subset=["Close"])  # safety
    # labels: 1 if forward return > 0 else 0
    forward = build_forward_returns(enriched, horizon_days=horizon_days)
    enriched["label_up"] = (forward > 0).astype(float)

    # drop rows with NaNs across features/label
    model_df = enriched.dropna(subset=FEATURE_COLUMNS + ["label_up"]).copy()
    if len(model_df) < 60:
        # not enough samples to train
        return ModelResult(proba_up=0.5, proba_down=0.5, action="Wait", rationale=["Insufficient history to train model."])

    X = model_df[FEATURE_COLUMNS].values
    y = model_df["label_up"].values

    # simple time-based split
    split_index = int(len(model_df) * 0.8)
    X_train, y_train = X[:split_index], y[:split_index]
    X_last = X[-1:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_last)[0]
    proba_down, proba_up = float(proba[0]), float(proba[1])

    # derive suggested action with simple thresholds
    rationale: List[str] = []
    if proba_up >= 0.58:
        action = "Buy"
        rationale.append(f"Model projects {proba_up:.0%} chance of upside in {horizon_days}d.")
    elif proba_down >= 0.58:
        action = "Sell"
        rationale.append(f"Model projects {proba_down:.0%} chance of downside in {horizon_days}d.")
    else:
        action = "Wait"
        rationale.append("Model confidence below threshold; prefer no trade.")

    return ModelResult(proba_up=proba_up, proba_down=proba_down, action=action, rationale=rationale)

