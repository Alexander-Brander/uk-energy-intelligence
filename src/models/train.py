"""Train the XGBoost peak-demand forecaster and save deployment artifacts.

Reproduces the training procedure from notebook 02 so the saved model
matches the README's reported test metrics (MAE 785, MAPE 2.46, R2 0.96).

Outputs (artifacts/):
- xgboost_forecaster.joblib   trained model
- feature_columns.json        feature names in the order the model expects
- inference_features.csv      daily features bundled with the deploy for the backtest demo
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "daily_features.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET = "peak_demand"
DATE_COL = "date"
SPLIT_DATE = pd.Timestamp("2025-07-04")

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err) / y_true) * 100),
        "R2": 1.0 - float(np.sum(err**2)) / ss_tot,
    }


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL)
    # CSV column order matters: XGBoost's colsample_bytree=0.8 interacts with column order,
    # so we preserve the order used by the original notebook to reproduce its metrics exactly.
    feature_cols = [c for c in df.columns if c not in (DATE_COL, TARGET)]
    if len(feature_cols) != 48:
        raise RuntimeError(f"Expected 48 features, got {len(feature_cols)}")

    train_mask = df[DATE_COL] <= SPLIT_DATE
    train, test = df[train_mask], df[~train_mask]
    print(f"Train: {len(train)} rows ({train[DATE_COL].min().date()} to {train[DATE_COL].max().date()})")
    print(f"Test:  {len(test)} rows ({test[DATE_COL].min().date()} to {test[DATE_COL].max().date()})")

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train[feature_cols], train[TARGET].to_numpy())

    metrics = evaluate(test[TARGET].to_numpy(), model.predict(test[feature_cols]))
    print(f"\nTest MAE:  {metrics['MAE']:.1f} MW")
    print(f"Test MAPE: {metrics['MAPE']:.2f} %")
    print(f"Test R2:   {metrics['R2']:.4f}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model.save_model(str(ARTIFACTS_DIR / "xgboost_forecaster.json"))
    (ARTIFACTS_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    shutil.copy(PROCESSED_CSV, ARTIFACTS_DIR / "inference_features.csv")
    print(f"\nArtifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
