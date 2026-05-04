"""Backtest inference for the deployed XGBoost peak-demand forecaster.

Mode A demo: caller supplies a date in the historical dataset and gets the
model's predicted peak demand alongside the actual recorded peak. The model
was trained on 2020-01-31 to 2025-07-04 and tested on 2025-07-05 onward;
dates in the train range will appear nearly perfect, dates after the split
are the genuine evaluation.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_forecaster.json"
FEATURES_PATH = ARTIFACTS_DIR / "inference_features.csv"
FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_columns.json"

SPLIT_DATE = pd.Timestamp("2025-07-04")


@lru_cache(maxsize=1)
def _load():
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    feature_cols = json.loads(FEATURE_COLS_PATH.read_text())
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"]).set_index("date").sort_index()
    return model, feature_cols, df


def predict_for_date(date: str | pd.Timestamp) -> dict:
    model, feature_cols, df = _load()
    ts = pd.Timestamp(date).normalize()
    if ts not in df.index:
        lo, hi = df.index.min().date(), df.index.max().date()
        raise ValueError(f"Date {ts.date()} not in inference dataset (available: {lo} to {hi})")
    row = df.loc[[ts], feature_cols]
    predicted = float(model.predict(row)[0])
    actual = float(df.at[ts, "peak_demand"])
    return {
        "date": str(ts.date()),
        "predicted_mw": round(predicted, 1),
        "actual_mw": round(actual, 1),
        "error_mw": round(predicted - actual, 1),
        "abs_error_pct": round(abs(predicted - actual) / actual * 100, 2),
        "in_training_set": ts <= SPLIT_DATE,
    }


def date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    _, _, df = _load()
    return df.index.min(), df.index.max()


def split_date() -> pd.Timestamp:
    return SPLIT_DATE


@lru_cache(maxsize=1)
def all_predictions() -> pd.DataFrame:
    """Predict every date in the inference set, returning a DataFrame indexed by date."""
    model, feature_cols, df = _load()
    preds = model.predict(df[feature_cols])
    out = df[["peak_demand"]].copy()
    out["predicted"] = preds
    out["error"] = out["predicted"] - out["peak_demand"]
    out["abs_error_pct"] = (out["error"].abs() / out["peak_demand"]) * 100
    return out


def model_info() -> dict:
    _, feature_cols, df = _load()
    return {
        "model": "XGBoost peak-demand forecaster",
        "earliest_date": df.index.min().date().isoformat(),
        "latest_date": df.index.max().date().isoformat(),
        "train_test_split_date": SPLIT_DATE.date().isoformat(),
        "feature_count": len(feature_cols),
        "test_metrics": {"MAE_MW": 784.9, "MAPE_pct": 2.46, "R2": 0.9611},
    }


if __name__ == "__main__":
    for d in ("2024-12-25", "2025-09-15", "2025-12-30"):
        print(predict_for_date(d))
