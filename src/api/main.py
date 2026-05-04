"""FastAPI service exposing the peak-demand forecaster as HTTP endpoints.

Local dev:    uvicorn src.api.main:app --reload
Interactive:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models import predict


class PredictRequest(BaseModel):
    date: dt.date = Field(..., description="ISO date (YYYY-MM-DD) to backtest")


class PredictResponse(BaseModel):
    date: str
    predicted_mw: float
    actual_mw: float
    error_mw: float
    abs_error_pct: float
    in_training_set: bool


class TestMetrics(BaseModel):
    MAE_MW: float
    MAPE_pct: float
    R2: float


class InfoResponse(BaseModel):
    model: str
    earliest_date: str
    latest_date: str
    train_test_split_date: str
    feature_count: int
    test_metrics: TestMetrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the cached model + features on startup so the first request is fast.
    predict.date_range()
    yield


app = FastAPI(
    title="UK Energy Intelligence — Peak Demand Forecaster",
    description="Mode A backtest API. POST a date, get predicted vs actual peak demand.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "docs": "/docs"}


@app.get("/info", response_model=InfoResponse)
def info() -> dict:
    return predict.model_info()


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> dict:
    try:
        return predict.predict_for_date(req.date.isoformat())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
