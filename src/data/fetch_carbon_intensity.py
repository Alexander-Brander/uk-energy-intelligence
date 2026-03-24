"""
Fetch carbon intensity data from the Carbon Intensity API.

API maintained by National Energy System Operator + University of Oxford.
Docs: https://carbon-intensity.github.io/api-definitions/

Returns half-hourly data with:
    - intensity.forecast: predicted gCO2/kWh
    - intensity.actual: measured gCO2/kWh
    - intensity.index: "very low" / "low" / "moderate" / "high" / "very high"

The API limits requests to 14-day windows, so we chunk longer date ranges.

Usage:
    cd uk-energy-intelligence
    .venv/Scripts/python -m src.data.fetch_carbon_intensity
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

from config.settings import CARBON_INTENSITY_BASE, DATA_RAW


MAX_WINDOW_DAYS = 14  # API limit per request


def fetch_carbon_intensity(date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch carbon intensity for a single date window (max 14 days).

    Args:
        date_from: ISO format e.g. "2025-01-01T00:00Z"
        date_to: ISO format e.g. "2025-01-14T00:00Z"
    """
    url = f"{CARBON_INTENSITY_BASE}/intensity/{date_from}/{date_to}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()["data"]
    rows = []
    for entry in data:
        intensity = entry.get("intensity", {})
        rows.append({
            "from": entry["from"],
            "to": entry["to"],
            "forecast": intensity.get("forecast"),
            "actual": intensity.get("actual"),
            "index": intensity.get("index"),
        })
    return pd.DataFrame(rows)


def fetch_carbon_intensity_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch carbon intensity over an arbitrary date range, chunked into 14-day windows.

    Args:
        start_date: "YYYY-MM-DD" format.
        end_date: "YYYY-MM-DD" format.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days

    all_dfs = []
    current = start

    with tqdm(total=total_days, desc="Fetching carbon intensity", unit="days") as pbar:
        while current < end:
            chunk_end = min(current + timedelta(days=MAX_WINDOW_DAYS), end)
            df = fetch_carbon_intensity(
                current.strftime("%Y-%m-%dT00:00Z"),
                chunk_end.strftime("%Y-%m-%dT00:00Z"),
            )
            all_dfs.append(df)
            days_fetched = (chunk_end - current).days
            pbar.update(days_fetched)
            current = chunk_end

    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Fetch carbon intensity data from 2024 to today
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching carbon intensity data (2024-01-01 to {today})...")
    df = fetch_carbon_intensity_range("2024-01-01", today)

    path = DATA_RAW / "carbon_intensity.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} records to {path}")
