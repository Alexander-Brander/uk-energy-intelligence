"""
Fetch historical weather data from the Open-Meteo API.

Open-Meteo is a free, open-source weather API that requires no API key
or authentication. It provides historical hourly weather data sourced
from national weather services (ECMWF, DWD, NOAA).

We fetch data for London (51.51, -0.13) as a representative UK location.
While the UK has regional variation, London is the largest demand centre
and national-level demand correlates strongly with London temperature.

Variables fetched:
    - temperature_2m: Air temperature at 2m height (C) - primary demand driver
    - wind_speed_10m: Wind speed at 10m height (km/h) - affects wind generation
    - cloud_cover: Total cloud cover (%) - affects solar generation
    - precipitation: Total precipitation (mm) - secondary demand factor

Usage:
    cd uk-energy-intelligence
    .venv/Scripts/python -m src.data.fetch_weather
"""

import requests
import pandas as pd
from datetime import datetime

from config.settings import DATA_RAW


# Open-Meteo historical weather API (free, no key required)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# London coordinates - representative UK demand centre
LATITUDE = 51.51
LONGITUDE = -0.13

# Variables to fetch
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
]


def fetch_weather_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather data for a date range.

    Args:
        start_date: "YYYY-MM-DD" format.
        end_date: "YYYY-MM-DD" format.

    Returns:
        DataFrame with hourly weather records.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/London",
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "cloud_cover": hourly["cloud_cover"],
        "precipitation": hourly["precipitation"],
    })

    return df


if __name__ == "__main__":
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Fetch weather data matching our demand data range (2020-2025)
    # Open-Meteo archive goes up to ~5 days ago, so use a safe end date
    today = datetime.now().strftime("%Y-%m-%d")

    print("Fetching weather data (2020-01-01 to recent)...")
    print("Location: London (51.51, -0.13)")
    print("Source: Open-Meteo Archive API (free, no key required)")

    df = fetch_weather_range("2020-01-01", today)

    path = DATA_RAW / "weather_london.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} records to {path}")
