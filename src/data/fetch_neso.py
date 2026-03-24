"""
Fetch energy data from the NESO (National Energy System Operator) Data Portal.

NESO uses a CKAN-based API. Each dataset has a unique resource ID.
Docs: https://api.neso.energy/api/3/action/datastore_search?resource_id=<id>

Data sources:
    - Historic Demand Data: half-hourly national demand, embedded wind/solar,
      interconnector flows. One resource per year (2009-2026).
      Fields: SETTLEMENT_DATE, SETTLEMENT_PERIOD (1-48), ND (national demand MW),
      TSD (transmission system demand MW), EMBEDDED_WIND_GENERATION,
      EMBEDDED_SOLAR_GENERATION, plus interconnector flows (IFA, BRITNED, etc.)

    - Historic Generation Mix: hourly generation by fuel type (gas, coal, nuclear,
      wind, solar, hydro, biomass, imports) plus carbon intensity.
      Single resource with 300K+ records from 2009 to present.
      Fields: DATETIME, GAS, COAL, NUCLEAR, WIND, SOLAR, HYDRO, BIOMASS,
      IMPORTS, GENERATION (total MW), CARBON_INTENSITY (gCO2/kWh), etc.

Usage:
    cd uk-energy-intelligence
    .venv/Scripts/python -m src.data.fetch_neso
"""

import requests
import pandas as pd
from tqdm import tqdm

from config.settings import NESO_API_BASE, DATA_RAW


# ── Resource IDs ──────────────────────────────────────────────────────────────
# Found via: https://api.neso.energy/api/3/action/package_search?q=demand

# Historic Generation Mix (hourly, 2009-present, ~300K records)
# Package: "Historic generation mix and carbon intensity"
GENERATION_MIX_ID = "f93d1835-75bc-43e5-84ad-12472b180a98"

# Historic Demand Data — one resource per year
# Package: "Historic Demand Data"
# Fields use text dates like "01-Jan-23" and SETTLEMENT_PERIOD 1-48
DEMAND_BY_YEAR = {
    2009: "ed8a37cb-65ac-4581-8dbc-a3130780da3a",
    2010: "b3eae4a5-8c3c-4df1-b9de-7db243ac3a09",
    2011: "01522076-2691-4140-bfb8-c62284752efd",
    2012: "4bf713a2-ea0c-44d3-a09a-63fc6a634b00",
    2013: "2ff7aaff-8b42-4c1b-b234-9446573a1e27",
    2014: "b9005225-49d3-40d1-921c-03ee2d83a2ff",
    2015: "cc505e45-65ae-4819-9b90-1fbb06880293",
    2016: "3bb75a28-ab44-4a0b-9b1c-9be9715d3c44",
    2017: "2f0f75b8-39c5-46ff-a914-ae38088ed022",
    2018: "fcb12133-0db0-4f27-a4a5-1669fd9f6d33",
    2019: "dd9de980-d724-415a-b344-d8ae11321432",
    2020: "33ba6857-2a55-479f-9308-e5c4c53d4381",
    2021: "18c69c42-f20d-46f0-84e9-e279045befc6",
    2022: "bb44a1b5-75b1-4db2-8491-257f23385006",
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
    2026: "8a4a771c-3929-4e56-93ad-cdf13219dea5",
}

# Demand Data Update (rolling recent data with forecast indicator)
# Has proper ISO date format and FORECAST_ACTUAL_INDICATOR column ("A"/"F")
DEMAND_UPDATE_ID = "177f6fa4-ae49-4182-81ea-0c6b35f26ca6"


# ── Core fetch functions ──────────────────────────────────────────────────────

def fetch_neso_dataset(resource_id: str, limit: int = 10000, offset: int = 0) -> pd.DataFrame:
    """Fetch a single page from the NESO CKAN datastore API."""
    url = f"{NESO_API_BASE}/datastore_search"
    params = {"resource_id": resource_id, "limit": limit, "offset": offset}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    records = response.json()["result"]["records"]
    return pd.DataFrame(records)


def fetch_all_records(resource_id: str, limit: int = 10000) -> pd.DataFrame:
    """Fetch all records from a NESO dataset, handling pagination."""
    # Get total count
    url = f"{NESO_API_BASE}/datastore_search"
    response = requests.get(url, params={"resource_id": resource_id, "limit": 1}, timeout=30)
    response.raise_for_status()
    total = response.json()["result"]["total"]

    print(f"  Total records: {total:,}")

    all_records = []
    offset = 0
    with tqdm(total=total, desc="  Fetching") as pbar:
        while offset < total:
            df = fetch_neso_dataset(resource_id, limit=limit, offset=offset)
            all_records.append(df)
            offset += limit
            pbar.update(len(df))

    return pd.concat(all_records, ignore_index=True)


# ── Dataset-specific fetchers ─────────────────────────────────────────────────

def fetch_generation_mix() -> pd.DataFrame:
    """Fetch the full historic generation mix dataset (2009-present).

    Returns hourly data with columns: DATETIME, GAS, COAL, NUCLEAR, WIND,
    SOLAR, HYDRO, BIOMASS, IMPORTS, GENERATION, CARBON_INTENSITY, etc.
    ~300K records — takes a few minutes.
    """
    print("Fetching historic generation mix...")
    return fetch_all_records(GENERATION_MIX_ID)


def fetch_demand_years(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """Fetch historic demand data for a range of years.

    Each year has ~17,500 records (48 half-hour periods x 365 days).
    Default fetches 2020-2025 (~105K records).

    Args:
        start_year: First year to fetch (earliest available: 2009).
        end_year: Last year to fetch (inclusive).
    """
    all_dfs = []
    for year in range(start_year, end_year + 1):
        if year not in DEMAND_BY_YEAR:
            print(f"  Skipping {year} — no resource ID available")
            continue
        print(f"Fetching demand data for {year}...")
        df = fetch_all_records(DEMAND_BY_YEAR[year])
        df["_year"] = year  # tag source year for debugging
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # 1. Generation mix (hourly, 2009-present)
    gen_df = fetch_generation_mix()
    gen_path = DATA_RAW / "generation_mix.csv"
    gen_df.to_csv(gen_path, index=False)
    print(f"Saved {len(gen_df):,} records to {gen_path}\n")

    # 2. Historic demand (half-hourly, 2020-2025)
    demand_df = fetch_demand_years(start_year=2020, end_year=2025)
    demand_path = DATA_RAW / "historic_demand.csv"
    demand_df.to_csv(demand_path, index=False)
    print(f"Saved {len(demand_df):,} records to {demand_path}\n")

    print("Done! Check data/raw/ for the downloaded CSVs.")
