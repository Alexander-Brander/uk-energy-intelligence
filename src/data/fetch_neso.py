"""Fetch energy data from NESO (National Energy System Operator) Data Portal."""

import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config.settings import NESO_API_BASE, DATA_RAW


# Known NESO dataset resource IDs
DATASETS = {
    "historic_demand": "f93d1835-75bc-43e5-84ad-12472b180a98",
    "generation_mix": "b2f03146-f05d-4b6e-96e0-3cf4a6c1d0a5",
}


def fetch_neso_dataset(resource_id: str, limit: int = 10000, offset: int = 0) -> pd.DataFrame:
    """Fetch a dataset from the NESO CKAN API.

    Args:
        resource_id: The CKAN resource identifier.
        limit: Number of records per request.
        offset: Starting record offset.

    Returns:
        DataFrame with the fetched records.
    """
    url = f"{NESO_API_BASE}/datastore_search"
    params = {"resource_id": resource_id, "limit": limit, "offset": offset}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    records = data["result"]["records"]
    return pd.DataFrame(records)


def fetch_all_records(resource_id: str, limit: int = 10000) -> pd.DataFrame:
    """Fetch all records from a NESO dataset, handling pagination.

    Args:
        resource_id: The CKAN resource identifier.
        limit: Records per page.

    Returns:
        DataFrame with all records.
    """
    all_records = []
    offset = 0

    # Get total count first
    url = f"{NESO_API_BASE}/datastore_search"
    params = {"resource_id": resource_id, "limit": 1}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    total = response.json()["result"]["total"]

    print(f"Total records: {total}")

    with tqdm(total=total, desc="Fetching records") as pbar:
        while offset < total:
            df = fetch_neso_dataset(resource_id, limit=limit, offset=offset)
            all_records.append(df)
            offset += limit
            pbar.update(len(df))

    return pd.concat(all_records, ignore_index=True)


def fetch_carbon_intensity(date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch carbon intensity data from the Carbon Intensity API.

    Args:
        date_from: Start date in ISO format (YYYY-MM-DDT00:00Z).
        date_to: End date in ISO format.

    Returns:
        DataFrame with carbon intensity records.
    """
    from config.settings import CARBON_INTENSITY_BASE

    url = f"{CARBON_INTENSITY_BASE}/intensity/{date_from}/{date_to}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()["data"]
    return pd.DataFrame(data)


if __name__ == "__main__":
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("Fetching historic demand data...")
    demand_df = fetch_all_records(DATASETS["historic_demand"])
    demand_path = DATA_RAW / "historic_demand.csv"
    demand_df.to_csv(demand_path, index=False)
    print(f"Saved {len(demand_df)} records to {demand_path}")
