"""Project configuration and constants."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"
DOCS_DIR = PROJECT_ROOT / "docs"

# NESO API
NESO_API_BASE = "https://api.neso.energy/api/3/action"

# Carbon Intensity API
CARBON_INTENSITY_BASE = "https://api.carbonintensity.org.uk"

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# RAG settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "energy_reports"

# Forecasting
FORECAST_HORIZON_HOURS = 24
