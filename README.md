# UK Energy Intelligence Assistant

A system combining quantitative UK energy data analysis with a RAG (Retrieval-Augmented Generation) pipeline over energy reports and policy documents. Ask questions like *"What drove the demand spike on 15 January 2025?"* and get answers grounded in both data and official NESO/Ofgem publications.

## Project Structure

```
uk-energy-intelligence/
├── config/             # Configuration and settings
├── data/
│   ├── raw/            # Raw downloaded data (gitignored)
│   ├── processed/      # Cleaned and feature-engineered data
│   └── embeddings/     # Vector store data
├── docs/               # Energy reports and PDFs for RAG corpus
├── notebooks/          # Exploratory analysis notebooks
├── src/
│   ├── api/            # API clients
│   ├── data/           # Data fetching and processing
│   ├── features/       # Feature engineering
│   ├── models/         # Forecasting models
│   ├── rag/            # RAG pipeline (chunking, embedding, retrieval)
│   └── visualization/  # Charts and dashboards
├── tests/              # Unit and integration tests
├── requirements.txt
└── README.md
```

## Data Sources

- **NESO Data Portal** — Historic electricity demand, generation mix, wind/solar outturn at half-hourly intervals
- **Carbon Intensity API** — Half-hourly carbon intensity data (NESO + University of Oxford)
- **NESO/Ofgem Reports** — Future Energy Scenarios, system performance reports, regulatory documents

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env       # add your API keys
```

## Phases

1. **Quantitative Analysis** — Demand forecasting, generation mix analysis, anomaly detection
2. **RAG Pipeline** — Document ingestion, embedding, retrieval over energy reports
3. **Integration** — Anomaly-triggered context retrieval, Streamlit UI
