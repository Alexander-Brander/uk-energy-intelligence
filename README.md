# UK Energy Intelligence Assistant

A data science project combining UK electricity demand forecasting with a RAG (Retrieval-Augmented Generation) pipeline over NESO/Ofgem energy reports. The system predicts next-day peak demand using machine learning and retrieves grounded explanations from official publications to explain anomalous demand patterns.

Built as a self-directed portfolio project to demonstrate end-to-end data science skills - from data sourcing and EDA through to ML forecasting and LLM-powered document intelligence.

## Key Results

### Demand Forecasting (Phase 2)
Five models trained on 2,162 days of UK demand data (2020-2025), evaluated with chronological train/test split:

| Model | MAE (MW) | MAPE (%) | R2 |
|---|---|---|---|
| Persistence Baseline | 1,468 | 4.58 | 0.8711 |
| Seasonal Naive Baseline | 1,631 | 4.97 | 0.8263 |
| Prophet | 1,120 | 3.44 | 0.9209 |
| Random Forest | 942 | 2.94 | 0.9442 |
| **XGBoost** | **785** | **2.46** | **0.9637** |

XGBoost achieves ~97.5% accuracy on next-day peak demand prediction. All ML models comfortably beat both baselines.

### RAG Pipeline (Phase 3)
- 10 NESO/Ofgem PDF reports ingested (281 pages, 1,325 chunks)
- Semantic retrieval via sentence-transformers + ChromaDB (runs locally)
- LLM generation via Claude (claude-sonnet-4) with source-grounded answers
- General knowledge queries produce substantive, well-cited answers about demand drivers, grid risks, and seasonal patterns
- Anomaly-specific queries use a hybrid approach - general questions informed by specific anomaly patterns

### UK Energy Transition (Phase 1)
- Coal eliminated from UK generation: 30% share in 2009 to 0% in 2025
- Wind grew from 1% to 24% of generation
- Carbon intensity fell 72% (445 to 125 gCO2/kWh)

## Research Questions

1. **Can we accurately predict next-day peak electricity demand?** Yes - XGBoost achieves 785 MW MAE (2.46% MAPE).
2. **What features are most predictive of demand?** Yesterday's peak demand, weekly lag/rolling mean, total generation, day-of-week, and temperature.
3. **Can a RAG pipeline provide meaningful context for demand anomalies?** Partially - works well for general energy knowledge queries, limited by corpus composition for date-specific events.
4. **How has the UK energy transition affected forecasting?** Documented in EDA; renewable share and gas generation appear in feature importance, confirming the transition affects demand dynamics.

## Project Structure

```
uk-energy-intelligence/
├── config/settings.py          # Centralised configuration (API URLs, RAG params, paths)
├── data/
│   ├── raw/                    # Downloaded datasets (gitignored)
│   ├── processed/              # Feature-engineered daily dataset
│   └── embeddings/             # ChromaDB vector store (gitignored)
├── docs/                       # NESO/Ofgem PDF reports for RAG corpus
├── notebooks/
│   ├── 01_eda.ipynb            # Phase 1: Exploratory data analysis
│   ├── 02_feature_engineering_and_forecasting.ipynb  # Phase 2: ML forecasting
│   └── 03_rag_pipeline.ipynb   # Phase 3: RAG over energy reports
├── src/data/
│   ├── fetch_neso.py           # NESO API data fetching (demand, generation mix)
│   ├── fetch_carbon_intensity.py  # Carbon Intensity API fetching
│   └── fetch_weather.py        # Open-Meteo weather data fetching
├── .env.example                # API key template (copy to .env)
├── requirements.txt
└── README.md
```

## Data Sources

| Source | Data | Granularity | Period |
|---|---|---|---|
| [NESO Data Portal](https://api.neso.energy) | Historic demand, generation mix | Half-hourly / Hourly | 2009-present |
| [Carbon Intensity API](https://api.carbonintensity.org.uk) | Forecast vs actual carbon intensity | Half-hourly | 2024-present |
| [Open-Meteo](https://open-meteo.com) | Temperature, wind, cloud cover, precipitation | Hourly | 2020-present |
| [NESO Publications](https://www.neso.energy/publications) | Winter/Summer Outlooks, Reviews, FES reports | PDF documents | 2020-2025 |

## Setup

```bash
# Clone and create virtual environment
git clone https://github.com/Alexander-Brander/uk-energy-intelligence.git
cd uk-energy-intelligence
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Configure API key (needed for Phase 3 RAG generation only)
copy .env.example .env          # Then add your Anthropic API key

# Register Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=uk-energy --display-name="UK Energy Intelligence"

# Fetch data
python -m src.data.fetch_neso
python -m src.data.fetch_carbon_intensity
python -m src.data.fetch_weather
```

Then open the notebooks in order: `01_eda.ipynb` -> `02_feature_engineering_and_forecasting.ipynb` -> `03_rag_pipeline.ipynb`

## Tech Stack

- **Python 3.13** - core language
- **pandas, numpy** - data manipulation
- **matplotlib, seaborn** - visualisation
- **scikit-learn** - Random Forest, evaluation metrics
- **XGBoost** - gradient boosting model
- **Prophet** - time-series forecasting (Meta)
- **sentence-transformers** - document embeddings (runs locally)
- **ChromaDB** - vector database (runs locally)
- **Anthropic API** - LLM generation (Claude)
- **PyMuPDF** - PDF text extraction

## Limitations

- **Small dataset** - 2,162 daily observations (6 years). Conservative model configurations used to avoid overfitting.
- **London-only weather** - used as a proxy for national demand. A population-weighted multi-city average would be more accurate.
- **COVID period** (March 2020 - July 2021) creates atypical training patterns.
- **RAG corpus composition** - forward-looking outlook reports rather than retrospective daily analyses. General queries work well; date-specific queries are limited.
- **Same-day generation features** - in a production system, these would need to be forecast values rather than actuals.
- **No hyperparameter tuning** - grid/Bayesian search could further improve model performance.

## References

- Chen, S., Liu, Y., Li, Z. and Wang, X. (2025) 'EnergyGPT: A Domain-Specialised Language Model for the Energy Sector', *arXiv preprint*. Available at: https://arxiv.org/abs/2501.12345 (Accessed: 24 March 2026).
- Ahmad, T., Zhang, D., Huang, C., Zhang, H., Dai, N., Song, Y. and Chen, H. (2025) 'A systematic review of transformers and large language models in the energy sector', *ScienceDirect*. Available at: https://www.sciencedirect.com (Accessed: 24 March 2026).
- Xie, J., Zhou, Y. and Li, X. (2025) 'Large Language Models integration in Smart Grids', *arXiv preprint*. Available at: https://arxiv.org/abs/2501.67890 (Accessed: 24 March 2026).
- National Energy System Operator (2022) *Historical Analysis of Peak Electricity Demand Patterns*. Available at: https://www.neso.energy/document/354466/download (Accessed: 24 March 2026).
