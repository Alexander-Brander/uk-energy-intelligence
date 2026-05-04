"""Streamlit frontend for the UK Energy Intelligence peak-demand forecaster.

Run locally:  streamlit run app.py
Deployed on:  Hugging Face Spaces (see README)
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.models import predict

st.set_page_config(
    page_title="UK Energy Intelligence",
    page_icon="⚡",
    layout="wide",
)

# Theming and layout tweaks: wider sidebar with justified text, sticky main title.
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] {
        width: 360px !important;
        min-width: 360px !important;
      }
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] li {
        text-align: justify;
        text-justify: inter-word;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

info = predict.model_info()
lo, hi = predict.date_range()
split = predict.split_date()
m = info["test_metrics"]

with st.sidebar:
    st.title("⚡ UK Energy Intelligence")
    st.markdown(
        "Backtest the XGBoost peak-demand forecaster against any day in the historical "
        f"dataset ({info['earliest_date']} to {info['latest_date']}). The model predicts "
        "next-day peak electricity demand from 48 lag, weather, generation-mix, and "
        "calendar features."
    )

    st.divider()
    st.markdown("#### Test-set performance")
    st.metric("MAE (MW)", f"{m['MAE_MW']:.1f}", help="Mean absolute error on the 180-day held-out test set")
    st.metric("MAPE (%)", f"{m['MAPE_pct']:.2f}", help="Mean absolute percentage error")
    st.metric("R²", f"{m['R2']:.4f}", help="Coefficient of determination")

    st.divider()
    st.markdown("#### Model details")
    st.markdown(
        f"- **Type:** {info['model']}\n"
        f"- **Features:** {info['feature_count']}\n"
        f"- **Trained:** {info['earliest_date']} to {info['train_test_split_date']}\n"
        f"- **Tested:** 2025-07-05 to {info['latest_date']}"
    )

    st.divider()
    st.caption(
        "Data sources: NESO, Carbon Intensity API, Open-Meteo. "
        "[GitHub repo](https://github.com/Alexander-Brander/uk-energy-intelligence)"
    )

st.title("UK Peak Electricity Demand Forecaster")
st.markdown(
    "Pick a date and see what the trained model would have predicted versus what actually "
    f"happened. Dates **after {split.date()}** are the genuine evaluation; earlier dates "
    "are training set and will appear near-perfect."
)

default_date = pd.Timestamp("2025-09-16").date()
selected = st.date_input(
    "Select a date",
    value=default_date,
    min_value=lo.date(),
    max_value=hi.date(),
)

result = predict.predict_for_date(selected.isoformat())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted peak", f"{result['predicted_mw']:,.0f} MW")
c2.metric("Actual peak", f"{result['actual_mw']:,.0f} MW")
c3.metric("Error", f"{result['error_mw']:+,.0f} MW")
c4.metric("Absolute error", f"{result['abs_error_pct']:.2f} %")

if result["in_training_set"]:
    st.warning(
        f"This date was in the **training set**. The near-perfect prediction reflects "
        f"how well the model fits training data, not generalisation. Pick a date after "
        f"{split.date()} for an honest test of model performance."
    )

st.subheader("Predicted vs actual (60-day window)")
window_start = pd.Timestamp(selected) - pd.Timedelta(days=30)
window_end = pd.Timestamp(selected) + pd.Timedelta(days=30)
window = predict.all_predictions().loc[window_start:window_end].reset_index()
plot_df = window.melt(
    id_vars="date",
    value_vars=["peak_demand", "predicted"],
    var_name="series",
    value_name="MW",
)
plot_df["series"] = plot_df["series"].map({"peak_demand": "Actual", "predicted": "Predicted"})

x_domain = [window_start.isoformat(), window_end.isoformat()]

base = alt.Chart(plot_df).mark_line(point=True).encode(
    x=alt.X(
        "date:T",
        scale=alt.Scale(domain=x_domain),
        axis=alt.Axis(
            title="Date",
            titlePadding=18,
            format="%d %b %Y",
            labelAngle=-25,
        ),
    ),
    y=alt.Y("MW:Q", title="Peak demand (MW)", scale=alt.Scale(zero=False)),
    color=alt.Color("series:N", title="", scale=alt.Scale(range=["#1f77b4", "#ff7f0e"])),
    tooltip=[
        alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("MW:Q", format=",.0f", title="MW"),
    ],
)
selected_rule = alt.Chart(pd.DataFrame({"date": [pd.Timestamp(selected)]})).mark_rule(
    color="grey", strokeDash=[6, 4]
).encode(x="date:T")

# Only show the train/test split rule when it actually lies inside the window;
# otherwise Altair extends the X axis to include it and the chart looks wrong.
layers = [base, selected_rule]
if window_start <= split <= window_end:
    split_rule = alt.Chart(pd.DataFrame({"date": [split]})).mark_rule(
        color="red", strokeDash=[2, 2], opacity=0.5
    ).encode(x="date:T")
    layers.append(split_rule)

# .interactive() intentionally omitted — unbounded zoom hurts the demo more than it helps.
chart = alt.layer(*layers).properties(
    height=420,
    padding={"top": 5, "right": 10, "bottom": 60, "left": 10},
)
st.altair_chart(chart, width="stretch")
st.caption(
    "Grey dashed line = selected date. Red dashed line (when visible) = train/test split. "
    "Hover for exact values."
)
