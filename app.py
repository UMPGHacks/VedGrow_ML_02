"""Streamlit web app for house price prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import prepare_prediction_row  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"

MODEL_LABELS = {
    "random_forest": "Random Forest (best)",
    "linear_regression": "Linear Regression",
    "ridge_regression": "Ridge Regression",
}


@st.cache_resource
def load_artifacts():
    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    preprocessing = joblib.load(MODELS_DIR / "preprocessing_artifacts.joblib")
    models = {
        name: joblib.load(MODELS_DIR / f"{name}.joblib")
        for name in MODEL_LABELS
        if (MODELS_DIR / f"{name}.joblib").exists()
    }
    return metadata, preprocessing, models


def predict_price(model_name: str, raw_row: dict, metadata, preprocessing, models) -> float:
    features = prepare_prediction_row(raw_row, preprocessing, metadata["defaults"])
    model = models[model_name]

    if model_name == "random_forest":
        return float(model.predict(features)[0])

    scaled = preprocessing.scaler.transform(features)
    return float(model.predict(scaled)[0])


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def render_metric_cards(metadata) -> None:
    metrics = metadata["metrics"]
    best = metadata["best_model"]
    cols = st.columns(3)
    cols[0].metric("Best Model", MODEL_LABELS[best])
    cols[1].metric("R² Score", f"{metrics[best]['r2']:.3f}")
    cols[2].metric("RMSE", format_currency(metrics[best]["rmse"]))


def build_input_form(defaults: dict, options: dict) -> dict:
    st.subheader("Property Details")

    tab_location, tab_structure, tab_interior, tab_garage = st.tabs(
        ["Location & Lot", "Structure", "Interior", "Garage & Extras"]
    )

    values: dict = {}

    with tab_location:
        col1, col2 = st.columns(2)
        with col1:
            values["MS Zoning"] = st.selectbox(
                "MS Zoning",
                options["MS Zoning"],
                index=options["MS Zoning"].index(defaults["MS Zoning"]),
            )
            values["Lot Shape"] = st.selectbox(
                "Lot Shape",
                options["Lot Shape"],
                index=options["Lot Shape"].index(defaults["Lot Shape"]),
            )
        with col2:
            values["Neighborhood"] = st.selectbox(
                "Neighborhood",
                options["Neighborhood"],
                index=options["Neighborhood"].index(defaults["Neighborhood"]),
            )
            values["Lot Area"] = st.number_input(
                "Lot Area (sq ft)",
                min_value=1000,
                max_value=100000,
                value=int(defaults["Lot Area"]),
                step=100,
            )

    with tab_structure:
        col1, col2, col3 = st.columns(3)
        with col1:
            values["Overall Qual"] = st.slider(
                "Overall Quality (1-10)",
                min_value=1,
                max_value=10,
                value=int(defaults["Overall Qual"]),
            )
            values["Overall Cond"] = st.slider(
                "Overall Condition (1-10)",
                min_value=1,
                max_value=10,
                value=int(defaults["Overall Cond"]),
            )
        with col2:
            values["Year Built"] = st.number_input(
                "Year Built",
                min_value=1872,
                max_value=2026,
                value=int(defaults["Year Built"]),
            )
            values["Year Remod/Add"] = st.number_input(
                "Year Remodeled",
                min_value=1872,
                max_value=2026,
                value=int(defaults["Year Remod/Add"]),
            )
            values["Yr Sold"] = st.number_input(
                "Year Sold",
                min_value=2006,
                max_value=2026,
                value=int(defaults["Yr Sold"]),
            )
        with col3:
            values["Exter Qual"] = st.selectbox(
                "Exterior Quality",
                options["Exter Qual"],
                index=options["Exter Qual"].index(defaults["Exter Qual"]),
            )
            values["Foundation"] = st.selectbox(
                "Foundation",
                options["Foundation"],
                index=options["Foundation"].index(defaults["Foundation"]),
            )
            values["Exterior 1st"] = st.selectbox(
                "Exterior 1st",
                options["Exterior 1st"],
                index=options["Exterior 1st"].index(defaults["Exterior 1st"]),
            )
            values["Exterior 2nd"] = st.selectbox(
                "Exterior 2nd",
                options["Exterior 2nd"],
                index=options["Exterior 2nd"].index(defaults["Exterior 2nd"]),
            )

    with tab_interior:
        col1, col2, col3 = st.columns(3)
        with col1:
            values["Gr Liv Area"] = st.number_input(
                "Above-ground Living Area (sq ft)",
                min_value=300,
                max_value=6000,
                value=int(defaults["Gr Liv Area"]),
                step=50,
            )
            values["1st Flr SF"] = st.number_input(
                "1st Floor Area (sq ft)",
                min_value=300,
                max_value=4000,
                value=int(defaults["1st Flr SF"]),
                step=50,
            )
            values["Total Bsmt SF"] = st.number_input(
                "Total Basement Area (sq ft)",
                min_value=0,
                max_value=2500,
                value=int(defaults["Total Bsmt SF"]),
                step=50,
            )
            values["BsmtFin SF 1"] = st.number_input(
                "Finished Basement Area (sq ft)",
                min_value=0,
                max_value=2500,
                value=int(defaults["BsmtFin SF 1"]),
                step=50,
            )
        with col2:
            values["Full Bath"] = st.number_input(
                "Full Baths",
                min_value=0,
                max_value=4,
                value=int(defaults["Full Bath"]),
            )
            values["Half Bath"] = st.number_input(
                "Half Baths",
                min_value=0,
                max_value=3,
                value=int(defaults["Half Bath"]),
            )
            values["Bsmt Full Bath"] = st.number_input(
                "Basement Full Baths",
                min_value=0,
                max_value=3,
                value=int(defaults["Bsmt Full Bath"]),
            )
            values["Bsmt Half Bath"] = st.number_input(
                "Basement Half Baths",
                min_value=0,
                max_value=2,
                value=int(defaults["Bsmt Half Bath"]),
            )
        with col3:
            values["Kitchen Qual"] = st.selectbox(
                "Kitchen Quality",
                options["Kitchen Qual"],
                index=options["Kitchen Qual"].index(defaults["Kitchen Qual"]),
            )
            values["Bsmt Qual"] = st.selectbox(
                "Basement Quality",
                options["Bsmt Qual"],
                index=options["Bsmt Qual"].index(defaults["Bsmt Qual"]),
            )
            values["BsmtFin Type 1"] = st.selectbox(
                "Basement Finish Type",
                options["BsmtFin Type 1"],
                index=options["BsmtFin Type 1"].index(defaults["BsmtFin Type 1"]),
            )
            values["Heating QC"] = st.selectbox(
                "Heating Quality",
                options["Heating QC"],
                index=options["Heating QC"].index(defaults["Heating QC"]),
            )
            values["Central Air"] = st.selectbox(
                "Central Air",
                options["Central Air"],
                index=options["Central Air"].index(defaults["Central Air"]),
            )
            values["TotRms AbvGrd"] = st.number_input(
                "Total Rooms Above Grade",
                min_value=2,
                max_value=15,
                value=int(defaults["TotRms AbvGrd"]),
            )
            values["Fireplaces"] = st.number_input(
                "Fireplaces",
                min_value=0,
                max_value=4,
                value=int(defaults["Fireplaces"]),
            )
            values["Mas Vnr Area"] = st.number_input(
                "Masonry Veneer Area (sq ft)",
                min_value=0,
                max_value=2000,
                value=int(defaults["Mas Vnr Area"]),
                step=10,
            )

    with tab_garage:
        col1, col2 = st.columns(2)
        with col1:
            values["Garage Type"] = st.selectbox(
                "Garage Type",
                options["Garage Type"],
                index=options["Garage Type"].index(defaults["Garage Type"]),
            )
            values["Garage Finish"] = st.selectbox(
                "Garage Finish",
                options["Garage Finish"],
                index=options["Garage Finish"].index(defaults["Garage Finish"]),
            )
            values["Garage Cars"] = st.number_input(
                "Garage Capacity (cars)",
                min_value=0,
                max_value=4,
                value=int(defaults["Garage Cars"]),
            )
            values["Garage Area"] = st.number_input(
                "Garage Area (sq ft)",
                min_value=0,
                max_value=1000,
                value=int(defaults["Garage Area"]),
                step=10,
            )
        with col2:
            values["Garage Yr Blt"] = st.number_input(
                "Garage Year Built",
                min_value=1900,
                max_value=2026,
                value=int(defaults["Garage Yr Blt"]),
            )
            values["Paved Drive"] = st.selectbox(
                "Paved Drive",
                options["Paved Drive"],
                index=options["Paved Drive"].index(defaults["Paved Drive"]),
            )
            values["Wood Deck SF"] = st.number_input(
                "Wood Deck Area (sq ft)",
                min_value=0,
                max_value=1000,
                value=int(defaults["Wood Deck SF"]),
                step=10,
            )
            values["Open Porch SF"] = st.number_input(
                "Open Porch Area (sq ft)",
                min_value=0,
                max_value=500,
                value=int(defaults["Open Porch SF"]),
                step=10,
            )

    return values


def main() -> None:
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="🏠",
        layout="wide",
    )

    if not METADATA_PATH.exists():
        st.error(
            "Trained models were not found. Run `python src/train_model.py` first, then restart the app."
        )
        st.stop()

    metadata, preprocessing, models = load_artifacts()
    defaults = metadata.get("form_defaults", metadata["defaults"])
    options = metadata["categorical_options"]

    st.title("🏠 House Price Predictor")
    st.caption("Predict sale prices using the Ames Housing dataset and machine learning models from your notebook.")

    render_metric_cards(metadata)

    with st.sidebar:
        st.header("Model Settings")
        model_name = st.selectbox(
            "Choose model",
            options=list(MODEL_LABELS.keys()),
            format_func=lambda key: MODEL_LABELS[key],
            index=list(MODEL_LABELS.keys()).index(metadata["best_model"]),
        )

        st.divider()
        st.subheader("Model Performance")
        selected_metrics = metadata["metrics"][model_name]
        st.write(f"**MAE:** {format_currency(selected_metrics['mae'])}")
        st.write(f"**RMSE:** {format_currency(selected_metrics['rmse'])}")
        st.write(f"**R²:** {selected_metrics['r2']:.3f}")

        st.divider()
        st.markdown(
            """
            **Tips**
            - Overall Quality and living area strongly affect price.
            - Garage size and basement finish also matter.
            - Random Forest gives the most accurate predictions.
            """
        )

    raw_row = build_input_form(defaults, options)
    house_age = int(raw_row["Yr Sold"] - raw_row["Year Built"])
    total_bath = (
        raw_row["Full Bath"]
        + 0.5 * raw_row["Half Bath"]
        + raw_row["Bsmt Full Bath"]
        + 0.5 * raw_row["Bsmt Half Bath"]
    )

    st.info(f"Derived features — House Age: **{house_age} years** | Total Bathrooms: **{total_bath:.1f}**")

    if st.button("Predict Price", type="primary", use_container_width=True):
        try:
            predicted_price = predict_price(model_name, raw_row, metadata, preprocessing, models)
            st.success(f"Predicted Sale Price: **{format_currency(predicted_price)}**")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    with st.expander("Compare all models for current inputs"):
        comparison_rows = []
        for name in MODEL_LABELS:
            if name not in models:
                continue
            price = predict_price(name, raw_row, metadata, preprocessing, models)
            scores = metadata["metrics"][name]
            comparison_rows.append(
                {
                    "Model": MODEL_LABELS[name],
                    "Predicted Price": format_currency(price),
                    "Test R²": f"{scores['r2']:.3f}",
                    "Test RMSE": format_currency(scores["rmse"]),
                }
            )
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
