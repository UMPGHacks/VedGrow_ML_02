"""Train and persist house price prediction models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import impute_missing_values, prepare_training_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "AmesHousing.csv"
MODELS_DIR = PROJECT_ROOT / "models"

INPUT_COLUMNS = [
    "MS Zoning",
    "Lot Area",
    "Lot Shape",
    "Neighborhood",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Exter Qual",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Area",
    "Foundation",
    "Bsmt Qual",
    "BsmtFin Type 1",
    "BsmtFin SF 1",
    "Total Bsmt SF",
    "Heating QC",
    "Central Air",
    "1st Flr SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Kitchen Qual",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Type",
    "Garage Yr Blt",
    "Garage Finish",
    "Garage Cars",
    "Garage Area",
    "Paved Drive",
    "Wood Deck SF",
    "Open Porch SF",
    "Yr Sold",
]


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_defaults(raw_df: pd.DataFrame) -> dict:
    defaults = {}
    for column in raw_df.columns:
        if column == "SalePrice":
            continue
        if pd.api.types.is_numeric_dtype(raw_df[column]):
            defaults[column] = float(raw_df[column].median())
        else:
            defaults[column] = raw_df[column].mode().iloc[0]
    return defaults


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    x, y, artifacts = prepare_training_data(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    artifacts.scaler = scaler

    models = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=10),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        if name == "random_forest":
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
        else:
            model.fit(x_train_scaled, y_train)
            predictions = model.predict(x_test_scaled)
        metrics[name] = evaluate_model(y_test, predictions)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    joblib.dump(artifacts, MODELS_DIR / "preprocessing_artifacts.joblib")

    raw_df = pd.read_csv(DATA_PATH)
    raw_df, _, _, _, _ = impute_missing_values(raw_df)

    metadata = {
        "feature_columns": artifacts.feature_columns,
        "metrics": metrics,
        "best_model": "random_forest",
        "defaults": build_defaults(raw_df),
        "form_defaults": {column: build_defaults(raw_df)[column] for column in INPUT_COLUMNS if column in raw_df.columns},
        "categorical_options": {
            column: sorted(raw_df[column].dropna().unique().tolist())
            for column in raw_df.select_dtypes(exclude=np.number).columns
            if column in INPUT_COLUMNS
        },
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Models saved to:", MODELS_DIR)
    for name, scores in metrics.items():
        print(f"\n{name.replace('_', ' ').title()}")
        print(f"  MAE : {scores['mae']:,.2f}")
        print(f"  RMSE: {scores['rmse']:,.2f}")
        print(f"  R2  : {scores['r2']:.3f}")


if __name__ == "__main__":
    main()
