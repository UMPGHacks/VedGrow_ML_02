"""Data preprocessing pipeline for the Ames Housing price prediction project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


CORRELATION_THRESHOLD = 0.30
TARGET_COLUMN = "SalePrice"


@dataclass
class PreprocessingArtifacts:
    numeric_imputer: SimpleImputer
    categorical_imputer: SimpleImputer
    numeric_columns: list[str]
    categorical_columns: list[str]
    dummy_columns: list[str]
    feature_columns: list[str]
    scaler: StandardScaler | None
    sale_price_iqr_bounds: tuple[float, float] | None = None


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer, SimpleImputer, list[str], list[str]]:
    df = df.copy()
    numeric_columns = [
        column
        for column in df.select_dtypes(include=np.number).columns.tolist()
        if column != TARGET_COLUMN
    ]
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    if numeric_columns:
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    if categorical_columns:
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    return df, numeric_imputer, categorical_imputer, numeric_columns, categorical_columns


def apply_imputation(
    df: pd.DataFrame,
    numeric_imputer: SimpleImputer,
    categorical_imputer: SimpleImputer,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    df = df.copy()

    # Use the schema stored by the fitted imputers rather than the artifact
    # lists alone.  This keeps inference compatible with artifacts produced by
    # an earlier version that accidentally included ``SalePrice`` among the
    # numeric columns.  The target is supplied as an all-missing temporary
    # column for that transform, then deliberately omitted from the result.
    numeric_schema = list(getattr(numeric_imputer, "feature_names_in_", numeric_columns))
    categorical_schema = list(getattr(categorical_imputer, "feature_names_in_", categorical_columns))

    if numeric_schema:
        numeric_input = df.reindex(columns=numeric_schema)
        numeric_values = numeric_imputer.transform(numeric_input)
        numeric_result = pd.DataFrame(numeric_values, index=df.index, columns=numeric_schema)
        numeric_cols = [column for column in numeric_schema if column != TARGET_COLUMN]
        df[numeric_cols] = numeric_result[numeric_cols]

    if categorical_schema:
        categorical_input = df.reindex(columns=categorical_schema)
        categorical_values = categorical_imputer.transform(categorical_input)
        categorical_result = pd.DataFrame(categorical_values, index=df.index, columns=categorical_schema)
        df[categorical_schema] = categorical_result[categorical_schema]
    return df


def remove_sale_price_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    q1 = df["SalePrice"].quantile(0.25)
    q3 = df["SalePrice"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = df[(df["SalePrice"] >= lower) & (df["SalePrice"] <= upper)].copy()
    return filtered, (lower, upper)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HouseAge"] = df["Yr Sold"] - df["Year Built"]
    df["TotalBath"] = (
        df["Full Bath"]
        + 0.5 * df["Half Bath"]
        + df["Bsmt Full Bath"]
        + 0.5 * df["Bsmt Half Bath"]
    )
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, drop_first=True)


def align_dummy_columns(df: pd.DataFrame, dummy_columns: list[str]) -> pd.DataFrame:
    aligned = pd.DataFrame(0, index=df.index, columns=dummy_columns, dtype=float)
    for column in df.columns:
        if column in aligned.columns:
            aligned[column] = df[column].values
    return aligned


def select_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    corr = df.corr(numeric_only=True)
    target_corr = corr["SalePrice"].abs().sort_values(ascending=False)
    selected = target_corr[target_corr > CORRELATION_THRESHOLD]

    if feature_columns is None:
        feature_columns = selected.index.drop("SalePrice").tolist()

    x = df[feature_columns]
    y = df["SalePrice"]
    return x, y, feature_columns


def prepare_training_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series, PreprocessingArtifacts]:
    df = load_raw_data(data_path)
    df, numeric_imputer, categorical_imputer, numeric_columns, categorical_columns = impute_missing_values(df)
    df, iqr_bounds = remove_sale_price_outliers(df)
    df = engineer_features(df)
    encoded = encode_categorical_features(df)
    dummy_columns = [column for column in encoded.columns if column != "SalePrice"]

    x, y, feature_columns = select_features(encoded)
    artifacts = PreprocessingArtifacts(
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        dummy_columns=dummy_columns,
        feature_columns=feature_columns,
        scaler=None,
        sale_price_iqr_bounds=iqr_bounds,
    )
    return x, y, artifacts


def build_full_row(partial_row: dict, full_defaults: dict) -> dict:
    row = dict(full_defaults)
    row.update(partial_row)
    return row


def prepare_prediction_row(
    raw_row: dict,
    artifacts: PreprocessingArtifacts,
    full_defaults: dict | None = None,
) -> pd.DataFrame:
    if full_defaults is not None:
        raw_row = build_full_row(raw_row, full_defaults)

    raw_row.pop(TARGET_COLUMN, None)
    df = pd.DataFrame([raw_row])
    df = apply_imputation(
        df,
        artifacts.numeric_imputer,
        artifacts.categorical_imputer,
        artifacts.numeric_columns,
        artifacts.categorical_columns,
    )
    df = engineer_features(df)
    encoded = encode_categorical_features(df)
    aligned = align_dummy_columns(encoded, artifacts.dummy_columns)
    return aligned[artifacts.feature_columns]
