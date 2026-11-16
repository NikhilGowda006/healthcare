from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DIABETES_ZERO_AS_MISSING, DISEASES


class DatasetValidationError(ValueError):
    """Raised when a source dataset does not match its documented schema."""


def validate_dataset(disease: str, dataframe: pd.DataFrame) -> None:
    if disease not in DISEASES:
        raise DatasetValidationError(f"Unknown disease: {disease}")

    specification = DISEASES[disease]
    target = specification["target"]
    expected = specification["features"] + [target]
    if disease == "breast_cancer":
        expected = ["id", target] + specification["features"]

    if list(dataframe.columns) != expected:
        raise DatasetValidationError(
            f"{disease} columns do not match the expected schema. "
            f"Expected {expected}; received {list(dataframe.columns)}."
        )
    if dataframe.empty:
        raise DatasetValidationError(f"{disease} dataset contains no records.")

    if disease == "breast_cancer":
        targets = set(dataframe[target].dropna().astype(str))
        if not targets.issubset({"B", "M"}) or targets != {"B", "M"}:
            raise DatasetValidationError("Breast cancer diagnosis must contain B and M.")
    elif disease == "heart_disease":
        numeric = dataframe[specification["features"] + [target]].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric[target].isna().any() or not set(numeric[target].unique()).issubset(
            {0, 1, 2, 3, 4}
        ):
            raise DatasetValidationError("Heart target `num` must contain values 0–4.")
    elif disease == "diabetes":
        numeric = dataframe[expected].apply(pd.to_numeric, errors="coerce")
        original_nonempty = dataframe[expected].apply(
            lambda column: column.notna() & column.astype(str).str.strip().ne("")
        )
        if (numeric.isna() & original_nonempty).any().any():
            raise DatasetValidationError("Diabetes dataset contains non-numeric values.")
        if not set(numeric[target].dropna().unique()).issubset({0, 1}):
            raise DatasetValidationError("Diabetes Outcome must contain only 0 and 1.")
    elif disease == "breast_cancer":
        numeric = dataframe[["id"] + specification["features"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.isna().any().any():
            raise DatasetValidationError("Breast cancer ID and measurements must be numeric.")
        if dataframe["id"].duplicated().any():
            raise DatasetValidationError("Breast cancer ID values must be unique.")


def load_dataset(disease: str, data_dir: Path | None = None) -> pd.DataFrame:
    if disease not in DISEASES:
        raise DatasetValidationError(f"Unknown disease: {disease}")
    path = (data_dir or Path(__file__).resolve().parents[1] / "datasets") / DISEASES[disease]["dataset"]
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(
        path,
        na_values=["?", ""],
        keep_default_na=True,
    )
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    validate_dataset(disease, dataframe)

    if disease == "heart_disease":
        for column in DISEASES[disease]["features"] + [DISEASES[disease]["target"]]:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        dataframe[DISEASES[disease]["target"]] = dataframe[
            DISEASES[disease]["target"]
        ].astype(int)
    elif disease == "diabetes":
        for column in DISEASES[disease]["features"] + [DISEASES[disease]["target"]]:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        for column in DIABETES_ZERO_AS_MISSING:
            dataframe.loc[dataframe[column] == 0, column] = np.nan
    else:
        for column in DISEASES[disease]["features"]:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    return dataframe


def dataset_analysis(disease: str, dataframe: pd.DataFrame) -> dict[str, Any]:
    specification = DISEASES[disease]
    target = specification["target"]
    feature_columns = specification["features"]
    analysis: dict[str, Any] = {
        "disease": disease,
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "columns": list(dataframe.columns),
        "dtypes": {key: str(value) for key, value in dataframe.dtypes.items()},
        "missing_values": {
            key: int(value) for key, value in dataframe.isna().sum().items()
        },
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "class_distribution": {
            str(key): int(value)
            for key, value in dataframe[target].value_counts(dropna=False).items()
        },
        "categorical_features": specification["categorical"],
        "numeric_summary": dataframe[feature_columns].describe().to_dict(),
    }
    return analysis


def save_processed_dataset(
    disease: str,
    dataframe: pd.DataFrame,
    data_dir: Path | None = None,
) -> Path:
    specification = DISEASES[disease]
    target_column = specification["target"]
    processed = dataframe[specification["features"]].copy()
    if disease == "breast_cancer":
        processed["target"] = (
            dataframe[target_column] == specification["positive_value"]
        ).astype(int)
    elif disease == "heart_disease":
        processed["target"] = (dataframe[target_column] > 0).astype(int)
    else:
        processed["target"] = dataframe[target_column].astype(int)

    output_dir = (data_dir or Path(__file__).resolve().parents[1] / "datasets") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{disease}_processed.csv"
    processed.to_csv(output_path, index=False)
    return output_path


def save_dataset_analysis(disease: str, analysis: dict[str, Any], reports_dir: Path) -> None:
    output = reports_dir / disease
    output.mkdir(parents=True, exist_ok=True)
    path = output / "dataset_analysis.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2, default=str)
