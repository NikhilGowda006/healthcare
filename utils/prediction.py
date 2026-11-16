from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import DISEASES, MODELS_DIR
from utils.validation import patient_frame, validate_patient_input


class ModelNotFoundError(FileNotFoundError):
    pass


def load_model(
    disease: str, model_name: str, models_dir: Path | None = None
) -> tuple[Any, dict[str, Any]]:
    root = models_dir or MODELS_DIR
    if disease not in DISEASES:
        raise ValueError(f"Unknown disease: {disease}")
    model_path = root / disease / f"{model_name}.joblib"
    metadata_path = root / disease / "metadata.json"
    if not model_path.is_file():
        raise ModelNotFoundError(
            f"Trained model is missing: {model_path}. Run `python -m training.train_all`."
        )
    if not metadata_path.is_file():
        raise ModelNotFoundError(
            f"Model metadata is missing: {metadata_path}. Run `python -m training.train_all`."
        )
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    pipeline = joblib.load(model_path)
    expected = DISEASES[disease]["features"]
    if metadata.get("feature_names") != expected:
        raise ValueError(f"Saved {disease} model feature metadata does not match config.")
    return pipeline, metadata


def predict(
    disease: str,
    model_name: str,
    values: dict[str, Any],
    models_dir: Path | None = None,
) -> dict[str, Any]:
    errors = validate_patient_input(disease, values)
    if errors:
        raise ValueError(" ".join(errors))

    pipeline, metadata = load_model(disease, model_name, models_dir)
    frame = patient_frame(disease, values)
    probabilities = pipeline.predict_proba(frame)[0]
    classes = list(pipeline.classes_)
    positive = DISEASES[disease]["positive_value"]
    positive_index = classes.index(1)
    probability_positive = float(probabilities[positive_index])
    threshold = float(metadata["models"][model_name]["threshold"])
    positive_class = probability_positive >= threshold
    prediction = positive if positive_class else DISEASES[disease]["negative_value"]

    return {
        "disease": disease,
        "model": model_name,
        "prediction": prediction,
        "prediction_label": (
            DISEASES[disease]["positive_label"]
            if positive_class
            else DISEASES[disease]["negative_label"]
        ),
        "positive_probability": probability_positive,
        "threshold": threshold,
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        },
    }
