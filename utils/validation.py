from __future__ import annotations

import math
from typing import Any

import pandas as pd

from config import DIABETES_RANGES, DISEASES, HEART_CATEGORIES, HEART_RANGES


def validate_patient_input(disease: str, values: dict[str, Any]) -> list[str]:
    if disease not in DISEASES:
        return [f"Unknown disease: {disease}"]

    errors: list[str] = []
    specification = DISEASES[disease]
    expected = specification["features"]
    missing = [feature for feature in expected if feature not in values or values[feature] is None]
    if missing:
        errors.append("Enter all required values: " + ", ".join(missing))
        return errors

    for feature in expected:
        value = values[feature]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{feature} must be a number.")
            continue
        if not math.isfinite(numeric):
            errors.append(f"{feature} must be a finite number.")
            continue

        if disease == "breast_cancer":
            if numeric < 0:
                errors.append(f"{feature} must be zero or greater.")
        elif disease == "heart_disease":
            lower, upper = HEART_RANGES[feature]
            if not lower <= numeric <= upper:
                errors.append(f"{feature} must be between {lower} and {upper}.")
            elif feature in HEART_CATEGORIES and numeric not in HEART_CATEGORIES[feature]:
                allowed = ", ".join(map(str, HEART_CATEGORIES[feature]))
                errors.append(f"{feature} must be one of: {allowed}.")
        else:
            lower, upper = DIABETES_RANGES[feature]
            if not lower <= numeric <= upper:
                errors.append(f"{feature} must be between {lower} and {upper}.")

    return errors


def patient_frame(disease: str, values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{feature: float(values[feature]) for feature in DISEASES[disease]["features"]}],
        columns=DISEASES[disease]["features"],
    )
