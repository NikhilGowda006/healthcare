from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from config import DISEASES


def prediction_report(
    disease: str,
    model_name: str,
    result: dict[str, Any],
    input_values: dict[str, Any],
) -> bytes:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "disease": DISEASES[disease]["title"],
        "selected_model": model_name,
        "predicted_class": result["prediction_label"],
        "model_positive_probability": result["positive_probability"],
        "decision_threshold": result["threshold"],
        "input_values": input_values,
        "input_sources": {
            feature: "Manually entered" for feature in input_values
        },
    }
    return json.dumps(report, indent=2, allow_nan=False).encode("utf-8")
