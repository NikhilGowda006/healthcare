from __future__ import annotations

import json

from config import DISEASES, REPORTS_DIR
from training.common import train_disease


def main() -> None:
    summaries = [train_disease(disease) for disease in DISEASES]
    output = REPORTS_DIR / "training_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    for summary in summaries:
        print(
            f"{summary['disease']}: best model by validation metric rank is "
            f"{summary['best_model_validation_rank']}"
        )
    print(f"Training reports saved under {REPORTS_DIR}")


if __name__ == "__main__":
    main()
