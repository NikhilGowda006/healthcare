from __future__ import annotations

import unittest

import pandas as pd

from config import DISEASES
from preprocessing.datasets import load_dataset
from utils.prediction import ModelNotFoundError, load_model, predict


class PredictionTests(unittest.TestCase):
    def test_each_disease_loads_a_disease_specific_model(self) -> None:
        for disease in DISEASES:
            with self.subTest(disease=disease):
                pipeline, metadata = load_model(disease, "logistic_regression")
                self.assertEqual(metadata["disease"], disease)
                self.assertEqual(
                    list(pipeline.feature_names_in_),
                    DISEASES[disease]["features"],
                )

    def test_prediction_uses_matching_pipeline_for_each_disease(self) -> None:
        for disease, specification in DISEASES.items():
            with self.subTest(disease=disease):
                row = load_dataset(disease).iloc[0]
                values = {
                    feature: (
                        100.0 if pd.isna(row[feature]) else float(row[feature])
                    )
                    for feature in specification["features"]
                }
                result = predict(disease, "logistic_regression", values)
                self.assertEqual(result["disease"], disease)
                self.assertIn(result["prediction"], (0, 1, "B", "M"))
                self.assertGreaterEqual(result["positive_probability"], 0.0)
                self.assertLessEqual(result["positive_probability"], 1.0)

    def test_missing_model_has_an_actionable_error(self) -> None:
        with self.assertRaisesRegex(ModelNotFoundError, "training.train_all"):
            load_model("diabetes", "not_a_model")


if __name__ == "__main__":
    unittest.main()
