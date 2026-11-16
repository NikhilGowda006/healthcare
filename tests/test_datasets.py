from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from config import DIABETES_ZERO_AS_MISSING, DISEASES
from preprocessing.datasets import (
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)


class DatasetTests(unittest.TestCase):
    def test_all_sources_match_their_declared_schemas(self) -> None:
        expected_rows = {"breast_cancer": 569, "heart_disease": 303, "diabetes": 768}
        for disease, expected in expected_rows.items():
            with self.subTest(disease=disease):
                dataframe = load_dataset(disease)
                self.assertEqual(len(dataframe), expected)
                self.assertEqual(
                    list(dataframe.columns),
                    (
                        ["id", DISEASES[disease]["target"]]
                        + DISEASES[disease]["features"]
                        if disease == "breast_cancer"
                        else DISEASES[disease]["features"] + [DISEASES[disease]["target"]]
                    ),
                )

    def test_heart_question_marks_are_loaded_as_missing(self) -> None:
        dataframe = load_dataset("heart_disease")
        self.assertGreater(int(dataframe[["ca", "thal"]].isna().sum().sum()), 0)

    def test_diabetes_dataset_zeros_are_missing_except_pregnancies(self) -> None:
        dataframe = load_dataset("diabetes")
        self.assertTrue(dataframe.loc[0, "Pregnancies"] == 6)
        self.assertTrue(pd.isna(dataframe.loc[0, "Insulin"]))
        self.assertTrue(set(DIABETES_ZERO_AS_MISSING).isdisjoint({"Pregnancies"}))

    def test_incompatible_dataset_schema_is_rejected(self) -> None:
        with self.assertRaises(DatasetValidationError):
            validate_dataset("heart_disease", pd.DataFrame({"Outcome": [0]}))

    def test_processed_exports_drop_identifiers_and_encode_targets(self) -> None:
        from preprocessing.datasets import save_processed_dataset

        with tempfile.TemporaryDirectory() as temporary:
            breast_path = save_processed_dataset(
                "breast_cancer", load_dataset("breast_cancer"), Path(temporary)
            )
            breast = pd.read_csv(breast_path)
            self.assertNotIn("id", breast.columns)
            self.assertEqual(set(breast["target"].unique()), {0, 1})

            heart_path = save_processed_dataset(
                "heart_disease", load_dataset("heart_disease"), Path(temporary)
            )
            heart = pd.read_csv(heart_path)
            self.assertEqual(set(heart["target"].unique()), {0, 1})


if __name__ == "__main__":
    unittest.main()
