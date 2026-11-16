from __future__ import annotations

import unittest

from config import DISEASES
from utils.validation import validate_patient_input


class ValidationTests(unittest.TestCase):
    def test_missing_and_invalid_fields_are_reported(self) -> None:
        errors = validate_patient_input("diabetes", {"Pregnancies": 0})
        self.assertTrue(any("required values" in error for error in errors))

    def test_valid_dataset_row_features_pass(self) -> None:
        values = {
            "Pregnancies": 0,
            "Glucose": 110,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 90,
            "BMI": 28.0,
            "DiabetesPedigreeFunction": 0.4,
            "Age": 30,
        }
        self.assertEqual(validate_patient_input("diabetes", values), [])

    def test_breast_cancer_requires_nonnegative_measurements(self) -> None:
        values = {feature: 1.0 for feature in DISEASES["breast_cancer"]["features"]}
        values["radius_mean"] = -1
        self.assertTrue(validate_patient_input("breast_cancer", values))

    def test_heart_categories_are_constrained_to_supported_values(self) -> None:
        values = {
            "age": 55,
            "sex": 1,
            "cp": 2,
            "trestbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 4,
        }
        self.assertTrue(
            any(
                "thal must be one of" in error
                for error in validate_patient_input("heart_disease", values)
            )
        )


if __name__ == "__main__":
    unittest.main()
