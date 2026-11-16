from __future__ import annotations

import json
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from config import DISEASES, HEART_CATEGORIES, REPORTS_DIR
from preprocessing.datasets import load_dataset
from utils.reporting import prediction_report


class DashboardTests(unittest.TestCase):
    def test_home_dashboard_buttons_navigate_to_each_disease(self) -> None:
        for page in ("Breast Cancer", "Heart Disease", "Diabetes", "Analytics"):
            with self.subTest(page=page):
                app = AppTest.from_file("app.py").run(timeout=30)
                app.button(key=f"open_{page}").click().run(timeout=30)
                self.assertEqual(list(app.exception), [])
                self.assertEqual(app.session_state["main_navigation"], page)

    def test_each_navigation_page_renders_without_exceptions(self) -> None:
        for page in (
            "Overview",
            "Breast Cancer",
            "Heart Disease",
            "Diabetes",
            "Analytics",
            "About",
        ):
            with self.subTest(page=page):
                app = AppTest.from_file("app.py").run(timeout=30)
                app.radio[0].set_value(page).run(timeout=30)
                self.assertEqual(list(app.exception), [])

    def test_compact_navigation_menu_routes_to_selected_page(self) -> None:
        app = AppTest.from_file("app.py").run(timeout=30)
        app.selectbox[0].set_value("Analytics").run(timeout=30)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["main_navigation"], "Analytics")

    def test_disease_workspaces_have_no_medical_upload_controls(self) -> None:
        for page in ("Breast Cancer", "Heart Disease", "Diabetes"):
            with self.subTest(page=page):
                app = AppTest.from_file("app.py").run(timeout=30)
                app.radio[0].set_value(page).run(timeout=30)

                self.assertEqual(list(app.exception), [])
                self.assertEqual(list(app.file_uploader), [])

    def test_prediction_report_lists_manual_inputs_without_long_disclaimer(self) -> None:
        report_bytes = prediction_report(
            "diabetes",
            "logistic_regression",
            {
                "prediction_label": "Diabetes class predicted",
                "positive_probability": 0.6,
                "threshold": 0.5,
            },
            {"Age": 30},
        )
        report = json.loads(report_bytes)

        self.assertEqual(report["input_sources"], {"Age": "Manually entered"})
        self.assertNotIn("disclaimer", report)

    def test_model_performance_uses_saved_test_metrics(self) -> None:
        app = AppTest.from_file("app.py").run(timeout=30)
        app.radio[0].set_value("Analytics").run(timeout=30)
        with (REPORTS_DIR / "breast_cancer" / "evaluation.json").open(
            encoding="utf-8"
        ) as file:
            report = json.load(file)
        expected_accuracy = report["models"]["logistic_regression"]["test_metrics"][
            "accuracy"
        ]

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.metric), 45)
        self.assertEqual(app.metric[0].value, f"{expected_accuracy:.1%}")

    def test_manual_prediction_submission_on_each_disease_page(self) -> None:
        page_names = {
            "breast_cancer": "Breast Cancer",
            "heart_disease": "Heart Disease",
            "diabetes": "Diabetes",
        }
        for disease, page in page_names.items():
            with self.subTest(disease=disease):
                app = AppTest.from_file("app.py").run(timeout=30)
                app.radio[0].set_value(page).run(timeout=30)
                row = load_dataset(disease).iloc[0]
                for feature in DISEASES[disease]["features"]:
                    value = row[feature]
                    if pd.isna(value):
                        value = 100.0 if disease == "diabetes" else 0.0
                    if disease == "heart_disease" and feature in HEART_CATEGORIES:
                        if int(value) not in HEART_CATEGORIES[feature]:
                            value = next(iter(HEART_CATEGORIES[feature]))
                        app.selectbox(key=f"{disease}_input_{feature}").set_value(int(value))
                    else:
                        app.number_input(key=f"{disease}_input_{feature}").set_value(
                            float(value)
                        )
                app.button[0].click().run(timeout=30)
                self.assertEqual(list(app.exception), [])
                self.assertEqual(len(app.metric), 1)
                self.assertEqual(
                    [button.label for button in app.get("download_button")],
                    ["Download prediction report (JSON)"],
                )


if __name__ == "__main__":
    unittest.main()
