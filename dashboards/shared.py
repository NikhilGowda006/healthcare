from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from config import (
    DIABETES_RANGES,
    DISEASES,
    HEART_CATEGORIES,
    HEART_RANGES,
    MODELS_DIR,
    REPORTS_DIR,
)
from utils.prediction import ModelNotFoundError, predict
from utils.reporting import prediction_report
from utils.validation import validate_patient_input

MODEL_NAMES = ("logistic_regression", "random_forest", "svm")
DIABETES_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood pressure (mmHg)",
    "SkinThickness": "Skin thickness (mm)",
    "Insulin": "Insulin (mu U/mL)",
    "BMI": "BMI (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age (years)",
}
HEART_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar > 120 mg/dL",
    "restecg": "Resting ECG",
    "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "Oldpeak (ST depression)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels (0–3)",
    "thal": "Thalassemia",
}


def _read_model_metadata(disease: str) -> dict[str, Any] | None:
    path = MODELS_DIR / disease / "metadata.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_evaluation_report(disease: str) -> dict[str, Any] | None:
    path = REPORTS_DIR / disease / "evaluation.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _field_label(disease: str, feature: str) -> str:
    if disease == "heart_disease":
        return HEART_LABELS[feature]
    if disease == "diabetes":
        return DIABETES_LABELS[feature]
    return feature.replace("_", " ").title()


def _field_help(disease: str, feature: str) -> str:
    help_text = {
        "diabetes": {
            "Pregnancies": "Number of pregnancies recorded in the source dataset.",
            "Glucose": "Plasma glucose concentration (mg/dL).",
            "BloodPressure": "Diastolic blood pressure (mmHg).",
            "SkinThickness": "Triceps skin-fold thickness (mm).",
            "Insulin": "Two-hour serum insulin (mu U/mL).",
            "BMI": "Body mass index (kg/m²).",
            "DiabetesPedigreeFunction": "Dataset pedigree function value; use the recorded numeric value.",
            "Age": "Age in years; this dataset represents women aged 21 and older.",
        },
        "heart_disease": {
            "age": "Age in years.",
            "sex": "Dataset encoding: 0 = female, 1 = male.",
            "cp": "Chest-pain type using the dataset's categories 1–4.",
            "trestbps": "Resting blood pressure (mmHg).",
            "chol": "Serum cholesterol (mg/dL).",
            "fbs": "Fasting blood sugar above 120 mg/dL: 0 = no, 1 = yes.",
            "restecg": "Resting ECG result using the dataset's categories.",
            "thalach": "Maximum heart rate achieved.",
            "exang": "Exercise-induced angina: 0 = no, 1 = yes.",
            "oldpeak": "ST depression induced by exercise relative to rest.",
            "slope": "Slope of the peak exercise ST segment.",
            "ca": "Number of major vessels (0–3).",
            "thal": "Thalassemia category used by the source dataset.",
        },
    }
    return help_text.get(disease, {}).get(
        feature,
        "Enter the measurement shown in the report, using the same units.",
    )


def _render_input(disease: str, feature: str) -> Any:
    key = f"{disease}_input_{feature}"
    label = _field_label(disease, feature)
    if disease == "heart_disease" and feature in HEART_CATEGORIES:
        options = [None] + list(HEART_CATEGORIES[feature])

        def display(value: int | None) -> str:
            return "Select a value" if value is None else HEART_CATEGORIES[feature][value]

        return st.selectbox(
            label,
            options,
            format_func=display,
            key=key,
            help=_field_help(disease, feature),
        )

    if disease == "heart_disease":
        minimum, maximum = HEART_RANGES[feature]
        step = 0.1 if feature == "oldpeak" else 1.0
    elif disease == "diabetes":
        minimum, maximum = DIABETES_RANGES[feature]
        step = 0.01 if feature in {"BMI", "DiabetesPedigreeFunction"} else 1.0
    else:
        minimum, maximum, step = 0.0, None, 0.01
    return st.number_input(
        label,
        min_value=float(minimum),
        max_value=float(maximum) if maximum is not None else None,
        value=None,
        step=float(step),
        key=key,
        help=_field_help(disease, feature),
    )


def render_disease_dashboard(disease: str) -> None:
    specification = DISEASES[disease]
    disease_titles = {
        "breast_cancer": "Breast Cancer Prediction",
        "heart_disease": "Heart Disease Prediction",
        "diabetes": "Diabetes Prediction",
    }
    disease_descriptions = {
        "breast_cancer": (
            "Explore the WDBC model using the original digitized fine-needle "
            "aspirate measurements."
        ),
        "heart_disease": (
            "Review the Cleveland dataset features and explore a model estimate "
            "alongside its held-out evaluation."
        ),
        "diabetes": (
            "Explore the Pima dataset model using its original clinical and "
            "demographic fields."
        ),
    }
    disease_accents = {
        "breast_cancer": "#8B5CF6",
        "heart_disease": "#22D3EE",
        "diabetes": "#34D399",
    }
    st.markdown(
        f"""
        <section class="page-hero" style="border-left-color: {disease_accents[disease]}">
            <div class="eyebrow">{specification['title']} · model workspace</div>
            <h1>{disease_titles[disease]}</h1>
            <p>{disease_descriptions[disease]}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.warning(specification["limitations"])
    st.caption(
        f"Dataset: {specification['source']}. "
        f"Model outputs are educational estimates, not diagnoses."
    )

    metadata = _read_model_metadata(disease)
    best_model = metadata.get("best_model_validation_rank") if metadata else None
    default_model = best_model if best_model in MODEL_NAMES else MODEL_NAMES[0]
    model_name = st.selectbox(
        "Model",
        MODEL_NAMES,
        index=MODEL_NAMES.index(default_model),
        format_func=lambda value: value.replace("_", " ").title(),
        key=f"{disease}_model",
    )
    if metadata:
        st.caption(
            "Recommended by validation-set metric ranking: "
            f"{best_model.replace('_', ' ').title()}. "
            "This selection did not use the test set."
        )
    else:
        st.info("Models are not trained yet. Run `python -m training.train_all`.")

    st.subheader("Manual model inputs")
    st.caption(
        "Enter the original dataset features in their stated units. Every field is required."
    )
    with st.form(f"{disease}_prediction_form"):
        values: dict[str, Any] = {}
        if disease == "breast_cancer":
            sections = (
                ("Mean measurements", specification["features"][:10]),
                ("Standard error measurements", specification["features"][10:20]),
                ("Worst measurements", specification["features"][20:]),
            )
        elif disease == "heart_disease":
            sections = (
                ("Patient information", specification["features"][:2]),
                ("Heart measurements", specification["features"][2:8]),
                ("Exercise and diagnostic information", specification["features"][8:]),
            )
        elif disease == "diabetes":
            patient_features = {"Pregnancies", "Age"}
            blood_measurement_features = {
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
            }
            sections = (
                (
                    "Patient information",
                    [
                        feature
                        for feature in specification["features"]
                        if feature in patient_features
                    ],
                ),
                (
                    "Glucose and blood measurements",
                    [
                        feature
                        for feature in specification["features"]
                        if feature in blood_measurement_features
                    ],
                ),
                (
                    "Additional health information",
                    [
                        feature
                        for feature in specification["features"]
                        if feature not in patient_features | blood_measurement_features
                    ],
                ),
            )
        else:
            sections = (("Required measurements", specification["features"]),)
        for section_title, features in sections:
            with st.container(border=True):
                st.markdown(f"##### {section_title}")
                columns = st.columns(2)
                for index, feature in enumerate(features):
                    with columns[index % len(columns)]:
                        values[feature] = _render_input(disease, feature)
        submitted = st.form_submit_button("Generate prediction")

    if not submitted:
        return

    errors = validate_patient_input(disease, values)
    if errors:
        for error in errors:
            st.error(error)
        return

    with st.spinner("Running the selected model..."):
        try:
            result = predict(disease, model_name, values)
        except ModelNotFoundError as exc:
            st.error(str(exc))
            return
        except (ValueError, OSError) as exc:
            st.error(f"Prediction could not be completed: {exc}")
            return

    st.subheader("Prediction result")
    is_positive = result["prediction"] == specification["positive_value"]
    result_style = "risk" if is_positive else ""
    st.markdown(
        f"""
        <div class="result-card {result_style}">
            <div class="result-label">Model output · {model_name.replace('_', ' ').title()}</div>
            <div class="result-value">{result['prediction_label']}</div>
            <p class="section-note">
                Model output for educational exploration; not a clinical finding.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(2)
    with metric_columns[0]:
        st.metric(
            "Estimated probability for the model's positive class",
            f"{result['positive_probability']:.1%}",
        )
    with metric_columns[1], st.container(border=True):
        st.markdown("**Selected model**")
        st.write(model_name.replace("_", " ").title())
        st.caption(
            f"Validation-selected threshold: {result['threshold']:.3f}. "
            "The probability is not clinically calibrated."
        )
    report_bytes = prediction_report(disease, model_name, result, values)
    st.download_button(
        "Download prediction report (JSON)",
        data=report_bytes,
        file_name=f"{disease}_prediction_report.json",
        mime="application/json",
    )

    model_details = metadata.get("models", {}).get(model_name) if metadata else None
    if model_details:
        with st.expander("Selected model's held-out test-set evaluation"):
            st.json(model_details["test_metrics"])
            st.caption(
                "Test-set metrics describe this dataset split only; they do not "
                "establish medical reliability."
            )
            plot = REPORTS_DIR / disease / f"{model_name}_confusion_matrix.png"
            if plot.is_file():
                st.image(str(plot), caption="Confusion matrix on held-out test data")


def render_model_performance() -> None:
    st.markdown('<div class="eyebrow">Saved evaluation reports</div>', unsafe_allow_html=True)
    st.title("Model performance")
    st.warning(
        "All reported test metrics are estimates on held-out portions of small, "
        "historical datasets. They do not establish clinical reliability."
    )
    for disease, specification in DISEASES.items():
        report = _read_evaluation_report(disease)
        st.header(specification["title"])
        if report is None:
            st.warning(
                f"Saved evaluation report is missing for {specification['title']}."
            )
            continue
        metadata = _read_model_metadata(disease)
        recommended = (
            metadata.get("best_model_validation_rank")
            if metadata
            else None
        )
        st.write(
            "Recommended by validation-set ranking: "
            f"**{recommended.replace('_', ' ').title() if recommended else 'Unavailable'}**"
        )
        for model_name, details in report["models"].items():
            metrics = details["test_metrics"]
            st.subheader(model_name.replace("_", " ").title())
            metric_columns = st.columns(5)
            for column, label, key in zip(
                metric_columns,
                ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"),
                ("accuracy", "precision", "recall_sensitivity", "f1", "roc_auc"),
            ):
                with column:
                    st.metric(label, f"{metrics[key]:.1%}")
            class_mapping = report["class_mapping"]
            class_names = [class_mapping[str(value)] for value in (0, 1)]
            confusion_matrix = pd.DataFrame(
                details["confusion_matrix"],
                index=[f"Actual · {name}" for name in class_names],
                columns=[f"Predicted · {name}" for name in class_names],
            )
            with st.expander(f"{model_name.replace('_', ' ').title()} confusion matrix"):
                st.dataframe(confusion_matrix, width="stretch")
                image = REPORTS_DIR / disease / f"{model_name}_confusion_matrix.png"
                if image.is_file():
                    st.image(str(image), caption="Saved held-out confusion matrix")
        columns = st.columns(2)
        for column, filename, caption in [
            (columns[0], "model_comparison.png", "Test-set model comparison"),
            (columns[1], "roc_curves.png", "Test-set ROC curves"),
        ]:
            image = REPORTS_DIR / disease / filename
            if image.is_file():
                column.image(str(image), caption=caption, width="stretch")
        evaluation = REPORTS_DIR / disease / "evaluation.json"
        if evaluation.is_file():
            with st.expander(f"{specification['title']} evaluation report"):
                st.download_button(
                    "Download evaluation JSON",
                    data=evaluation.read_bytes(),
                    file_name=f"{disease}_evaluation.json",
                    mime="application/json",
                    key=f"{disease}_evaluation_download",
                )
