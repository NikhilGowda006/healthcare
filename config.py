from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

DISEASES = {
    "breast_cancer": {
        "title": "Breast Cancer",
        "dataset": "breast_cancer.csv",
        "target": "diagnosis",
        "features": [
            f"{feature}_{statistic}"
            for statistic in ("mean", "se", "worst")
            for feature in (
                "radius",
                "texture",
                "perimeter",
                "area",
                "smoothness",
                "compactness",
                "concavity",
                "concave points",
                "symmetry",
                "fractal dimension",
            )
        ],
        "categorical": [],
        "positive_value": "M",
        "negative_value": "B",
        "positive_label": "Malignant class predicted",
        "negative_label": "Benign class predicted",
        "source": "UCI Wisconsin Diagnostic Breast Cancer (WDBC)",
        "source_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "limitations": (
            "The 30 inputs are nuclear measurements computed from digitized fine-needle "
            "aspirate images. Ordinary pathology reports generally do not contain all "
            "30 measurements."
        ),
    },
    "heart_disease": {
        "title": "Heart Disease",
        "dataset": "heart_disease.csv",
        "target": "num",
        "features": [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ],
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "positive_value": 1,
        "negative_value": 0,
        "positive_label": "Disease-present class predicted",
        "negative_label": "No-disease class predicted",
        "source": "UCI Heart Disease, Cleveland processed subset",
        "source_url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "limitations": (
            "This is a small historical Cleveland cohort. The original target values 1–4 "
            "are grouped as disease present; 0 means no disease. It is not a clinical tool."
        ),
    },
    "diabetes": {
        "title": "Diabetes",
        "dataset": "diabetes.csv",
        "target": "Outcome",
        "features": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        "categorical": [],
        "positive_value": 1,
        "negative_value": 0,
        "positive_label": "Diabetes class predicted",
        "negative_label": "No-diabetes class predicted",
        "source": "Pima Indians Diabetes Database",
        "source_url": "https://github.com/Zue77/Pima-Indians-Diabetes-Dataset/blob/main/diabetes.csv",
        "limitations": (
            "The Pima dataset represents a specific population of women aged 21 and older. "
            "It is not representative of all people and is not clinically validated."
        ),
    },
}

HEART_CATEGORIES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic",
    },
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {
        1: "Upsloping",
        2: "Flat",
        3: "Downsloping",
    },
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}

DIABETES_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

HEART_RANGES = {
    "age": (18, 110),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (50, 300),
    "chol": (0, 800),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (40, 250),
    "exang": (0, 1),
    "oldpeak": (-10, 20),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}

DIABETES_RANGES = {
    "Pregnancies": (0, 30),
    "Glucose": (1, 500),
    "BloodPressure": (1, 300),
    "SkinThickness": (1, 150),
    "Insulin": (1, 3000),
    "BMI": (1, 150),
    "DiabetesPedigreeFunction": (0, 10),
    "Age": (21, 120),
}
