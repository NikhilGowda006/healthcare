<<<<<<< HEAD
# HealthAI — Multi-Disease Prediction System

An educational Streamlit project with a responsive top navigation interface for separate binary classifiers trained on the Wisconsin Diagnostic Breast Cancer, Cleveland Heart Disease, and Pima Indians Diabetes datasets. Each disease has its own schema, preprocessing pipeline, model artifacts, evaluation reports, and dashboard.

> **Use:** For education only; not for clinical decisions.

## Features

- Independent dashboards for breast cancer, heart disease, and diabetes.
- Responsive black-and-graphite healthcare analytics interface with horizontal navigation, disease cards, guided manual-entry forms, and saved evaluation dashboards.
- Manual data entry with required-field and range validation.
- Three algorithms per disease: Logistic Regression, Random Forest, and SVM.
- Stratified train/validation/test splits. Hyperparameter tuning uses training folds; thresholds and model selection use validation data; test data is held out for final evaluation.
- Confusion matrices, ROC curves, comparison charts, JSON/CSV evaluation reports, and downloadable prediction summaries.
- No patient inputs are written to disk by the application.

## Dataset sources and scope

| Dataset | Expected rows | Input fields | Target | Notes |
|---|---:|---:|---|---|
| [Wisconsin Diagnostic Breast Cancer (WDBC)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 30 numerical measurements | `diagnosis` (`M` / `B`) | `id` is excluded from training. Measurements describe digitized FNA images, and ordinary pathology reports generally do not provide all 30. |
| [UCI Heart Disease, Cleveland subset](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | 13 fields | `num` (0 vs. 1–4) | `?` is treated as missing and imputed inside the training pipeline. The cohort is small and historical. |
| [Pima Indians Diabetes Database](https://github.com/Zue77/Pima-Indians-Diabetes-Dataset/blob/main/diabetes.csv) | 768 | 8 fields | `Outcome` (0 / 1) | Represents a specific population of women aged 21 and older; it is not representative of everyone. |

Place the files in `datasets\` as `breast_cancer.csv`, `heart_disease.csv`, and `diabetes.csv`. The supplied WDBC and Cleveland archives were preserved outside the project; the WDBC and Cleveland CSVs include documented headers. The supplied Pima text file was copied to `diabetes.csv` after checking its header and dimensions.

The WDBC columns are `id`, `diagnosis`, followed by the 30 feature fields: mean, standard error (`se`), and worst values for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. The heart schema is `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num`. The diabetes schema is `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`.

Training writes separate, unscaled feature/target CSVs to `datasets\processed\`. These retain missing values for imputation inside training-only pipelines; no statistics from validation or test rows are used to prepare model inputs.

For the Pima training data, zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are handled as missing values and imputed by a training-fitted median. Zero Pregnancies remains a valid value. Patient-entered values in these five fields must be positive; invalid values are not silently replaced.

## Install on Windows

Use Python 3.10 or later. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and run

Train all nine models and generate evaluation artifacts:

```powershell
python -m training.train_all
```

Train one disease only:

```powershell
python -m training.train_breast_cancer
python -m training.train_heart_disease
python -m training.train_diabetes
```

Launch the app:

```powershell
python -m streamlit run app.py
```

Run automated tests:

```powershell
python -m unittest discover -s tests -v
```

## Model development and evaluation

For each disease, dataset schema, target distribution, data types, missing values, numeric summaries, and duplicate rows are written to `reports\<disease>\dataset_analysis.json`. Heart target values 1–4 are grouped as disease present. Training uses a stratified 70/15/15 train/validation/test split. Preprocessors are fitted only inside training folds/pipelines. Logistic Regression and SVM use scaled numeric features; Random Forest does not. Missing numeric and categorical values are imputed by the training pipeline, with categorical columns one-hot encoded.

Hyperparameters are selected by ROC-AUC cross-validation on the training partition. Each model's probability threshold is selected on validation data by balanced accuracy. The recommended model is selected by average validation rank across ROC-AUC, precision, sensitivity, specificity, and F1—not test accuracy. The final metrics and charts are generated on the untouched test partition.

Artifacts are saved to:

- `models\<disease>\` — three Joblib pipelines and metadata, including feature order, class mapping, selected thresholds, validation rankings, and training configuration.
- `reports\<disease>\` — test metrics, classification reports, dataset analysis, confusion matrix PNGs, ROC curves, model comparison charts, and CSV/JSON summaries.

The held-out scores characterize only the selected dataset split. They are not evidence of clinical performance or generalization to new populations.

## Project structure

```text
app.py
config.py
datasets/
models/
training/
  common.py
  train_all.py
  train_breast_cancer.py
  train_heart_disease.py
  train_diabetes.py
preprocessing/
  datasets.py
  pipelines.py
dashboards/
  shared.py
  breast_cancer.py
  heart_disease.py
  diabetes.py
utils/
  prediction.py
  reporting.py
  validation.py
reports/
tests/
requirements.txt
```

## Limitations and disclaimer

The source datasets are small, historical, and population-specific. Saved test scores do not establish clinical performance or generalization to other populations.
=======
# healthcare
HealthAI is a Machine Learning project that predicts Breast Cancer, Heart Disease, and Diabetes using patient data. Built with Python and Streamlit, it provides an easy-to-use interface for disease prediction and model performance analysis.
>>>>>>> b63314d44f94564d290c71ff622a5727eefff1fd
