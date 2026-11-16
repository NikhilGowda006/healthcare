from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import DISEASES, MODELS_DIR, REPORTS_DIR
from preprocessing.datasets import (
    dataset_analysis,
    load_dataset,
    save_dataset_analysis,
    save_processed_dataset,
)
from preprocessing.pipelines import build_preprocessor

RANDOM_STATE = 42
MODEL_NAMES = ("logistic_regression", "random_forest", "svm")


def _model_and_grid(disease: str, model_name: str) -> tuple[Any, dict[str, list[Any]], bool]:
    weighted = disease in {"heart_disease", "diabetes"}
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=3000, solver="liblinear", random_state=RANDOM_STATE)
        grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"] if weighted else [None],
        }
        scale_numeric = True
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        grid = {
            "classifier__max_depth": [None, 8],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__class_weight": [None, "balanced"] if weighted else [None],
        }
        scale_numeric = False
    elif model_name == "svm":
        classifier = SVC(probability=True, random_state=RANDOM_STATE)
        grid = {
            "classifier__C": [0.5, 1.0, 5.0],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__class_weight": [None, "balanced"] if weighted else [None],
        }
        scale_numeric = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(disease, scale_numeric)),
            ("classifier", classifier),
        ]
    )
    return pipeline, grid, scale_numeric


def _threshold_for_validation(y_true: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = [
        balanced_accuracy_score(y_true, (probabilities >= threshold).astype(int))
        for threshold in candidates
    ]
    best = max(scores)
    tied = [threshold for threshold, score in zip(candidates, scores) if score == best]
    return float(min(tied, key=lambda threshold: abs(float(threshold) - 0.5)))


def _metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }
    return metrics, matrix, predictions


def _plot_confusion(
    matrix: np.ndarray,
    output: Path,
    title: str,
    labels: list[str],
) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axis,
    )
    axis.set(title=title, xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_roc(
    y_true: pd.Series,
    model_probabilities: dict[str, np.ndarray],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    for model_name, probabilities in model_probabilities.items():
        false_positive, true_positive, _ = roc_curve(y_true, probabilities)
        axis.plot(false_positive, true_positive, label=model_name.replace("_", " "))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(
        title="Test-set ROC curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_comparison(results: dict[str, dict[str, Any]], output: Path) -> None:
    rows = [
        {"model": model_name, **details["test_metrics"]}
        for model_name, details in results.items()
    ]
    metrics_frame = pd.DataFrame(rows).set_index("model")[
        ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc"]
    ]
    axis = metrics_frame.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0)
    axis.set(title="Test-set model comparison", ylabel="Score")
    axis.legend(loc="lower left", bbox_to_anchor=(1, 0))
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=150)
    plt.close(axis.figure)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def train_disease(
    disease: str,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    if disease not in DISEASES:
        raise ValueError(f"Unknown disease: {disease}")
    model_root = models_dir or MODELS_DIR
    report_root = reports_dir or REPORTS_DIR
    disease_model_dir = model_root / disease
    disease_report_dir = report_root / disease
    disease_model_dir.mkdir(parents=True, exist_ok=True)
    disease_report_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset(disease)
    save_processed_dataset(disease, dataframe)
    save_dataset_analysis(
        disease, dataset_analysis(disease, dataframe), report_root
    )
    specification = DISEASES[disease]
    target_column = specification["target"]
    features = specification["features"]
    X = dataframe[features].copy()
    target = dataframe[target_column]
    if disease == "breast_cancer":
        y = (target == specification["positive_value"]).astype(int)
    elif disease == "heart_disease":
        y = (target > 0).astype(int)
    else:
        y = target.astype(int)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )

    results: dict[str, dict[str, Any]] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    test_reports: dict[str, Any] = {}

    for model_name in MODEL_NAMES:
        pipeline, parameter_grid, scale_numeric = _model_and_grid(disease, model_name)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        validation_probabilities = best_pipeline.predict_proba(X_validation)[:, 1]
        threshold = _threshold_for_validation(y_validation, validation_probabilities)
        validation_metrics, _, _ = _metrics(
            y_validation, validation_probabilities, threshold
        )
        test_scores = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics, test_matrix, test_predictions = _metrics(
            y_test, test_scores, threshold
        )
        test_probabilities[model_name] = test_scores
        class_labels = [
            specification["negative_label"],
            specification["positive_label"],
        ]
        report = classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        joblib.dump(best_pipeline, disease_model_dir / f"{model_name}.joblib")
        _plot_confusion(
            test_matrix,
            disease_report_dir / f"{model_name}_confusion_matrix.png",
            f"{specification['title']} - {model_name.replace('_', ' ')}",
            ["No disease", "Disease"],
        )
        results[model_name] = {
            "threshold": threshold,
            "best_parameters": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "confusion_matrix": test_matrix.tolist(),
            "classification_report": report,
            "training_preprocessing": {
                "scaled_numeric_features": scale_numeric,
                "categorical_features": specification["categorical"],
            },
        }
        test_reports[model_name] = report

    validation_frame = pd.DataFrame(
        {
            model_name: details["validation_metrics"]
            for model_name, details in results.items()
        }
    ).T
    selection_metrics = [
        "roc_auc",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
    ]
    ranks = validation_frame[selection_metrics].rank(
        ascending=False, method="average"
    )
    average_rank = ranks.mean(axis=1)
    best_model = str(average_rank.idxmin())
    for model_name, details in results.items():
        details["validation_average_rank"] = float(average_rank[model_name])

    _plot_roc(
        y_test,
        test_probabilities,
        disease_report_dir / "roc_curves.png",
    )
    _plot_comparison(results, disease_report_dir / "model_comparison.png")

    metadata = {
        "disease": disease,
        "source": specification["source"],
        "source_url": specification["source_url"],
        "feature_names": features,
        "target": target_column,
        "positive_class": specification["positive_value"],
        "negative_class": specification["negative_value"],
        "class_mapping": {
            "0": specification["negative_label"],
            "1": specification["positive_label"],
        },
        "split_sizes": {
            "training": int(len(X_train)),
            "validation": int(len(X_validation)),
            "testing": int(len(X_test)),
        },
        "best_model_validation_rank": best_model,
        "selection_metrics": selection_metrics,
        "models": results,
    }
    with (disease_model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(_jsonable(metadata), file, indent=2)
    with (disease_report_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(_jsonable(metadata), file, indent=2)

    table = pd.DataFrame(
        {
            model_name: details["test_metrics"]
            for model_name, details in results.items()
        }
    ).T
    table.to_csv(disease_report_dir / "evaluation.csv", index_label="model")

    try:
        dataset_source = DISEASES[disease]["dataset"]
        report_metadata = {
            "disease": disease,
            "dataset": dataset_source,
            "training_configuration": {
                "random_state": RANDOM_STATE,
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "cross_validation_folds": 3,
                "test_set_used_for_tuning_or_selection": False,
            },
        }
        with (disease_report_dir / "training_configuration.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(report_metadata, file, indent=2)
    except OSError:
        raise

    return {
        "disease": disease,
        "best_model_validation_rank": best_model,
        "results": results,
        "report_dir": str(disease_report_dir),
    }
