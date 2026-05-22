import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from typing import Union, Optional
import warnings

from .model_evaluation import SklearnEstimator, CustomThresholds
from ._validation import validate_sklearn_inputs


def evaluate_classification_model_with_cross_validation(
    model: SklearnEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    average: str = "weighted",
    custom_thresholds: Optional[CustomThresholds] = None,
) -> pd.DataFrame:
    """
    Evaluates a classification model using cross-validation and generates a performance summary table.

    Parameters
    ----------
    model : estimator object
        The machine learning model to evaluate (must implement fit and predict).
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values (class labels).
    cv : int, default=5
        Number of cross-validation folds.
    average : str, default='weighted'
        Averaging strategy for multiclass/multilabel targets:
        'micro', 'macro', 'weighted', 'samples', or None.
    custom_thresholds : CustomThresholds, optional
        Custom thresholds for performance evaluation. If None, uses default thresholds.

    Returns
    -------
    pd.DataFrame
        A summary table containing performance metrics, their interpretations, and variability.
    """
    # Input validation using shared validation utilities
    X_array, y_array = validate_sklearn_inputs(model, X, y, cv, check_y_numeric=False)

    # Check if model appears to be for classification
    if hasattr(model, "_estimator_type"):
        if model._estimator_type != "classifier":
            warnings.warn(
                f"Model appears to be a '{model._estimator_type}', but this function is designed for classification models.",
                UserWarning,
            )

    # Determine if it's binary or multiclass
    unique_labels = np.unique(y_array)
    n_classes = len(unique_labels)
    is_binary = n_classes == 2

    # Define scoring metrics
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average=average, zero_division=0),
        "recall": make_scorer(recall_score, average=average, zero_division=0),
        "f1": make_scorer(f1_score, average=average, zero_division=0),
    }

    # Add ROC AUC for binary classification
    if is_binary:
        scoring["roc_auc"] = "roc_auc"

    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    # Use custom thresholds if provided, otherwise use defaults
    if custom_thresholds is None:
        custom_thresholds = CustomThresholds()

    # Define performance categories
    def get_classification_performance(score: float) -> str:
        """Get performance category for classification metrics (0-1 scale)"""
        t_exc, t_good, t_acc, t_poor = custom_thresholds.classification_thresholds
        if score >= t_exc:
            return "Excellent"
        elif score >= t_good:
            return "Good"
        elif score >= t_acc:
            return "Acceptable"
        elif score >= t_poor:
            return "Poor"
        else:
            return "Very Poor"

    # Prepare results data
    metrics_to_report = [
        ("Accuracy", "accuracy", "correctly classified samples / total samples"),
        ("Precision", "precision", f"true positives / predicted positives, {average} average"),
        ("Recall", "recall", f"true positives / actual positives, {average} average"),
        ("F1-Score", "f1", f"harmonic mean of precision and recall, {average} average"),
    ]
    
    if is_binary:
        metrics_to_report.append(("ROC AUC", "roc_auc", "area under ROC curve"))

    results_list = []
    t_exc, t_good, t_acc, t_poor = custom_thresholds.classification_thresholds
    thresh_desc = f"≥{t_exc} = Excellent, {t_good}-{t_exc} = Good, {t_acc}-{t_good} = Acceptable, {t_poor}-{t_acc} = Poor, <{t_poor} = Very Poor"

    for display_name, internal_key, calc_info in metrics_to_report:
        scores = cv_results[f"test_{internal_key}"]
        mean = np.mean(scores)
        std = np.std(scores)
        
        results_list.append({
            "Metric": display_name,
            "Value": mean,
            "Std Dev": std,
            "Threshold": thresh_desc,
            "Calculation": f"{mean:.4f} ({calc_info})",
            "Performance": get_classification_performance(mean)
        })

    return pd.DataFrame(results_list)
