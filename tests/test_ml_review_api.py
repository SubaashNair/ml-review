import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from ml_review import ReviewResult, evaluate
from ml_review.diagnostics import residual
from ml_review.metrics import roc


def test_review_result_reports_and_dict_access():
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    review = evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_folds=2,
        shap_mode="off",
    )

    assert isinstance(review, ReviewResult)
    assert review["task_type"] == "classification"
    assert isinstance(review.report(), pd.DataFrame)
    assert isinstance(review.importance.report(), pd.DataFrame)
    assert isinstance(review.to_dict(), dict)


def test_roc_and_residual_result_objects_return_dataframes():
    X_class, y_class = make_classification(n_samples=80, n_features=4, random_state=0)
    classifier = LogisticRegression(max_iter=1000)
    roc_result = roc.binary(classifier, X_class, y_class, cv=2)

    assert isinstance(roc_result.report(), pd.DataFrame)
    assert isinstance(roc_result.thresholds(), pd.DataFrame)
    assert isinstance(roc_result.to_dict(), dict)

    X_reg, y_reg = make_regression(n_samples=80, n_features=4, random_state=0)
    residual_result = residual.calculate(LinearRegression(), X_reg, y_reg, cv=2)

    assert isinstance(residual_result.report(), pd.DataFrame)
    assert "residuals" in residual_result


def test_legacy_import_warns():
    import extended_sklearn_metrics

    with pytest.warns(FutureWarning, match="compatibility package"):
        importlib.reload(extended_sklearn_metrics)


def test_fake_shap_backend_populates_review_payload(monkeypatch):
    class FakeExplanation:
        def __init__(self, X):
            self.values = np.asarray(X) * 0.1
            self.base_values = np.zeros(len(X))

    class FakeExplainer:
        def __init__(self, prediction_fn, background):
            self.prediction_fn = prediction_fn
            self.background = background

        def __call__(self, X):
            return FakeExplanation(X)

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=FakeExplainer))

    X, y = make_classification(n_samples=80, n_features=4, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
    model = RandomForestClassifier(n_estimators=5, random_state=7).fit(X_train, y_train)
    review = evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_folds=2,
        shap_mode="on",
        shap_background_size=8,
        shap_sample_size=10,
    )

    payload = review["feature_importance"]
    assert "shap_importance" in payload
    assert payload["shap_importance"]["sample_count"] == 10
    assert isinstance(review.shap.report(), pd.DataFrame)
    assert "SHAP_Importance" in review.importance.report().columns


def test_missing_shap_skips_auto_and_errors_when_required(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)
    X, y = make_classification(n_samples=60, n_features=4, random_state=9)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    review = evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_folds=2,
        shap_mode="auto",
    )

    assert "shap_importance" not in review["feature_importance"]
    with pytest.raises(ImportError, match="ml-review\\[shap\\]"):
        evaluate(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            cv_folds=2,
            shap_mode="on",
        )
