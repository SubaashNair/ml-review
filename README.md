# MLReview

MLReview provides model evaluation, inspection, diagnostics, and reporting for
scikit-learn estimators. It replaces the `extended-sklearn-metrics`
distribution with a smaller canonical API and a review-oriented result model.

## Install

```bash
pip install ml-review
```

SHAP inspection is optional:

```bash
pip install ml-review[shap]
```

MLReview uses three spellings deliberately:

| Surface | Name |
| --- | --- |
| Product name | `MLReview` |
| PyPI distribution | `ml-review` |
| Python import package | `ml_review` |

## Quick Start

```python
from ml_review import evaluate
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
review = evaluate(model, X_train, y_train, X_test, y_test)

summary = review.report()
importance = review.importance.report()
```

`evaluate(...)` returns `ReviewResult`. User-facing reports are DataFrames,
while rich result state remains available through mapping reads and raw export:

```python
review["performance"]
review.get("feature_importance")
raw = review.to_dict()
```

## Canonical API

Keep the top-level import small:

```python
from ml_review import evaluate
from ml_review.evaluation import Thresholds, classification_cv, regression_cv
```

Use namespace imports for grouped workflows:

| Area | Import |
| --- | --- |
| Evaluation reports | `from ml_review.reporting import evaluation, fairness` |
| Importance and SHAP | `from ml_review.inspection import importance, shap` |
| ROC and PR review | `from ml_review.metrics import roc` |
| Residual diagnostics | `from ml_review.diagnostics import residual` |
| Performance plots | `from ml_review.plotting import performance` |
| ROC plots | `from ml_review.plotting import roc_plot` |

## Result Workflows

```python
review.report()
review.print_summary()

review.importance.report()
review.importance.plot()

review.fairness.report()
review.fairness.plot()
```

Methods are thin wrappers around shared functions, so the functional form also
works:

```python
from ml_review.inspection import importance
from ml_review.reporting import evaluation

evaluation.report(review)
importance.report(review)
```

Cross-validation summary APIs still return DataFrames directly:

```python
from ml_review.evaluation import classification_cv, regression_cv

classification_table = classification_cv(classifier, X, y, cv=5)
regression_table = regression_cv(regressor, X, y, cv=5)
```

## Optional SHAP Inspection

SHAP is preferred by `evaluate(...)` when it is installed. If it is missing in
the default `auto` mode, MLReview keeps built-in and permutation importance
behavior. Use `shap_mode="on"` to require SHAP or `shap_mode="off"` to skip it.

```python
review = evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    shap_mode="auto",
    shap_background_size=100,
    shap_sample_size=200,
)
```

```python
from ml_review.inspection import shap

shap_table = review.shap.report()
review.shap.plot_importance()
review.shap.plot_explanation(sample_index=0)

# Equivalent functional calls
shap.report(review)
shap.plot_importance(review)
```

SHAP guidance in the report:
- the baseline is the model output before feature contributions are added
- positive SHAP values raise the explained output and negative values lower it
- global SHAP importance averages absolute contributions across sampled rows
- local SHAP plots explain one stored sampled prediction
- classifier local plots default to the predicted class unless an output index
  is provided

MLReview stores bounded serialized SHAP payloads in review data rather than
native SHAP explanation objects.

## ROC and Residual Review

ROC functions receive an estimator and data, matching the existing
scikit-learn-oriented implementation:

```python
from ml_review.metrics import roc

roc_result = roc.binary(classifier, X, y, cv=5)
roc_table = roc_result.report()
thresholds = roc_result.thresholds()

pr_result = roc.precision_recall(classifier, X, y, cv=5)
pr_table = pr_result.report()
```

Residual diagnostics also use a result object with DataFrame reports:

```python
from ml_review.diagnostics import residual

residual_result = residual.calculate(regressor, X, y, cv=5)
residual_table = residual_result.report()
residual_result.plot()
```

## Migration

`extended_sklearn_metrics` remains importable from the new distribution as a
warning-bearing compatibility path. New releases are published as `ml-review`.

| Legacy API | Canonical MLReview API |
| --- | --- |
| `final_model_evaluation(...)` | `evaluate(...)` |
| `evaluate_model_with_cross_validation(...)` | `regression_cv(...)` |
| `evaluate_classification_model_with_cross_validation(...)` | `classification_cv(...)` |
| `CustomThresholds(...)` | `Thresholds(...)` |
| `create_evaluation_report(results)` | `review.report()` |
| `create_feature_importance_report(results)` | `review.importance.report()` |
| `create_feature_importance_plot(results)` | `review.importance.plot()` |
| `create_fairness_report(results)` | `review.fairness.report()` |
| `calculate_roc_metrics(...)` | `roc.binary(...)` |
| `calculate_residual_diagnostics(...)` | `residual.calculate(...)` |

See [MIGRATION.md](MIGRATION.md) for import examples and compatibility notes.

## Roadmap

The MLReview foundation release covers the package rename, namespace cleanup,
result objects, DataFrame-first reports, compatibility shims, and optional SHAP
inspection. Planned later feature releases cover:

1. calibration and probability reliability review
2. unsupervised review for clustering, PCA, and embeddings
3. deeper inspection, fairness checks, and report artifacts

See [ROADMAP.md](ROADMAP.md) for the feature pipeline and guardrails.

## License

MIT
