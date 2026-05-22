# MLReview Migration Guide

`ml-review` supersedes the `extended-sklearn-metrics` distribution.

## Name Mapping

| Surface | Old | New |
| --- | --- | --- |
| Install | `pip install extended-sklearn-metrics` | `pip install ml-review` |
| Python import root | `extended_sklearn_metrics` | `ml_review` |
| Product name | Extended Sklearn Metrics | MLReview |

The old import package is included as a compatibility layer and emits a
`FutureWarning` to make migration visible.

## Main Workflow

Legacy:

```python
from extended_sklearn_metrics import (
    create_feature_importance_report,
    final_model_evaluation,
)

results = final_model_evaluation(model, X_train, y_train, X_test, y_test)
feature_table = create_feature_importance_report(results)
```

Canonical MLReview:

```python
from ml_review import evaluate

review = evaluate(model, X_train, y_train, X_test, y_test)
feature_table = review.importance.report()
```

## API Mapping

| Legacy API | MLReview API |
| --- | --- |
| `final_model_evaluation(...)` | `evaluate(...)` |
| `evaluate_model_with_cross_validation(...)` | `regression_cv(...)` |
| `evaluate_classification_model_with_cross_validation(...)` | `classification_cv(...)` |
| `CustomThresholds(...)` | `Thresholds(...)` |
| `create_evaluation_report(results)` | `review.report()` |
| `print_evaluation_summary(results)` | `review.print_summary()` |
| `create_feature_importance_report(results)` | `review.importance.report()` |
| `create_feature_importance_plot(results)` | `review.importance.plot()` |
| `create_fairness_report(results)` | `review.fairness.report()` |
| `create_fairness_comparison_plot(results)` | `review.fairness.plot()` |
| `calculate_roc_metrics(...)` | `roc.binary(...)` |
| `calculate_multiclass_roc_metrics(...)` | `roc.multiclass(...)` |
| `calculate_precision_recall_metrics(...)` | `roc.precision_recall(...)` |
| `find_optimal_thresholds(...)` | `roc_result.thresholds()` |
| `create_threshold_analysis_report(...)` | `roc_result.report()` |
| `calculate_residual_diagnostics(...)` | `residual.calculate(...)` |
| `create_residual_summary_report(...)` | `residual_result.report()` |
| `create_residual_plots(...)` | `residual_result.plot()` |

## Output Mapping

`evaluate(...)` now returns `ReviewResult` instead of exposing a plain
dictionary as the canonical API. It remains mapping-compatible:

```python
review["performance"]
review.to_dict()
```

Reports remain DataFrame-first:

```python
review.report()
review.importance.report()
roc_result.report()
residual_result.report()
```

## Optional SHAP

Install SHAP support separately:

```bash
pip install ml-review[shap]
```

```python
review = evaluate(..., shap_mode="auto")
review.shap.report()
review.shap.plot_importance()
```

Use `shap_mode="on"` when missing SHAP must be treated as an error. Use
`shap_mode="off"` when the evaluation should skip SHAP entirely.
