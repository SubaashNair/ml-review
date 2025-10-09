# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-01-XX

### Major Refactoring - Improved Architecture and Performance

This release includes significant architectural improvements focused on code quality, maintainability, and performance while maintaining 100% backward compatibility.

### Added

- **Shared Validation Module** (`_validation.py`)
  - `validate_estimator()` - Check sklearn compatibility
  - `validate_input_arrays()` - Shape and size validation
  - `validate_no_nan_inf()` - Data quality checks
  - `validate_cv_parameter()` - Cross-validation validation
  - `validate_sklearn_inputs()` - Complete validation pipeline

- **Lazy Matplotlib Import System** (`_plotting_backend.py`)
  - `get_matplotlib()` - Cached lazy import
  - `check_matplotlib_available()` - Availability check
  - `require_matplotlib()` - Function decorator
  - `reset_matplotlib_cache()` - Testing utility

- **Modular Visualization Package**
  - Split monolithic `visualizations.py` into focused sub-modules:
    - `visualizations/_base.py` - Common utilities
    - `visualizations/performance.py` - Performance plots
    - `visualizations/roc_curves.py` - ROC/AUC plots
    - `visualizations/residuals.py` - Residual diagnostics
    - `visualizations/comprehensive.py` - Multi-panel dashboards
    - `visualizations/fairness.py` - Fairness comparisons

### Changed

- **Code Deduplication**
  - Eliminated 150+ lines of duplicate validation code
  - Refactored `model_evaluation.py` to use shared validation (reduced by 58 lines)
  - Refactored `classification_evaluation.py` to use shared validation (reduced by 56 lines)

- **Import Optimization**
  - Removed 80+ lines of repetitive matplotlib try/except blocks
  - Implemented lazy matplotlib imports for ~10ms faster package import time
  - All visualization functions now use `@require_matplotlib` decorator

- **Code Organization**
  - Split 1,487-line `visualizations.py` into 6 focused modules (100-437 lines each)
  - Improved module cohesion and separation of concerns
  - Better code readability and maintainability

- **Documentation**
  - Completely rewrote README.md with detailed examples and explanations
  - Removed all emoji characters for professional appearance
  - Added comprehensive API reference with examples
  - Expanded usage guide with production-ready workflow examples
  - Added architecture section documenting the new modular structure

### Performance

- Reduced total codebase by 1,750 lines (-21.5% from 8,128 to 6,378 lines)
- Faster package imports through lazy matplotlib loading
- More efficient validation (no redundant checks)

### Backward Compatibility

- 100% backward compatible with v0.3.x
- All existing code works without modification
- No breaking changes to public API
- All 79 tests passing

### Technical Details

**Files Created:**
- `extended_sklearn_metrics/_validation.py` (219 lines)
- `extended_sklearn_metrics/_plotting_backend.py` (111 lines)
- `extended_sklearn_metrics/visualizations/__init__.py` (72 lines)
- `extended_sklearn_metrics/visualizations/_base.py` (83 lines)
- `extended_sklearn_metrics/visualizations/performance.py` (308 lines)
- `extended_sklearn_metrics/visualizations/roc_curves.py` (437 lines)
- `extended_sklearn_metrics/visualizations/residuals.py` (380 lines)
- `extended_sklearn_metrics/visualizations/comprehensive.py` (359 lines)
- `extended_sklearn_metrics/visualizations/fairness.py` (221 lines)

**Files Modified:**
- `extended_sklearn_metrics/model_evaluation.py` (reduced from 216 to 158 lines)
- `extended_sklearn_metrics/classification_evaluation.py` (reduced from 177 to 121 lines)
- `extended_sklearn_metrics/visualizations.py` (refactored to use lazy imports)
- `README.md` (expanded from 540 to 1,400+ lines with detailed documentation)

## [0.3.5] - 2024-XX-XX

### Fixed

- Fixed AttributeError in feature interactions analysis when data has insufficient samples
- Enhanced correlation validation and error handling for feature interaction detection
- Improved robustness for edge cases in interaction analysis

## [0.3.4] - 2024-XX-XX

### Added

- Added `suppress_warnings` parameter to `final_model_evaluation()` function
- Users can now suppress sklearn warnings about feature names and other non-critical warnings
- Implemented clean context manager approach for warning suppression

### Example

```python
results = final_model_evaluation(
    model, X_train, y_train, X_test, y_test,
    suppress_warnings=True  # Suppress sklearn warnings
)
```

## [0.3.3] - 2024-XX-XX

### Fixed

- Fixed AttributeError in error correlation analysis when X_test has insufficient samples
- Enhanced validation and error handling for correlation calculations
- Improved robustness for edge cases with small datasets

## [0.3.2] - 2024-XX-XX

### Fixed

- Fixed AttributeError in model complexity analysis for tree-based models
- Enhanced error handling in comprehensive evaluation framework

## [0.3.1] - 2024-XX-XX

### Changed

- Improved error handling and stability
- Enhanced compatibility with different sklearn model types

## [0.3.0] - 2024-XX-XX

### Major Release - Comprehensive Evaluation Framework

### Added

- **ROC/AUC Analysis**
  - Comprehensive ROC curve analysis with threshold optimization
  - Multi-class ROC support using one-vs-rest approach
  - Precision-Recall curves and AUC-PR metrics
  - `calculate_roc_metrics()` - Binary ROC analysis
  - `calculate_multiclass_roc_metrics()` - Multi-class ROC
  - `calculate_precision_recall_metrics()` - PR curve analysis
  - `find_optimal_thresholds()` - Threshold optimization
  - `create_threshold_analysis_report()` - Detailed threshold report
  - `print_roc_auc_summary()` - Console summary

- **Comprehensive Evaluation**
  - `final_model_evaluation()` - Complete model assessment
  - Hold-out test evaluation with cross-validation stability
  - Feature importance analysis (built-in + permutation)
  - Model interpretation and complexity assessment
  - Error analysis and residual diagnostics
  - Fairness evaluation across demographic groups

- **Reporting Functions**
  - `create_evaluation_report()` - Detailed metrics DataFrame
  - `print_evaluation_summary()` - Executive summary
  - `create_feature_importance_report()` - Feature rankings
  - `create_fairness_report()` - Fairness analysis by group

- **Visualizations**
  - `create_roc_curve_plot()` - ROC curve visualization
  - `create_precision_recall_plot()` - PR curve visualization
  - `create_multiclass_roc_plot()` - Multi-class ROC
  - `create_threshold_analysis_plot()` - Threshold analysis
  - `create_comprehensive_evaluation_plots()` - Evaluation dashboard
  - `create_feature_importance_plot()` - Feature importance chart
  - `create_fairness_comparison_plot()` - Fairness comparison

### Changed

- Complete API overhaul for better usability
- Comprehensive documentation with examples
- Production-ready evaluation capabilities

## [0.2.0] - 2024-XX-XX

### Added

- Residual diagnostics for regression models
- `calculate_residual_diagnostics()` - Statistical tests
- `create_residual_summary_report()` - Diagnostic report
- `print_residual_diagnostics_report()` - Console output
- Enhanced visualization capabilities

### Changed

- Improved cross-validation metrics
- Better error handling
- More informative console output

## [0.1.0] - 2024-XX-XX

### Initial Release

### Added

- Basic classification and regression evaluation
- Cross-validation support with custom thresholds
- `evaluate_model_with_cross_validation()` - General evaluation
- `evaluate_classification_model_with_cross_validation()` - Classification
- `CustomThresholds` class for defining performance thresholds
- Basic performance visualizations
- Core metrics: accuracy, precision, recall, F1, R², RMSE, MAE

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

## Links

- [GitHub Repository](https://github.com/SubaashNair/extended-sklearn-metrics)
- [PyPI Package](https://pypi.org/project/extended-sklearn-metrics/)
- [Issue Tracker](https://github.com/SubaashNair/extended-sklearn-metrics/issues)
