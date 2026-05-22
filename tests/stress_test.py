import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extended_sklearn_metrics import (
    evaluate_model_with_cross_validation,
    evaluate_classification_model_with_cross_validation,
    final_model_evaluation,
    CustomThresholds
)

def stress_test_regression():
    print("--- Stress Testing Regression ---")
    # Large dataset: 10,000 samples, 50 features
    X, y = make_regression(n_samples=10000, n_features=50, noise=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42) # Fast forest
    
    start_time = time.time()
    # Test with 10 folds
    result = evaluate_model_with_cross_validation(model, X, y, cv=10)
    end_time = time.time()
    
    print(f"Regression CV (10k samples, 50 features, 10 folds) took {end_time - start_time:.2f} seconds")
    print(result[['Metric', 'Value', 'Std Dev', 'Performance']])

def stress_test_classification():
    print("--- Stress Testing Classification ---")
    # Large dataset: 10,000 samples, 50 features, 3 classes
    X, y = make_classification(n_samples=10000, n_features=50, n_informative=20, n_classes=3, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    start_time = time.time()
    # Test with 10 folds
    result = evaluate_classification_model_with_cross_validation(model, X, y, cv=10)
    end_time = time.time()
    
    print(f"Classification CV (10k samples, 50 features, 10 folds) took {end_time - start_time:.2f} seconds")
    print(result[['Metric', 'Value', 'Std Dev', 'Performance']])

def stress_test_comprehensive():
    print("--- Stress Testing Comprehensive Evaluation ---")
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add protected attributes for fairness analysis
    protected_attributes = {
        'gender': np.random.choice(['male', 'female'], size=len(y_test)),
        'age_group': np.random.choice(['young', 'middle', 'senior'], size=len(y_test))
    }
    
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    
    start_time = time.time()
    results = final_model_evaluation(
        model, X_train, y_train, X_test, y_test,
        protected_attributes=protected_attributes,
        suppress_warnings=True
    )
    end_time = time.time()
    
    print(f"Comprehensive evaluation took {end_time - start_time:.2f} seconds")
    print(f"Task type: {results['task_type']}")
    print(f"Metrics calculated: {list(results.keys())}")
    
    # Check if TypedDict structure is respected (at least presence of keys)
    expected_keys = ['performance', 'cv_stability', 'feature_importance', 'error_analysis', 'fairness_analysis', 'interpretation']
    for key in expected_keys:
        if key in results:
            print(f"✓ Found {key}")
        else:
            print(f"✗ Missing {key}")

if __name__ == "__main__":
    stress_test_regression()
    stress_test_classification()
    stress_test_comprehensive()
