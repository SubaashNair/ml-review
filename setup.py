from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-review",
    version="0.4.0",
    author="Subashanan Nair",
    author_email="subashnair12@gmail.com",
    description="MLReview model evaluation, inspection, diagnostics, and reporting for scikit-learn estimators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SubaashNair/ml-review",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "extended_sklearn_metrics": ["*.py"],
        "ml_review": ["*.py"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "shap": [
            "shap>=0.44,<0.50; python_version < '3.11'",
            "shap>=0.50; python_version >= '3.11'",
        ],
    },
)
