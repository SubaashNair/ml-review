# MLReview Feature Roadmap

## Release 1: Foundation

Goal: establish MLReview as the package identity and public API foundation.

Deliverables:
- `ml-review` distribution and `ml_review` import package
- compatibility shims for `extended_sklearn_metrics`
- short namespace API and result objects
- DataFrame-first report outputs with raw `.to_dict()` export
- optional SHAP feature importance and explanation UX

Exit criteria:
- canonical imports and package builds pass
- legacy imports remain available with migration warnings
- SHAP optional dependency behavior is documented and tested

## Release 2: Calibration and Reliability

Goal: review probability quality for classification workflows.

Planned work:
- calibration report DataFrames
- reliability plots
- Brier-style summary metrics where applicable
- confidence and miscalibration guidance

Boundary: do not add calibration model training or deployment monitoring unless
they are approved as separate scope.

## Release 3: Unsupervised Review Foundation

Goal: add `X`-first unsupervised review APIs without overloading supervised
`evaluate(...)`.

Planned namespaces:

```python
from ml_review.unsupervised import cluster, projection
```

Planned first result objects:

| Area | API | Result |
| --- | --- | --- |
| Clustering | `cluster.review(model, X)` | `ClusterResult` |
| PCA | `projection.pca(model, X)` | `ProjectionResult` |
| Embeddings/t-SNE | `projection.embedding(model, X)` | `EmbeddingResult` |

Planned reports and plots:
- clustering quality and size summaries
- clustering metrics with and without reference labels
- PCA explained variance and loading reports
- PCA variance and projection plots
- embedding coordinate and embedding scatter views

## Release 4: Inspection Depth

Candidate work:
- PDP and ICE inspection helpers
- ALE exploration if justified
- comparison of importance methods
- correlated-feature and importance-stability guidance
- richer error slices

## Release 5: Fairness and Evaluation Checks

Candidate work:
- intersectional fairness reports
- user-selected fairness metrics
- subgroup sample-size warnings
- baseline comparisons
- thresholded pass/fail review checks

## Release 6: Report Artifacts

Candidate work:
- HTML and Markdown exports
- composed review artifacts from evaluation, SHAP, calibration, fairness, and
  unsupervised sections

## Guardrails

Keep MLReview focused on evaluation, diagnostics, inspection, explanation, and
readable review artifacts. Avoid turning it into a training framework,
hyperparameter optimization suite, monitoring platform, fairness mitigation
library, or experiment tracker.
