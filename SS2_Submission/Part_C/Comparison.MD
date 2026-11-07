# Model comparison

**Script:** [`run_pipeline.py`](../Part_B/run_pipeline.py). **Data:** [`performance_metrics.csv`](results/performance_metrics.csv). **Figures:** [ROC](results/roc_comparison.png) and [precision–recall](results/precision_recall_comparison.png).

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.653 | 0.740 |
| Precision | 0.170 | 0.195 |
| Recall | 0.518 | 0.402 |
| F1 | 0.256 | 0.263 |
| ROC-AUC | 0.641 | 0.640 |
| Average precision | 0.209 | 0.198 |

**Interpretation:** Random Forest has higher accuracy, precision and F1 at threshold 0.5, but Logistic Regression has higher recall, ROC-AUC and average precision. Because the STADIOcare problem emphasises identifying at-risk patients, Logistic Regression is a reasonable provisional choice for further validation; it also offers simpler interpretation. Neither model is deployment-ready: precision is low, threshold selection is unoptimised, and validation is limited to a single holdout. This comparison is descriptive; no inferential significance test has been performed.
