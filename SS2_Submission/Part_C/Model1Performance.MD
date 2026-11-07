# Logistic Regression performance

**Evaluation script:** [`run_pipeline.py`](../Part_B/run_pipeline.py), function `main`. **Results:** [`performance_metrics.csv`](results/performance_metrics.csv), [`test predictions`](results/Logistic_Regression_test_predictions.csv), [`confusion matrix`](results/Logistic_Regression_confusion_matrix.png).

| Metric | Test result |
|---|---:|
| Test encounters | 20,488 |
| Positive prevalence | 11.55% |
| Accuracy | 0.653 |
| Precision (30-day readmission) | 0.170 |
| Recall (30-day readmission) | 0.518 |
| F1 | 0.256 |
| ROC-AUC | 0.641 |
| Average precision (PR-AUC summary) | 0.209 |
| True negatives | 12,148 |
| False positives | 5,973 |
| False negatives | 1,141 |
| True positives | 1,226 |

All figures reflect a **single patient-grouped holdout split** and a fixed 0.5 decision threshold. No uncertainty intervals, temporal validation, probability calibration or clinical utility analysis are included. High accuracy can be misleading because only about 11.6% of the test encounters have the positive outcome.
