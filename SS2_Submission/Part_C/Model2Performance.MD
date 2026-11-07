# Random Forest performance

**Evaluation script:** [`run_pipeline.py`](../Part_B/run_pipeline.py), function `main`. **Results:** [`performance_metrics.csv`](results/performance_metrics.csv), [`test predictions`](results/Random_Forest_test_predictions.csv), [`confusion matrix`](results/Random_Forest_confusion_matrix.png).

| Metric | Test result |
|---|---:|
| Test encounters | 20,488 |
| Positive prevalence | 11.55% |
| Accuracy | 0.740 |
| Precision (30-day readmission) | 0.195 |
| Recall (30-day readmission) | 0.402 |
| F1 | 0.263 |
| ROC-AUC | 0.640 |
| Average precision (PR-AUC summary) | 0.198 |
| True negatives | 14,207 |
| False positives | 3,914 |
| False negatives | 1,416 |
| True positives | 951 |

All figures reflect a **single patient-grouped holdout split** and a fixed 0.5 decision threshold. No uncertainty intervals, temporal validation, probability calibration or clinical utility analysis are included. High accuracy can be misleading because only about 11.6% of the test encounters have the positive outcome.
