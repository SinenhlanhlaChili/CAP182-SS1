# Model 2 – Random Forest

**Script:** [`run_pipeline.py`](run_pipeline.py), `models['Random_Forest']`.

Random Forest provides a nonlinear ensemble comparator. Settings: `n_estimators=100`, `max_depth=14`, `min_samples_leaf=5`, `class_weight='balanced_subsample'`, `n_jobs=-1`, `random_state=42`. It uses the same features, patient-grouped split and training-only preprocessing as Model 1. No hyperparameter search is performed. The reported classification threshold is 0.5. See [performance](../Part_C/Model2Performance.MD).
