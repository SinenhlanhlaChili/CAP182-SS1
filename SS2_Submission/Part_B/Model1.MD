# Model 1 – Logistic Regression

**Script:** [`run_pipeline.py`](run_pipeline.py), `models['Logistic_Regression']`.

Logistic Regression is the interpretable linear baseline for binary 30-day readmission prediction. The pipeline uses training-only imputation, scaling and one-hot encoding, then `LogisticRegression(max_iter=1500, class_weight='balanced', solver='liblinear', random_state=42)`. Balanced class weights counter the low positive-class prevalence. No hyperparameter search is performed. Scores are probabilities; the reported confusion matrix uses threshold 0.5. See [performance](../Part_C/Model1Performance.MD).
