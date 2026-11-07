# Preprocessing

**Script:** [`run_pipeline.py`](run_pipeline.py) (`prepare`, `make_preprocessor`, `main`).

1. Read `Datasets/diabetic_data.csv`; treat `?` as missing.
2. Convert `readmitted` to binary: `<30` = 1; other values = 0.
3. Exclude `encounter_id` and `patient_nbr` as predictors; retain patient ID only for grouped splitting.
4. Exclude `weight`, `payer_code`, `medical_specialty`, `max_glu_serum` and `A1Cresult` due to high missingness; exclude `discharge_disposition_id` to avoid post-discharge status proxies and possible outcome-related information.
5. Use `GroupShuffleSplit` with `patient_nbr`, 80%/20%, `random_state=42`, preventing patient overlap.
6. For numeric columns: median imputation and standardisation. For categorical columns: most-frequent imputation and one-hot encoding with unknown categories ignored. All transformations are fitted on training data only via a scikit-learn Pipeline.
7. Retain the natural class imbalance; use model-specific balanced class weights. No resampling is performed.

**Limitations:** Excluding laboratory variables sacrifices potentially useful information; missingness may not be random. Group splitting does not guarantee a temporal or hospital-level external validation. No claim is made that all features are available at a single clinical prediction time.
