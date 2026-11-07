# Feature engineering

**Script:** [`run_pipeline.py`](run_pipeline.py), function `prepare` and `diagnosis_group`.

- Transform each ICD-9 diagnosis (`diag_1`, `diag_2`, `diag_3`) into broad disease categories: diabetes, circulatory, respiratory, digestive, injury, musculoskeletal, genitourinary, neoplasm, external cause, supplementary and other/missing. These are approximate groupings rather than a full validated clinical code mapping.
- Add `prior_utilisation_total` = `number_inpatient` + `number_outpatient` + `number_emergency` (missing components treated as zero for this derived feature only).
- Treat `age` as a categorical age band rather than a precise continuous age.
- Treat admission-type and admission-source IDs as categorical codes; see `Datasets/IDS_mapping.csv` for descriptions.
- Keep existing utilisation, medication, procedure and encounter-related features; categorical features are one-hot encoded.

**Caution:** Do not interpret feature association as causality. Prior utilisation is a historical proxy, not a verified longitudinal STADIOcare record. The codebook should be consulted before any deployment mapping.
