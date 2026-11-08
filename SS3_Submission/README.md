# CAP182 Capstone – SS3
## Predicting 30-Day Hospital Readmission Using Machine Learning

### 1. Project Overview

This project forms part of the CAP182 Capstone assessment and focuses on developing a machine learning classification model to predict the likelihood of hospital readmission within 30 days following discharge.

The project is conducted within the STADIOcare project context. Its objective is to investigate how predictive analytics can support the identification of patients who may require additional attention during discharge planning and post-discharge care.

The supplied dataset is synthetic and does not represent actual STADIOcare patient records.

### 2. Project Objectives

- Prepare and integrate the supplied healthcare datasets.
- Identify and address data-quality issues.
- Develop a Logistic Regression classification model.
- Evaluate model performance using appropriate classification metrics.
- Assess the potential application of predictive analytics in healthcare decision support.

### 3. Repository Structure

```text
SS3_Submission/
│
├── data/
│   ├── README.md
│   ├── table1_admissions_adt.csv
│   ├── table2_patient_clinical.csv
│   ├── table3_previous_utilisation.csv
│   └── table4_discharge_readmission_outcome.csv
│
├── SS3_PartB.ipynb
├── SS3_PartC_Presentation.pptx
├── Part_A_Peer_Review_Vusumuzi_Nkosi_2026.docx
├── requirements.txt
└── README.md
```

### 4. Dataset Description

Four datasets were supplied for the analysis:

- Admission, discharge and transfer information.
- Patient clinical and demographic information.
- Previous healthcare utilisation.
- Discharge and readmission outcome information.

After duplicate removal, filtering and integration, the final analytical dataset contained 6,000 index admissions.

The target variable identifies whether a patient experienced a readmission within 30 days.

### 5. Methodology

The project followed these main analytical stages:

1. Data loading and profiling.
2. Data cleaning and preprocessing.
3. Removal of potential data leakage variables.
4. Integration of the four datasets.
5. Grouped training, validation and test splitting.
6. Logistic Regression model development.
7. Probability threshold selection using the validation dataset.
8. Final evaluation using the independent test dataset.

The dataset was divided into 70% training, 15% validation and 15% testing.

### 6. Model Performance

The Logistic Regression model was evaluated using the independent test dataset.

The selected probability threshold was 0.33.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 49.22% |
| Precision | 25.00% |
| Recall | 74.86% |
| F1-score | 37.48% |
| F2-score | 53.52% |
| ROC-AUC | 66.17% |
| Average Precision | 37.34% |

The model achieved a recall of approximately 74.86%, indicating that it identified a substantial proportion of actual readmission cases in the test dataset.

However, the precision of 25% indicates a relatively high number of false-positive predictions. Further model improvement and validation are therefore necessary.

### 7. Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- GitHub

### 8. Running the Notebook

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run `SS3_PartB.ipynb`.

Ensure that the four CSV files are available in the `data/` directory.

### 9. Limitations and Data Governance

The supplied dataset is synthetic and should not be interpreted as actual hospital patient data.

The model has not undergone external clinical validation or prospective testing. Further assessment is required, including temporal validation, model calibration, subgroup performance evaluation and clinical stakeholder review.

The model is intended as an exploratory decision-support baseline and not as a replacement for clinical judgement.

### 10. Presentation and Video Submission

The Part C presentation is included in this repository.
https://youtu.be/mo8dIgsmSm8


**Presentation:** `SS3_PartC_Presentation.pptx`

**Video presentation link:** To be added after recording and uploading the presentation video.

**Recorded presentation:** https://youtu.be/mo8dIgsmSm8

### 11. Part A – Peer Review

The Part A peer review report is included in this repository:

`Part_A_Peer_Review_Vusumuzi_Nkosi_2026.docx`

### 12. Conclusion

This project demonstrates the development and evaluation of a Logistic Regression model for predicting 30-day hospital readmission using the supplied synthetic healthcare dataset.

The results demonstrate the potential of machine learning for healthcare risk identification while highlighting the importance of model validation, responsible data use and clinical oversight.
