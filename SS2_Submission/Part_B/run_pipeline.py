"""CAP182 SS2: reproducible 30-day diabetes readmission proof of concept.
Run: python Part_B/run_pipeline.py --data Datasets/diabetic_data.csv --output Part_C/results
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)

DROP = ['encounter_id', 'patient_nbr', 'readmitted', 'weight', 'payer_code',
        'medical_specialty', 'max_glu_serum', 'A1Cresult', 'discharge_disposition_id']

def diagnosis_group(code):
    if pd.isna(code): return 'missing'
    s = str(code).strip()
    if s.startswith('V'): return 'supplementary_V'
    if s.startswith('E'): return 'external_E'
    try: v = float(s)
    except ValueError: return 'other'
    if 390 <= v < 460 or v == 785: return 'circulatory'
    if 460 <= v < 520 or v == 786: return 'respiratory'
    if 520 <= v < 580 or v == 787: return 'digestive'
    if 250 <= v < 251: return 'diabetes'
    if 800 <= v < 1000: return 'injury'
    if 710 <= v < 740: return 'musculoskeletal'
    if 580 <= v < 630 or v == 788: return 'genitourinary'
    if 140 <= v < 240: return 'neoplasm'
    return 'other'

def prepare(df):
    df = df.replace('?', np.nan).copy()
    y = (df['readmitted'] == '<30').astype(int)
    groups = df['patient_nbr'].astype(str)
    X = df.drop(columns=DROP, errors='ignore')
    for col in ['diag_1', 'diag_2', 'diag_3']:
        X[col] = X[col].map(diagnosis_group)
    X['age'] = X['age'].astype('string').fillna('Unknown')
    for col in ['admission_type_id', 'admission_source_id']:
        X[col] = X[col].astype('string')
    X['prior_utilisation_total'] = (pd.to_numeric(X['number_inpatient'], errors='coerce').fillna(0)
        + pd.to_numeric(X['number_outpatient'], errors='coerce').fillna(0)
        + pd.to_numeric(X['number_emergency'], errors='coerce').fillna(0))
    return X, y, groups

def make_preprocessor(X):
    cat = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat)
    ])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='Datasets/diabetic_data.csv')
    p.add_argument('--output', default='Part_C/results')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, low_memory=False)
    X, y, groups = prepare(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=42)
    train, test = next(splitter.split(X, y, groups))
    assert not set(groups.iloc[train]).intersection(set(groups.iloc[test]))
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1500, class_weight='balanced', solver='liblinear', random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5,
                class_weight='balanced_subsample', n_jobs=-1, random_state=42)
    }
    results = []
    fig_roc, ax_roc = plt.subplots(figsize=(7,5))
    fig_pr, ax_pr = plt.subplots(figsize=(7,5))
    for name, estimator in models.items():
        pipe = Pipeline([('preprocess', make_preprocessor(X_train)), ('model', estimator)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:,1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
        record = {'model': name, 'n_train':len(train), 'n_test':len(test),
          'test_positive_rate':float(y_test.mean()), 'threshold':0.5,
          'accuracy':accuracy_score(y_test,pred), 'precision':precision_score(y_test,pred,zero_division=0),
          'recall':recall_score(y_test,pred,zero_division=0), 'f1':f1_score(y_test,pred,zero_division=0),
          'roc_auc':roc_auc_score(y_test,prob), 'average_precision':average_precision_score(y_test,prob),
          'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
        results.append(record)
        pd.DataFrame({'encounter_id':df.iloc[test]['encounter_id'].values,
                      'actual':y_test.values, 'probability':prob, 'predicted':pred}).to_csv(out/f'{name}_test_predictions.csv',index=False)
        fpr,tpr,_ = roc_curve(y_test,prob)
        ax_roc.plot(fpr,tpr,label=f'{name} (AUC={record["roc_auc"]:.3f})')
        precision,recall,_ = precision_recall_curve(y_test,prob)
        ax_pr.plot(recall,precision,label=f'{name} (AP={record["average_precision"]:.3f})')
        fig,ax=plt.subplots(figsize=(4.5,4))
        ax.imshow([[tn,fp],[fn,tp]],cmap='Greys')
        for i,row in enumerate([[tn,fp],[fn,tp]]):
            for j,val in enumerate(row): ax.text(j,i,str(val),ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        ax.set_xticks([0,1],['Predicted 0','Predicted 1']); ax.set_yticks([0,1],['Actual 0','Actual 1'])
        ax.set_title(name.replace('_',' ')); fig.tight_layout(); fig.savefig(out/f'{name}_confusion_matrix.png',dpi=160); plt.close(fig)
        print(name,record,flush=True)
    ax_roc.plot([0,1],[0,1],'--',color='grey');ax_roc.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC comparison')
    ax_roc.legend();fig_roc.tight_layout();fig_roc.savefig(out/'roc_comparison.png',dpi=160);plt.close(fig_roc)
    ax_pr.axhline(y_test.mean(),linestyle='--',color='grey',label='Prevalence baseline')
    ax_pr.set(xlabel='Recall',ylabel='Precision',title='Precision–recall comparison');ax_pr.legend()
    fig_pr.tight_layout();fig_pr.savefig(out/'precision_recall_comparison.png',dpi=160);plt.close(fig_pr)
    pd.DataFrame(results).to_csv(out/'performance_metrics.csv',index=False)
    with open(out/'experiment_config.json','w') as f:
        json.dump({'random_state':42,'split':'GroupShuffleSplit patient-level 80/20',
                   'positive_class':'<30','excluded_columns':DROP,
                   'note':'Exploratory holdout only; no external validation or tuning'},f,indent=2)

if __name__ == '__main__': main()
