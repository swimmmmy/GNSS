"""Stage-2 NAV-PVT Random Forest: leave-one-experiment-out evaluation.
Run after: python build_stage2_nav_rf.py
Labels remain PROVISIONAL (attack timestamps assumed Norway local UTC+2).
Does not overwrite the trained model or raw data.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

BASE = Path('stage2_nav_rf')
DATA = BASE / 'nav_features_provisional.csv'
OUT = BASE / 'cross_validation'
FEATURES = ['fixType','numSV','hAcc','vAcc','sAcc','headAcc','pDOP','gSpeed']
for sec in (1,3,5):
    FEATURES.extend(f'{x}_delta_{sec}s' for x in ('distance_m','speed_mps','hAcc','pDOP','numSV'))

def main():
    if not DATA.exists():
        raise SystemExit('먼저 python build_stage2_nav_rf.py 를 실행하세요.')
    df = pd.read_csv(DATA, low_memory=False)
    missing = [c for c in ['experiment','label',*FEATURES] if c not in df.columns]
    if missing: raise SystemExit(f'누락 컬럼: {missing}')
    df = df[df['label'].isin([0,1])].copy()
    df['label'] = df['label'].astype(int)
    for c in FEATURES: df[c] = pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    experiments = sorted(df['experiment'].dropna().unique())
    if len(experiments)<3: raise SystemExit('서로 다른 실험 3개 이상 필요')
    OUT.mkdir(parents=True, exist_ok=True)
    results=[]; predictions=[]
    for exp in experiments:
        train = df[df.experiment != exp]
        test = df[df.experiment == exp]
        if train.label.nunique()!=2 or test.label.nunique()!=2:
            print(f'SKIP {exp}: 학습 또는 검증에 정상/공격 두 클래스가 없음')
            continue
        if any(train[c].notna().sum()==0 for c in FEATURES):
            print(f'SKIP {exp}: 학습에 완전히 누락된 특징값 있음')
            continue
        model = make_pipeline(
            SimpleImputer(strategy='median'),
            RandomForestClassifier(n_estimators=200,min_samples_leaf=5,
                                   class_weight='balanced',random_state=42,n_jobs=-1)
        )
        model.fit(train[FEATURES],train['label'])
        pred = model.predict(test[FEATURES])
        prob = model.predict_proba(test[FEATURES])[:,list(model.classes_).index(1)] if hasattr(model,'classes_') else model.predict_proba(test[FEATURES])[:,1]
        tn,fp,fn,tp = confusion_matrix(test.label,pred,labels=[0,1]).ravel()
        row = {'test_experiment':exp,'train_rows':len(train),'test_rows':len(test),
               'normal_rows':int(tn+fp),'attack_rows':int(fn+tp),
               'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
               'accuracy':float(accuracy_score(test.label,pred)),
               'precision':float(precision_score(test.label,pred,zero_division=0)),
               'recall':float(recall_score(test.label,pred,zero_division=0)),
               'f1':float(f1_score(test.label,pred,zero_division=0)),
               'false_positive_rate':float(fp/(fp+tn)) if fp+tn else None}
        results.append(row)
        out=test[['experiment','label']].copy()
        if 'second' in test: out['second']=test['second']
        out['prediction']=pred;out['meaconing_probability']=prob
        predictions.append(out)
        print(f"{exp}: 정상 오탐 {fp}/{tn+fp}, 미코닝 탐지 {tp}/{tp+fn} ({row['recall']*100:.2f}%)")
    if not results: raise SystemExit('검증 가능한 실험이 없습니다.')
    report=pd.DataFrame(results)
    report.to_csv(OUT/'per_experiment_metrics.csv',index=False,encoding='utf-8-sig')
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'all_predictions.csv',index=False,encoding='utf-8-sig')
    totals=report[['TN','FP','FN','TP']].sum()
    tp,fn,fp,tn=(int(totals[k]) for k in ['TP','FN','FP','TN'])
    summary={'folds':len(results),'total_confusion_matrix':[[tn,fp],[fn,tp]],
             'pooled_recall':tp/(tp+fn) if tp+fn else None,
             'pooled_fpr':fp/(fp+tn) if fp+tn else None,
             'macro_recall':float(report.recall.mean()),
             'warning':'PROVISIONAL labels: UTC+2 hypothesis not independently confirmed. Other receiver/hardware not validated. Multiple 1-second samples within each experiment are temporally correlated.'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== 교차검증 완료 (임시 라벨 기준) ===')
    print('실험별 결과:', OUT/'per_experiment_metrics.csv')
    print('전체 예측:', OUT/'all_predictions.csv')
    print('요약:', OUT/'summary.json')
    print('전체 혼동행렬:',summary['total_confusion_matrix'])
    print('평균 탐지율:',round(summary['macro_recall']*100,2),'%')
    print('주의: 시간대 가설과 수신기 차이 검증 전에는 실제 공격 탐지 성능으로 발표하지 마세요.')

if __name__=='__main__': main()
