"""Stage-2 GNSS RF audit: group holdout, leakage-safe features, permutation importance.
Reads stage2_nav_rf/nav_features_provisional.csv without modifying it.
All labels are PROVISIONAL until timestamp offset is independently validated.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.inspection import permutation_importance

SRC = Path('stage2_nav_rf/nav_features_provisional.csv')
OUT = Path('stage2_nav_rf/leakage_audit')
OUT.mkdir(parents=True, exist_ok=True)
if not SRC.exists():
    raise SystemExit(f'파일 없음: {SRC}\n먼저 python build_stage2_nav_rf.py 를 실행하세요.')
df = pd.read_csv(SRC, low_memory=False)
print('입력 파일:', SRC, '행:', len(df))
print('컬럼:', ', '.join(df.columns))

def pick(candidates):
    return next((c for c in candidates if c in df.columns), None)
group_col = pick(['experiment','scenario','scenario_id','experiment_id'])
label_col = pick(['label','attack_label','target','y','is_attack','is_meaconing'])
if not group_col or not label_col:
    raise SystemExit('실험 구분 또는 라벨 컬럼을 찾지 못했습니다. 출력된 컬럼명을 확인해 주세요.')

labels = df[label_col]
if pd.api.types.is_numeric_dtype(labels):
    y = pd.to_numeric(labels, errors='coerce')
else:
    vals = labels.astype(str).str.strip().str.lower()
    mapping = {'normal':0,'0':0,'meaconing':1,'attack':1,'1':1,'미코닝':1,'정상':0}
    y = vals.map(mapping)
valid = y.isin([0,1]) & df[group_col].notna()
df = df.loc[valid].copy()
df['_audit_y'] = y.loc[valid].astype(int)
if df[group_col].nunique() < 3:
    raise SystemExit('서로 다른 실험이 최소 3개 필요합니다.')

# Never let IDs, time, raw position or experiment metadata become model features.
blocked_exact = {
    group_col,label_col,'_audit_y','receiver_state','real_time','timestamp','time',
    'date','datetime','iTOW','lat','lon','latitude','longitude',
    'year','month','day','hour','min','sec','identity','satellite','svId','gnssId',
    'scenario_id','scenario','experiment_id','source_file','attack_type',
    'power','frequency_band','rover_state','label_source','label_status'
}
blocked_words = ['timestamp','datetime','experiment','scenario','label','target','prediction',
                 'probability','attack_start','attack_end','time_of_day','elapsed',
                 'latitude','longitude','source_file','receiver_state','_id']
feature_cols = []
for col in df.columns:
    if col in blocked_exact or any(w in col.lower() for w in blocked_words):
        continue
    numeric = pd.to_numeric(df[col], errors='coerce')
    if numeric.notna().sum() < max(20, len(df)*0.2) or numeric.nunique(dropna=True) < 2:
        continue
    feature_cols.append(col)
if not feature_cols:
    raise SystemExit('안전한 숫자 특징값이 없습니다. 전처리 코드에서 변화량 컬럼을 확인하세요.')
print('사용 특징값:', feature_cols)
X = df[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)

rows=[]; importances=[]
for experiment in sorted(df[group_col].unique()):
    test_mask = df[group_col].eq(experiment).to_numpy()
    train_mask = ~test_mask
    y_train = df.loc[train_mask,'_audit_y'].to_numpy()
    y_test = df.loc[test_mask,'_audit_y'].to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print('건너뜀 (한 클래스만 존재):',experiment)
        continue
    model = make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(
        n_estimators=200,random_state=42,n_jobs=-1,class_weight='balanced_subsample',
        min_samples_leaf=5))
    model.fit(X.loc[train_mask],y_train)
    pred = model.predict(X.loc[test_mask])
    tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    rows.append({'experiment':experiment,'n_test':len(y_test),'tn':int(tn),'fp':int(fp),
                 'fn':int(fn),'tp':int(tp),'accuracy':accuracy_score(y_test,pred),
                 'recall':recall_score(y_test,pred,zero_division=0),
                 'precision':precision_score(y_test,pred,zero_division=0),
                 'fpr':fp/(tn+fp) if tn+fp else np.nan})
    # Permutation on unseen experiment: more meaningful than tree impurity importance.
    imp=permutation_importance(model,X.loc[test_mask],y_test,n_repeats=3,
                               random_state=42,scoring='balanced_accuracy',n_jobs=1)
    importances.extend({'experiment':experiment,'feature':col,'importance':float(score)}
                       for col,score in zip(feature_cols,imp.importances_mean))
    print(f'{experiment}: 정상 오탐 {fp}/{tn+fp}, 공격 탐지 {tp}/{tp+fn}')
if not rows:
    raise SystemExit('검증 가능한 실험이 없습니다.')
metrics=pd.DataFrame(rows)
importance=pd.DataFrame(importances)
metrics.to_csv(OUT/'per_experiment_metrics.csv',index=False,encoding='utf-8-sig')
importance.to_csv(OUT/'permutation_importance_by_experiment.csv',index=False,encoding='utf-8-sig')
importance.groupby('feature',as_index=False)['importance'].mean().sort_values('importance',ascending=False).to_csv(
    OUT/'feature_importance.csv',index=False,encoding='utf-8-sig')
tn,fp,fn,tp=[int(metrics[c].sum()) for c in ['tn','fp','fn','tp']]
report={'warning':'PROVISIONAL: attack time offset not independently confirmed',
        'input':str(SRC),'label_column':label_col,'group_column':group_col,
        'excluded_columns':sorted(set(df.columns)-set(feature_cols)-{'_audit_y'}),
        'features':feature_cols,'confusion_matrix':[[tn,fp],[fn,tp]],
        'overall_recall':tp/(tp+fn) if tp+fn else None,
        'overall_fpr':fp/(fp+tn) if fp+tn else None,
        'overall_accuracy':(tn+tp)/(tn+fp+fn+tp)}
(OUT/'audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('\n=== 누수 위험 컬럼 제외 후 교차검증 ===')
print('전체 혼동행렬:',report['confusion_matrix'])
print(f"탐지율: {report['overall_recall']:.2%} / 오탐률: {report['overall_fpr']:.2%}")
print('결과 저장:',OUT)
print('주의: 시간 보정 가설이 검증되기 전의 임시 라벨 결과입니다.')
