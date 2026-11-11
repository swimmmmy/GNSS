"""Train a stage-2 Random Forest ONLY on verified, labeled experiments.

Input: per-experiment CSVs from prepare_stage2.py. The program refuses
unknown labels and requires >=2 distinct experiments in train and test.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

FEATURES = [
    'l1_mean', 'l1_max', 'l1_std', 'l1_count',
    'l1_satellites', 'l1_doppler_abs_mean', 'l1_mean_delta',
    'gSpeed', 'hAcc', 'numSV', 'pDOP',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('stage2_output'))
    p.add_argument('--outdir', type=Path, default=Path('stage2_model'))
    p.add_argument('--test-experiment', help='Exact experiment string to hold out for testing')
    args = p.parse_args()
    files = sorted(args.data_dir.glob('meaconing_features_*.csv'))
    if not files:
        raise SystemExit('먼저 python prepare_stage2.py 를 실행하세요.')
    frames = []
    for file in files:
        df = pd.read_csv(file)
        if not {'label', 'experiment'}.issubset(df.columns):
            raise SystemExit(f'필수 컬럼 누락: {file}')
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    labels = pd.to_numeric(data['label'], errors='coerce')
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise SystemExit('학습 중단: 공격 시각이 검증되지 않은 label=-1/결측 기록이 있습니다. '
                         '시간 기준을 확인하고 라벨을 검증한 뒤 다시 실행하세요.')
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise SystemExit(f'특징값 컬럼 누락: {missing}')
    experiments = sorted(data['experiment'].dropna().unique())
    if len(experiments) < 2:
        raise SystemExit('학습 중단: 서로 다른 실험이 2개 이상 필요합니다.')
    test_exp = args.test_experiment or experiments[-1]
    if test_exp not in experiments:
        raise SystemExit(f'알 수 없는 검증 실험: {test_exp}\n가능한 값: {experiments}')
    train = data[data['experiment'] != test_exp].copy()
    test = data[data['experiment'] == test_exp].copy()
    if train.empty or test.empty or train['label'].nunique() != 2 or test['label'].nunique() != 2:
        raise SystemExit('학습 중단: 학습/검증 실험 각각에 정상(0)과 공격(1) 기록이 필요합니다.')
    for col in FEATURES:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        test[col] = pd.to_numeric(test[col], errors='coerce')
    if any(train[c].notna().sum() == 0 for c in FEATURES):
        raise SystemExit('학습 중단: 일부 특징값이 학습 구간에서 모두 비어 있습니다.')
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('forest', RandomForestClassifier(n_estimators=200, random_state=42,
                                         class_weight='balanced', n_jobs=-1,
                                         min_samples_leaf=5))
    ])
    pipe.fit(train[FEATURES], train['label'].astype(int))
    pred = pipe.predict(test[FEATURES])
    prob = pipe.predict_proba(test[FEATURES])[:, list(pipe.classes_).index(1)]
    report = classification_report(test['label'].astype(int), pred,
                                   labels=[0,1], target_names=['정상','미코닝'],
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(test['label'].astype(int), pred, labels=[0,1]).tolist()
    args.outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': pipe, 'features': FEATURES, 'classes': [0,1]},
                args.outdir/'meaconing_rf_v1.joblib')
    result = test[['experiment', 'second', 'label']].copy()
    result['prediction'] = pred
    result['meaconing_probability'] = prob
    result.to_csv(args.outdir/'predictions.csv', index=False, encoding='utf-8-sig')
    summary = {'train_experiments': sorted(train['experiment'].unique().tolist()),
               'test_experiment': test_exp, 'train_rows': len(train),
               'test_rows': len(test), 'features': FEATURES,
               'confusion_matrix_0_1': matrix, 'classification_report': report,
               'warning': 'Performance is on held-out experiment only; receiver/device transfer not validated.'}
    (args.outdir/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('2차 Random Forest 학습 완료')
    print('학습 실험:', summary['train_experiments'])
    print('검증 실험:', test_exp)
    print('혼동행렬 [[TN, FP], [FN, TP]]:', matrix)
    print('미코닝 재현율:', round(report['미코닝']['recall'], 4))
    print('결과 저장:', args.outdir)

if __name__ == '__main__':
    main()
