"""1차 GNSS 이상 탐지: 정상 기록만으로 Isolation Forest 학습."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES = ['satellites_used', 'hdop', 'cn0_mean_dbhz', 'cn0_max_dbhz', 'speed_kmh']
VALIDATION_FILES = {'test_07', 'test_08'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='gnss_parsed.csv')
    p.add_argument('--outdir', default='model')
    args = p.parse_args()
    df = pd.read_csv(args.input, dtype={'utc_time': str, 'date_ddmmyy': str})
    missing = [c for c in ['source_file', *FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f'CSV 컬럼이 없습니다: {missing}')
    groups = df['source_file'].astype(str).str.replace('\\', '/', regex=False).str.split('/').str[0]
    is_valid = groups.isin(VALIDATION_FILES)
    train, valid = df.loc[~is_valid].copy(), df.loc[is_valid].copy()
    if train.empty or valid.empty:
        raise ValueError('학습/검증 데이터가 비어 있습니다. source_file을 확인하세요.')
    for c in FEATURES:
        train[c] = pd.to_numeric(train[c], errors='coerce')
        valid[c] = pd.to_numeric(valid[c], errors='coerce')
    all_missing = [c for c in FEATURES if train[c].notna().sum() == 0]
    if all_missing:
        raise ValueError(f'학습 데이터에 유효한 값이 없는 컬럼: {all_missing}')
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('forest', IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=-1)),
    ])
    pipe.fit(train[FEATURES])
    train_scores = pipe.decision_function(train[FEATURES])
    # 초기 기준값: 학습 정상 기록의 하위 1% (성능이 검증된 임계값은 아님)
    threshold = float(np.quantile(train_scores, 0.01))
    valid_scores = pipe.decision_function(valid[FEATURES])
    valid['anomaly_score'] = valid_scores
    valid['prediction'] = np.where(valid_scores < threshold, '이상 의심', '정상 패턴')
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': pipe, 'features': FEATURES, 'threshold': threshold}, out / 'gnss_iforest_v1.joblib')
    valid.to_csv(out / 'validation_results.csv', index=False, encoding='utf-8-sig')
    report = {
        'train_rows': int(len(train)), 'validation_rows': int(len(valid)),
        'train_files': sorted(groups[~is_valid].unique().tolist()),
        'validation_files': sorted(groups[is_valid].unique().tolist()),
        'threshold': threshold,
        'validation_suspected_count': int((valid_scores < threshold).sum()),
        'validation_suspected_rate_percent': round(float((valid_scores < threshold).mean() * 100), 2),
        'train_missing_values': train[FEATURES].isna().sum().to_dict(),
        'validation_missing_values': valid[FEATURES].isna().sum().to_dict(),
    }
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('=== 1차 Isolation Forest 학습 완료 ===')
    print('학습:', report['train_files'], f"({len(train):,}행)")
    print('검증:', report['validation_files'], f"({len(valid):,}행)")
    print(f"검증 데이터 이상 의심: {report['validation_suspected_count']}행 ({report['validation_suspected_rate_percent']}%)")
    print('결과 저장:', out)
    print('주의: 이상 의심은 스푸핑 확정이 아닙니다. 검증 데이터가 실제 정상이라는 전제가 필요합니다.')

if __name__ == '__main__':
    main()
