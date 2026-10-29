
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# 1. 학습된 v2 모델 불러오기
saved = joblib.load("model/gnss_iforest_v2.joblib")

pipeline = saved["pipeline"]
features = saved["features"]
threshold = saved["threshold"]

# 2. 검증 데이터 불러오기
df = pd.read_csv("model/validation_results_v2.csv")

normal = df[features].dropna().copy()

# 3. 테스트할 상황
tests = {
    "정상 원본": {},

    "HDOP 증가": {
        "hdop": 3.0
    },

    "신호 세기 감소": {
        "cn0_mean_dbhz": 20.0,
        "cn0_max_dbhz": 30.0
    },

    "신호 세기 증가": {
        "cn0_mean_dbhz": 45.0,
        "cn0_max_dbhz": 50.0
    },

    "신호 측정 수 감소": {
        "cn0_count": 5
    },

    "복합 변화": {
        "hdop": 3.0,
        "cn0_mean_dbhz": 20.0,
        "cn0_max_dbhz": 30.0,
        "cn0_count": 5
    }
}

results = []

# 4. 상황별 AI 테스트
for name, changes in tests.items():
    sample = normal.copy()

    for column, value in changes.items():
        sample[column] = value

    scores = pipeline.decision_function(sample[features])
    predictions = scores < threshold

    detected = int(np.sum(predictions))
    rate = detected / len(sample) * 100

    results.append({
        "테스트": name,
        "전체": len(sample),
        "이상 의심": detected,
        "비율(%)": round(rate, 2),
        "평균 점수": round(float(np.mean(scores)), 4)
    })

# 5. 결과 출력
result_df = pd.DataFrame(results)

print("\n=== Isolation Forest v2 테스트 ===")
print(result_df.to_string(index=False))

# 6. 결과 저장
Path("model").mkdir(exist_ok=True)

result_df.to_csv(
    "model/test_results_v2.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n결과 저장: model/test_results_v2.csv")
