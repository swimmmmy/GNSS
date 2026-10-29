
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

saved = joblib.load("model/gnss_iforest_v1.joblib")

pipeline = saved["pipeline"]
features = saved["features"]
threshold = saved["threshold"]

df = pd.read_csv("model/validation_results.csv")

# 정상 검증 데이터 중 100개 사용
normal = df[features].dropna().head(100).copy()

# 테스트 상황 정의
tests = {
    "정상 원본": {},

    "위성 수 감소": {
        "satellites_used": 5
    },

    "HDOP 증가": {
        "hdop": 3.0
    },

    "신호 세기 감소": {
        "cn0_mean_dbhz": 20.0,
        "cn0_max_dbhz": 30.0
    },

    "속도 증가": {
        "speed_kmh": 15.0
    },

    "복합 변화": {
        "satellites_used": 8,
        "hdop": 1.5,
        "cn0_mean_dbhz": 25.0,
        "cn0_max_dbhz": 35.0,
        "speed_kmh": 8.0
    }
}

results = []

for name, changes in tests.items():
    sample = normal.copy()

    for column, value in changes.items():
        sample[column] = value

    scores = pipeline.decision_function(sample[features])
    predictions = scores < threshold

    detected = int(np.sum(predictions))

    results.append({
        "테스트": name,
        "전체": len(sample),
        "이상 의심": detected,
        "탐지 비율(%)": round(detected / len(sample) * 100, 2),
        "평균 이상 점수": round(float(np.mean(scores)), 4)
    })

result_df = pd.DataFrame(results)

print("\n=== 특징값별 민감도 테스트 ===")
print(result_df.to_string(index=False))

Path("model").mkdir(exist_ok=True)
result_df.to_csv(
    "model/feature_test_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n결과 저장: model/feature_test_results.csv")
