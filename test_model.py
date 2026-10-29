
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# 학습된 모델 불러오기
saved = joblib.load("model/gnss_iforest_v1.joblib")

# 저장 구조에 맞춰 모델과 전처리기 확인
print("저장된 객체 유형:", type(saved))
if isinstance(saved, dict):
    print("저장된 항목:", list(saved.keys()))

# 기존 정상 검증 데이터 불러오기
df = pd.read_csv("model/validation_results.csv")

features = [
    "satellites_used",
    "hdop",
    "cn0_mean_dbhz",
    "cn0_max_dbhz",
    "speed_kmh"
]

normal = df[features].dropna().head(100).copy()

# 테스트용 이상 데이터 생성
synthetic = normal.copy()

# 위성 수 감소
synthetic["satellites_used"] = 4

# HDOP 증가
synthetic["hdop"] = 8.0

# 평균 및 최대 신호 세기 감소
synthetic["cn0_mean_dbhz"] = 12.0
synthetic["cn0_max_dbhz"] = 18.0

# 비정상적으로 높은 이동 속도
synthetic["speed_kmh"] = 120.0

Path("model").mkdir(exist_ok=True)
synthetic.to_csv(
    "model/synthetic_test.csv",
    index=False
)

print("합성 이상 데이터 100행 생성 완료!")
print("model/synthetic_test.csv")


# 1. 저장된 모델과 설정 가져오기
pipeline = saved["pipeline"]
features = saved["features"]
threshold = saved["threshold"]

# 2. 정상 데이터와 합성 이상 데이터 예측
normal_scores = pipeline.decision_function(normal[features])
synthetic_scores = pipeline.decision_function(synthetic[features])

# 3. 저장된 기준값으로 정상/이상 판단
normal_predictions = np.where(
    normal_scores < threshold, "이상 의심", "정상 패턴"
)

synthetic_predictions = np.where(
    synthetic_scores < threshold, "이상 의심", "정상 패턴"
)

# 4. 결과 출력
normal_count = np.sum(normal_predictions == "이상 의심")
synthetic_count = np.sum(synthetic_predictions == "이상 의심")

print("\n=== 1차 Isolation Forest 테스트 ===")
print(f"정상 데이터 이상 의심: {normal_count}/100")
print(f"합성 이상 데이터 탐지: {synthetic_count}/100")
print(f"합성 이상 탐지 비율: {synthetic_count}%")

# 5. 결과 CSV 저장
result = synthetic.copy()
result["anomaly_score"] = synthetic_scores
result["prediction"] = synthetic_predictions

result.to_csv(
    "model/synthetic_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("결과 저장: model/synthetic_predictions.csv")
