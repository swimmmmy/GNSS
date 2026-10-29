
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 파싱된 GPS 데이터 불러오기
df = pd.read_csv("gnss_parsed.csv")

print("=== GNSS 데이터 분석 ===")
print(f"전체 데이터: {len(df)}개")

# 2. 데이터가 빠진 부분 확인
print("\n=== 누락된 데이터 ===")
print(df.isnull().sum())

# 3. GPS 측정값의 기본 통계
features = [
    "speed_kmh",
    "satellites_used",
    "hdop",
    "altitude_m",
    "cn0_mean_dbhz",
    "cn0_max_dbhz"
]

print("\n=== GPS 데이터 통계 ===")
print(df[features].describe())

# 4. 그래프 저장 폴더 생성
Path("graphs").mkdir(exist_ok=True)

# 5. GPS 데이터 변화 그래프 생성
for feature in features:
    plt.figure(figsize=(11, 4))

    # 파일별 데이터가 섞이지 않도록 구분
    for name, group in df.groupby("source_file", sort=False):
        plt.plot(
            range(len(group)),
            group[feature],
            label=name,
            alpha=0.8
        )

    plt.title(f"GNSS - {feature}")
    plt.xlabel("Sample Index (per file)")
    plt.ylabel(feature)
    plt.legend(fontsize=7)
    plt.tight_layout()

    plt.savefig(f"graphs/{feature}.png", dpi=150)
    plt.close()

print("\n분석 완료! graphs 폴더를 확인하세요.")
