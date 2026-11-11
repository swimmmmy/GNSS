from pathlib import Path
import pandas as pd

root = Path("GNSS_DATASET_JAMMING_SPOOFING/Meaconing")

files = [
    p for p in root.rglob("rinex.csv")
    if p.parent.name == "3.1.3,3.1.4"
]

if not files:
    raise FileNotFoundError("미코닝 CSV를 찾지 못했습니다.")

file = files[0]
print("읽는 파일:", file)

df = pd.read_csv(file)

names = {
    "time": "측정 시간",
    "satellite": "위성 번호",
    "pseudorange_L1": "L1 의사거리",
    "carrier_phase_L1": "L1 반송파 위상",
    "doppler_L1": "L1 도플러",
    "snr_L1": "L1 신호 품질",
    "pseudorange_L2": "L2 의사거리",
    "carrier_phase_L2": "L2 반송파 위상",
    "doppler_L2": "L2 도플러",
    "snr_L2": "L2 신호 품질"
}

df = df.rename(columns=names)

for col in ["L1 신호 품질", "L2 신호 품질"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

summary = df.groupby("위성 번호").agg(
    L1_평균=("L1 신호 품질", "mean"),
    L1_표준편차=("L1 신호 품질", "std"),
    L1_최댓값=("L1 신호 품질", "max"),
    L2_평균=("L2 신호 품질", "mean"),
    L2_표준편차=("L2 신호 품질", "std"),
    L2_최댓값=("L2 신호 품질", "max")
).reset_index().round(3)

output = "meaconing_3_1_3_3_1_4.xlsx"

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="미코닝 데이터", index=False)
    summary.to_excel(writer, sheet_name="위성별 통계", index=False)

print("엑셀 생성 완료:", output)
print("전체 행:", len(df))
