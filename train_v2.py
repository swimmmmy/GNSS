
"""1차 GNSS 이상 탐지 v2: 신호 품질 중심 Isolation Forest."""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# 1. AI 학습에 사용할 특징값
FEATURES = [
    "hdop",
    "cn0_mean_dbhz",
    "cn0_max_dbhz",
    "cn0_count"
]

# 2. 학습에 사용하지 않을 검증 파일
VALIDATION_FILES = {"test_07", "test_08"}


def main():
    parser = argparse.ArgumentParser(
        description="GNSS Isolation Forest v2 학습"
    )

    parser.add_argument(
        "--input",
        default="gnss_parsed.csv"
    )

    parser.add_argument(
        "--outdir",
        default="model"
    )

    args = parser.parse_args()

    # 3. CSV 불러오기
    df = pd.read_csv(
        args.input,
        dtype={
            "utc_time": str,
            "date_ddmmyy": str
        }
    )

    # 필요한 컬럼 확인
    required = ["source_file", *FEATURES]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CSV에 필요한 컬럼이 없습니다: {missing}"
        )

    # 4. 학습 / 검증 데이터 분리
    groups = (
        df["source_file"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[0]
    )

    is_valid = groups.isin(VALIDATION_FILES)

    train = df.loc[~is_valid].copy()
    valid = df.loc[is_valid].copy()

    if train.empty or valid.empty:
        raise ValueError(
            "학습 또는 검증 데이터가 비어 있습니다."
        )

    # 5. 숫자 데이터로 변환
    for col in FEATURES:
        train[col] = pd.to_numeric(
            train[col], errors="coerce"
        )

        valid[col] = pd.to_numeric(
            valid[col], errors="coerce"
        )

    # 모든 값이 비어 있는 컬럼 확인
    all_missing = [
        col for col in FEATURES
        if train[col].notna().sum() == 0
    ]

    if all_missing:
        raise ValueError(
            f"유효한 값이 없는 컬럼: {all_missing}"
        )

    # 6. Isolation Forest 모델 생성
    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "forest",
            IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # 7. 정상 GNSS 데이터 학습
    pipeline.fit(train[FEATURES])

    # 8. 학습 데이터 이상 점수 계산
    train_scores = pipeline.decision_function(
        train[FEATURES]
    )

    # 초기 임계값: 학습 점수의 하위 1%
    threshold = float(
        np.quantile(train_scores, 0.01)
    )

    # 9. 검증 데이터 이상 점수 계산
    valid_scores = pipeline.decision_function(
        valid[FEATURES]
    )

    valid["anomaly_score"] = valid_scores

    valid["prediction"] = np.where(
        valid_scores < threshold,
        "이상 의심",
        "정상 패턴"
    )

    # 10. 결과 저장 폴더
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 11. 학습된 AI 모델 저장
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": FEATURES,
            "threshold": threshold
        },
        out / "gnss_iforest_v2.joblib"
    )

    # 12. 검증 결과 CSV 저장
    valid.to_csv(
        out / "validation_results_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 13. 학습 결과 보고서
    suspected_count = int(
        (valid_scores < threshold).sum()
    )

    suspected_rate = round(
        suspected_count / len(valid) * 100,
        2
    )

    report = {
        "model_version": "v2",
        "features": FEATURES,
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "train_files": sorted(
            groups[~is_valid].unique().tolist()
        ),
        "validation_files": sorted(
            groups[is_valid].unique().tolist()
        ),
        "threshold": threshold,
        "validation_suspected_count": suspected_count,
        "validation_suspected_rate_percent": suspected_rate,
        "train_missing_values": (
            train[FEATURES].isna().sum().to_dict()
        ),
        "validation_missing_values": (
            valid[FEATURES].isna().sum().to_dict()
        )
    }

    (out / "report_v2.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # 14. 실행 결과 출력
    print("\n=== 1차 Isolation Forest v2 학습 완료 ===")

    print("사용 특징값:", FEATURES)

    print(
        "학습:",
        report["train_files"],
        f"({len(train):,}행)"
    )

    print(
        "검증:",
        report["validation_files"],
        f"({len(valid):,}행)"
    )

    print(f"이상 판단 기준값: {threshold:.6f}")

    print(
        f"검증 데이터 이상 의심: "
        f"{suspected_count}행 ({suspected_rate}%)"
    )

    print("결과 저장:", out)

    print(
        "주의: 이상 의심은 실제 GPS 공격 확정이 아닙니다."
    )


if __name__ == "__main__":
    main()
