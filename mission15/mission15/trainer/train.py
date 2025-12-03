import os, json, argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib

# ==========================
# 상수 정의 (데이터셋의 피처/타깃 변수)
# ==========================
FEATURES = [
    'Hours Studied',                   # 공부한 시간
    'Previous Scores',                 # 이전 점수
    'Extracurricular Activities',      # 비교과 활동 여부 (Yes/No)
    'Sleep Hours',                     # 수면 시간
    'Sample Question Papers Practiced' # 모의 문제 풀이 수
]
TARGET = 'Performance Index'           # 예측 목표 변수 (성취 지수)


# ==========================
# 머신러닝 파이프라인 구성
# ==========================
def build_pipeline():
    # 수치형 변수와 범주형 변수 구분
    num_features = [
        "Hours Studied",
        "Previous Scores",
        "Sleep Hours",
        "Sample Question Papers Practiced"
    ]
    cat_features = ["Extracurricular Activities"]

    # 전처리 정의
    pre = ColumnTransformer(
        transformers=[
            # 수치형은 StandardScaler 적용
            ("num", StandardScaler(), num_features),
            # 범주형은 Yes/No를 0/1로 인코딩
            ("cat", OrdinalEncoder(categories=[["No", "Yes"]], dtype="int8"), cat_features),
        ],
        remainder="drop",                    # 지정되지 않은 컬럼은 제거
        verbose_feature_names_out=False,     # 피처 이름을 간단하게 유지
    )

    # 전처리 + Ridge 회귀 모델 결합
    return Pipeline(steps=[
        ("preprocess", pre),
        ("model", Ridge(random_state=42)),
    ])


# ==========================
# 메인 학습 함수
# ==========================
def main(args):
    # 데이터 불러오기 및 중복 제거
    df = pd.read_csv(args.train_csv).drop_duplicates()
    X = df[FEATURES].copy()
    y = df[TARGET].values

    # 파이프라인 생성
    pipe = build_pipeline()

    # 학습/검증 데이터 분리
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.valid_size, random_state=42
    )

    # GridSearchCV로 Ridge 회귀의 alpha(정규화 강도) 튜닝
    gs = GridSearchCV(
        pipe,
        param_grid={"model__alpha":[0.01,0.1,1.0,3.0,10.0,30.0,100.0]}, # 후보 알파 값
        scoring="neg_root_mean_squared_error",  # RMSE (부호만 음수로 반환됨)
        cv=args.cv,                             # 교차 검증 폴드 수
        n_jobs=-1                               # 병렬 처리
    )
    gs.fit(X_tr, y_tr)

    # 최적 파이프라인 선택
    best_pipe = gs.best_estimator_

    # 검증 데이터셋에서 성능 평가
    y_pred = best_pipe.predict(X_va)
    holdout_rmse = float(np.sqrt(mean_squared_error(y_va, y_pred)))
    cv_best_rmse = float(-gs.best_score_)  # GridSearchCV의 점수는 음수이므로 부호 반전

    # 결과 출력
    print(f"[Holdout RMSE] {holdout_rmse:.4f}")
    print(f"[CV best RMSE] {cv_best_rmse:.4f}")
    print(f"[Best alpha]   {gs.best_params_['model__alpha']}")

    # 최종 모델 (선택적으로 전체 데이터로 재학습)
    final_pipe = best_pipe
    if args.refit_full:
        final_pipe = gs.best_estimator_.set_params(**gs.best_params_)
        final_pipe.fit(X, y)

    # 모델 저장
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(final_pipe, args.model_out)
    print(f"Saved → {args.model_out}")

    # 메트릭 저장 (json 형식)
    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump({
                "holdout_rmse": holdout_rmse,
                "cv_best_rmse": cv_best_rmse,
                "best_alpha": gs.best_params_["model__alpha"],
                "features": FEATURES,
                "target": TARGET,
                "refit_full": bool(args.refit_full),
            }, f, ensure_ascii=False, indent=2)


# ==========================
# CLI 실행부 (파라미터 받기)
# ==========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="훈련 데이터 CSV 경로")
    p.add_argument("--model_out", default="model.pkl", help="저장할 모델 파일명")
    p.add_argument("--metrics_out", default="metrics.json", help="저장할 메트릭 파일명")
    p.add_argument("--valid_size", type=float, default=0.2, help="검증 데이터 비율")
    p.add_argument("--cv", type=int, default=5, help="교차 검증 폴드 수")
    p.add_argument("--refit_full", action="store_true", help="전체 데이터로 최종 재학습 여부")
    args = p.parse_args()
    main(args)