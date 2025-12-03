# backend.py
# -----------------------------------------------------------------------------
# 프로젝트: 영화 관리 + 리뷰 감성/별점 추론 API (FastAPI + ONNX Runtime)
# 개요:
#  - 영화/리뷰 CRUD, 리뷰별 감성/별점 추론, 평균 별점 집계 API 제공
#  - 추론 모델: INT8 양자화된 ONNX(1~5 클래스 별점 분류)
#  - 저장소: 인메모리 리스트(실서비스 시 DB로 대체 가능; 보고서의 ERD/DDL 참고)
# -----------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware  # (선택) 브라우저 직접 호출 허용시 사용

from pydantic import BaseModel, Field, HttpUrl, constr
from typing import List, Optional
from enum import Enum
from datetime import date, datetime, timezone
import itertools
import os

# ============================= ONNX 추론 준비 (CPU 고정) =============================
# - 모델은 1~5 별점 분류를 수행하고, 각 별점의 확률분포(logits→softmax)를 출력
# - transformers 토크나이저로 텍스트를 토큰화하여 ORT 세션에 입력
# - INT8 양자화 모델 사용으로 CPU에서도 경량 추론
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

STAR_MODEL_DIR = "models/star-rating-onnx-int8"
ONNX_FILE = os.path.join(STAR_MODEL_DIR, "model_quantized.onnx")

print(">>> ORT available providers:", ort.get_available_providers())

# 모델/토크나이저 로드 안전장치: 파일/경로 점검 + 명확한 에러 메시지
if not os.path.exists(ONNX_FILE):
    raise RuntimeError(f"ONNX file not found: {ONNX_FILE}. Export or path check required.")

try:
    # ORT 세션: CPU EP 고정 (배포 환경에서 GPU 미사용 가정)
    _sess = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX Session: {e}")

try:
    # 프리트레인된 토크나이저 로드 (모델 디렉토리의 vocab/config를 사용)
    _tok = AutoTokenizer.from_pretrained(STAR_MODEL_DIR, use_fast=True)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer from {STAR_MODEL_DIR}: {e}")


def _softmax(x: np.ndarray) -> np.ndarray:
    """안정적인 softmax (log-sum-exp)"""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def infer_review_metrics(content: str) -> dict:
    """
    리뷰 텍스트를 ONNX 모델로 추론하여 감성/별점 지표를 산출.

    계산 방식 요약:
    - 모델 출력: 5개 클래스(별점 1~5)의 확률분포 probs
    - star       : argmax(probs) + 1  (1~5 별점)
    - star_conf  : probs[star-1]      (해당 별점의 확신도)
    - soft_star  : Σ_{i=1..5} (i * probs[i-1])  (확률 가중 평균 별점)
    - sentiment_label :
        star ∈ {1,2} → "negative"
        star = 3     → "neutral"
        star ∈ {4,5} → "positive"
    - sentiment_score : ((star - 3) / 2) * star_conf
        → 범위 [-1, 1] (별점 편차를 확신도로 스케일링)

    반환 필드:
      sentiment_label, sentiment_score, star, star_conf, soft_star
    """
    if not content or not content.strip():
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "star": 3,
            "star_conf": 0.0,
            "soft_star": 3.0,
        }

    # 토큰화 → ONNX 입력 형식 정렬 → 추론
    enc = _tok(content, max_length=256, truncation=True, padding=True, return_tensors="np")
    input_names = {i.name for i in _sess.get_inputs()}           # 세션 입력 이름과 교집합만 전달
    inputs = {k: v for k, v in enc.items() if k in input_names}

    out = _sess.run(None, inputs)                                # 보통 첫 출력이 logits
    logits = out[0] if isinstance(out, (list, tuple)) else out
    if logits.ndim == 1:
        logits = logits[None, :]
    probs = _softmax(logits)[0]                                  # shape (5,)

    idx = int(np.argmax(probs))
    star = idx + 1
    conf = float(probs[idx])
    soft_star = float(np.sum((np.arange(1, 6) * probs)))

    label = "positive" if star >= 4 else ("negative" if star <= 2 else "neutral")
    score_pm1 = ((star - 3) / 2.0) * conf

    return {
        "sentiment_label": label,
        "sentiment_score": round(score_pm1, 4),
        "star": star,
        "star_conf": round(conf, 4),
        "soft_star": round(soft_star, 4),
    }


# ================================== 스키마 정의 ==================================
# Pydantic 모델: 요청/응답 스키마를 명확히 문서화하여 Swagger UI(/docs)에 반영

class Genre(str, Enum):
    """장르 Enum (프론트 multiselect와 동기화)"""
    action      = "액션"
    comedy      = "코미디"
    drama       = "드라마"
    sf          = "SF"
    horror      = "호러"
    romance     = "로맨스"
    thriller    = "스릴러"
    adventure   = "모험"
    crime       = "범죄"
    mystery     = "미스터리"
    documentary = "다큐멘터리"
    animation   = "애니메이션"
    fantasy     = "판타지"
    family      = "가족"
    music       = "음악"
    musical     = "뮤지컬"
    war         = "전쟁"
    sport       = "스포츠"
    disaster    = "재난"


class MovieIn(BaseModel):
    """영화 등록/수정 입력"""
    title: constr(min_length=1)
    release_date: date
    director: constr(min_length=1)
    genre: List[Genre] = Field(default_factory=list)
    poster_url: Optional[HttpUrl] = None


class MovieOut(MovieIn):
    """영화 조회 응답"""
    id: int


class ReviewIn(BaseModel):
    """리뷰 등록 입력"""
    movie_id : int
    author   : constr(min_length=1)
    content  : constr(min_length=3)


class ReviewOut(ReviewIn):
    """리뷰 조회 응답 (추론 결과 포함)"""
    id               : int
    created_at       : datetime
    sentiment_label  : str
    sentiment_score  : float
    star             : Optional[int] = None
    star_conf        : Optional[float] = None
    soft_star        : Optional[float] = None


class ReviewWithTitle(ReviewOut):
    """리뷰 조회 응답 + 영화 제목(조인 결과)"""
    movie_title: str


class BaseResponse(BaseModel):
    """공통 베이스 응답"""
    success: bool
    message: str


class ErrorResponse(BaseModel):
    """에러 응답 포맷統一"""
    success: bool = False
    error: str
    error_code: int
    details: Optional[dict] = None


class SuccessMovie(BaseResponse):
    success: bool = True
    data: MovieOut


class SuccessMovies(BaseResponse):
    success: bool = True
    data: List[MovieOut]


class SuccessReview(BaseResponse):
    success: bool = True
    data: ReviewOut


class RatingOut(BaseModel):
    """영화별 평점 요약 (실시간 집계)"""
    movie_id: int
    count: int
    avg_star: Optional[float] = None
    avg_soft_star: Optional[float] = None
    avg_star_conf_weighted: Optional[float] = None


class SuccessRating(BaseResponse):
    success: bool = True
    data: RatingOut


# ================================== 앱/미들웨어/예외 ==================================

app = FastAPI(title="Movie API")

# 브라우저에서 직접 호출할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 운영에서는 도메인 화이트리스트 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 유효성 에러 → 일관된 포맷으로 변환
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="ValidationError", error_code=422, details={"errors": exc.errors()}
        ).model_dump()
    )

# FastAPI/Starlette HTTP 예외 → 일관된 포맷으로 변환
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail) if isinstance(exc.detail, str) else "HTTPError",
            error_code=exc.status_code
        ).model_dump()
    )


# ================================== 인메모리 "DB" ==================================
# 실제 DB 대신 간단한 리스트/시퀀스로 구현 (테스트/미션용)
# - 서버 재시작 시 데이터가 초기화됨
# - 보고서에서는 ERD/DDL 제안 스키마 참조
movie_id_seq = itertools.count(1)
movies_db: List[MovieOut] = []

review_id_seq = itertools.count(1)
reviews_db: List[ReviewOut] = []


# ================================== 엔드포인트 ==================================

@app.get("/", response_model=BaseResponse)
def hello():
    """헬로 엔드포인트: 서버 구동 확인"""
    return BaseResponse(success=True, message="영화 API에 오신 것을 환영합니다!")


@app.get("/health", response_model=BaseResponse)
def health():
    """헬스체크: 모델/토크나이저 준비 상태 확인 (세션 I/O 접근 테스트 포함)"""
    try:
        assert _sess is not None and _tok is not None
        _ = [i.name for i in _sess.get_inputs()]
        return BaseResponse(success=True, message=f"ok(model=stars:{STAR_MODEL_DIR})")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not ready: {e}")


# ----------------------------- 영화 -----------------------------

@app.post("/movie", response_model=SuccessMovie, status_code=status.HTTP_201_CREATED)
def add_movie(movie: MovieIn):
    """
    영화 등록
    - 제목 중복 방지: 공백/대소문자 무시하여 비교
    """
    norm_new = movie.title.strip().lower()
    if any(m.title.strip().lower() == norm_new for m in movies_db):
        raise HTTPException(status_code=409, detail="이미 같은 제목의 영화가 존재합니다.")
    new = MovieOut(id=next(movie_id_seq), **movie.model_dump())
    movies_db.append(new)
    return SuccessMovie(message=f"영화 '{new.title}' 추가 완료", data=new)


@app.get("/movie", response_model=SuccessMovies)
def list_movies(
    genre: Optional[Genre] = None,
    q: Optional[str] = None,
    director: Optional[str] = None,
    release_date: Optional[date] = None,
    release_date_from: Optional[date] = None,
    release_date_to: Optional[date] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    영화 목록 조회(필터/페이징)
    - q         : 제목 부분일치
    - director  : 감독 이름 부분일치
    - release_* : 날짜 단건/범위 필터
    """
    items = movies_db
    if genre:
        items = [m for m in items if genre in m.genre]
    if q:
        ql = q.lower()
        items = [m for m in items if ql in m.title.lower()]
    if director:
        dl = director.lower()
        items = [m for m in items if dl in m.director.lower()]
    if release_date:
        items = [m for m in items if m.release_date == release_date]
    if release_date_from:
        items = [m for m in items if m.release_date >= release_date_from]
    if release_date_to:
        items = [m for m in items if m.release_date <= release_date_to]

    sliced = items[offset: offset + limit]
    return SuccessMovies(message="ok", data=sliced)


@app.delete("/movie", status_code=status.HTTP_204_NO_CONTENT)
def clear_movies():
    """전체 초기화 (영화/리뷰 모두 삭제)"""
    movies_db.clear()
    reviews_db.clear()
    return


@app.get("/movie/{movie_id}", response_model=SuccessMovie)
def get_movie(movie_id: int):
    """영화 상세 조회"""
    m = next((m for m in movies_db if m.id == movie_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="영화를 찾을 수 없습니다.")
    return SuccessMovie(message="ok", data=m)


@app.delete("/movie/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_movie(movie_id: int):
    """영화 삭제 (+해당 영화의 리뷰 일괄 삭제)"""
    idx = next((i for i, m in enumerate(movies_db) if m.id == movie_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="영화를 찾을 수 없습니다.")
    del movies_db[idx]
    global reviews_db
    reviews_db = [r for r in reviews_db if r.movie_id != movie_id]
    return


# ----------------------------- 리뷰 -----------------------------

@app.post("/review", response_model=SuccessReview, status_code=status.HTTP_201_CREATED)
def create_review(review: ReviewIn):
    """
    리뷰 등록 + ONNX 추론
    - 입력 텍스트에 대해 감성/별점 지표를 계산하여 함께 저장
    """
    m = next((m for m in movies_db if m.id == review.movie_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="영화를 찾을 수 없습니다.")

    met = infer_review_metrics(review.content)
    new = ReviewOut(
        id=next(review_id_seq),
        movie_id=review.movie_id,
        author=review.author,
        content=review.content,
        created_at=datetime.now(timezone.utc),
        sentiment_label=met["sentiment_label"],
        sentiment_score=met["sentiment_score"],
        star=met.get("star"),
        star_conf=met.get("star_conf"),
        soft_star=met.get("soft_star"),
    )
    reviews_db.append(new)
    return SuccessReview(message="리뷰 등록 완료", data=new)


@app.get("/review", response_model=List[ReviewWithTitle])
def list_reviews(
    q_title: Optional[str] = Query(default=None, description="영화 제목 부분일치"),
    q_content: Optional[str] = Query(default=None, description="리뷰 내용 부분일치"),
    q_author: Optional[str] = Query(default=None, description="작성자 부분일치"),

    score_min: Optional[float] = Query(default=None, ge=-1.0, le=1.0),
    score_max: Optional[float] = Query(default=None, ge=-1.0, le=1.0),

    star_eq:  Optional[int]   = Query(default=None, ge=1, le=5, description="별점 정확히 일치"),
    star_min: Optional[int]   = Query(default=None, ge=1, le=5, description="별점 최소"),
    star_max: Optional[int]   = Query(default=None, ge=1, le=5, description="별점 최대"),

    soft_star_min: Optional[float] = Query(default=None, ge=1.0, le=5.0),
    soft_star_max: Optional[float] = Query(default=None, ge=1.0, le=5.0),

    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    리뷰 목록 조회(검색/필터/페이징)
    - 제목/내용/작성자 부분일치
    - 감성 점수 범위, 별점(정확/범위), soft_star 범위 필터
    - 최신 작성 순 정렬
    """
    title_by_id = {m.id: m.title for m in movies_db}
    items: List[ReviewOut] = reviews_db

    if q_title:
        qt = q_title.lower()
        items = [r for r in items if qt in title_by_id.get(r.movie_id, "").lower()]
    if q_content:
        qc = q_content.lower()
        items = [r for r in items if qc in r.content.lower()]
    if q_author:
        qa = q_author.lower()
        items = [r for r in items if qa in r.author.lower()]

    if score_min is not None:
        items = [r for r in items if r.sentiment_score >= score_min]
    if score_max is not None:
        items = [r for r in items if r.sentiment_score <= score_max]

    if star_eq is not None:
        items = [r for r in items if r.star is not None and r.star == star_eq]
    else:
        if star_min is not None:
            items = [r for r in items if r.star is not None and r.star >= star_min]
        if star_max is not None:
            items = [r for r in items if r.star is not None and r.star <= star_max]

    if soft_star_min is not None:
        items = [r for r in items if r.soft_star is not None and r.soft_star >= soft_star_min]
    if soft_star_max is not None:
        items = [r for r in items if r.soft_star is not None and r.soft_star <= soft_star_max]

    items = sorted(items, key=lambda r: r.created_at, reverse=True)
    sliced = items[offset: offset + limit]

    return [
        ReviewWithTitle(**r.model_dump(), movie_title=title_by_id.get(r.movie_id, ""))
        for r in sliced
    ]


@app.delete("/review/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_review(review_id: int):
    """리뷰 단건 삭제"""
    idx = next((i for i, r in enumerate(reviews_db) if r.id == review_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="리뷰를 찾을 수 없습니다.")
    del reviews_db[idx]
    return


# ----------------------------- 평균 평점 -----------------------------

@app.get("/rating/average", response_model=SuccessRating)
def rating_average(movie_id: int):
    """
    평균 평점 집계 (실시간)
    - avg_star                : 단순 평균(리뷰별 star의 평균)
    - avg_soft_star           : soft_star(확률 가중 평균 별점)의 평균
    - avg_star_conf_weighted  : star_conf(확률)를 가중치로 사용한 가중 평균
        * 신뢰도(conf)가 높은 별점의 영향력을 더 크게 반영
    """
    rs = [r for r in reviews_db if r.movie_id == movie_id and r.star is not None]
    n = len(rs)
    if n == 0:
        return SuccessRating(
            message="no ratings",
            data=RatingOut(movie_id=movie_id, count=0)
        )

    avg_star = sum(r.star for r in rs if r.star is not None) / n
    soft_vals = [r.soft_star for r in rs if r.soft_star is not None]
    avg_soft = sum(soft_vals)/len(soft_vals) if soft_vals else None

    conf_vals = [r.star_conf for r in rs if r.star_conf is not None]
    if conf_vals and all(v is not None for v in conf_vals):
        wsum = sum(conf_vals) or 1.0
        avg_conf_w = sum((r.star or avg_star) * (r.star_conf or 0.0) for r in rs) / wsum
    else:
        avg_conf_w = None

    return SuccessRating(
        message="ok",
        data=RatingOut(
            movie_id=movie_id,
            count=n,
            avg_star=round(avg_star, 3),
            avg_soft_star=round(avg_soft, 3) if avg_soft is not None else None,
            avg_star_conf_weighted=round(avg_conf_w, 3) if avg_conf_w is not None else None
        )
    )