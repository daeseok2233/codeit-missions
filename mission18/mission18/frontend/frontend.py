import os
import requests
from requests.adapters import HTTPAdapter, Retry
import streamlit as st
from datetime import date
from urllib.parse import urlparse

# =============================================================================
# 프로젝트: 영화 관리 + 리뷰 감성/별점 추론 프론트엔드 (Streamlit)
# 개요:
#  - FastAPI 백엔드(API_BASE)와 통신하여 영화/리뷰 CRUD 및 분석 결과 표시
#  - 네트워크 안정성: 세션 재시도/백오프 + 요청 타임아웃 적용
#  - UI 성능: st.cache_data로 리스트/감정 분석 API 결과를 짧게 캐싱
# =============================================================================

# -----------------------------
# 기본 설정 (환경 변수 + Streamlit 페이지)
# -----------------------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000") 
TIMEOUT_S = 12  # 모든 요청 공통 타임아웃(초)

st.set_page_config(page_title="스프린트 미션18", layout="wide")
st.title("🎥 무비 매니저")

# 선택된 영화 상태 (목록 <-> 상세 전환용 화면 스위치 플래그)
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None


# -----------------------------
# 공통 HTTP 세션 (재시도/백오프 설정)
# -----------------------------
def get_session() -> requests.Session:
    """
    HTTP 오류(429/5xx)나 순간 네트워크 이슈를 흡수하기 위한 공용 세션.
    - Retry: 지수 백오프(0.3s, 0.6s, 1.2s)로 최대 3회 재시도
    - 모든 프로토콜(http/https)에 동일 정책 적용
    """
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "POST", "DELETE", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


SESSION = get_session()  # 전역 재사용(연결 재사용 + 정책 일괄 적용)


# -----------------------------
# 유틸 함수
# -----------------------------
def url_points_to_image(url: str) -> bool:
    """
    URL이 실제 '이미지'를 가리키는지 보수적으로 검사.
    - HEAD로 Content-Type/Length 확인 → 모호하면 GET(stream)로 재확인
    - 확장자 힌트(likely)도 보조 신호로 사용(완벽한 보장은 아님)
    """
    if not url:
        return False
    try:
        ext = (urlparse(url).path or "").lower()
        likely = ext.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"))

        r = SESSION.head(url, timeout=6, allow_redirects=True)
        ctype = (r.headers.get("Content-Type") or "").lower()
        clen = int(r.headers.get("Content-Length") or 0)
        if r.ok and ("image" in ctype) and clen != 0:
            return True

        # HEAD로 판별 어려우면 실제 GET으로 확인(일부 서버는 HEAD 미지원/부정확)
        r = SESSION.get(url, stream=True, timeout=TIMEOUT_S)
        ctype = (r.headers.get("Content-Type") or "").lower()
        return (r.ok and "image" in ctype) or likely
    except Exception:
        return False


def clear_cache():
    """Streamlit 캐시 초기화 (검색 조건 변경/등록/삭제 이후 UI 갱신용)."""
    try:
        fetch_movies_cached.clear()
        api_get_reviews.clear()
        api_get_avg_star.clear()
        api_search_reviews.clear()
    except Exception:
        st.cache_data.clear()


def render_stars(avg: float | None, max_stars: int = 5) -> str:
    """평균 평점을 문자열(⭐ x/5)로 반환."""
    if avg is None:
        return "평점 없음"
    return f"⭐ {avg:.2f}/{max_stars}"


def get_sentiment_fields(rv: dict):
    """
    백엔드 응답 필드명이 다를 가능성(레거시/실습편차)을 흡수하는 헬퍼.
    - sentiment_label / sentiment_score 우선, label/score는 폴백
    - None만 폴백 대상으로 취급(0.0은 유효값)
    """
    def pick(primary: str, fallback: str):
        return rv[primary] if (primary in rv and rv[primary] is not None) else rv.get(fallback)

    lbl = pick("sentiment_label", "label")
    sc  = pick("sentiment_score", "score")
    return lbl, sc

# -----------------------------
# API 래퍼 (캐시 포함)
# -----------------------------
@st.cache_data(ttl=5)
def fetch_movies_cached(params: dict | None = None):
    """
    영화 목록 조회(캐시).
    - 캐시 키 안정화를 위해 params를 frozenset으로 변환
    - 백엔드에서 {"data": [...]} 형태를 우선 처리
    """
    key = frozenset((params or {}).items())
    resp = SESSION.get(f"{API_BASE}/movie", params=dict(key), timeout=TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def api_post_movie(payload: dict):
    """영화 등록(캐시 무효화는 호출측에서 clear_cache())."""
    return SESSION.post(f"{API_BASE}/movie", json=payload, timeout=TIMEOUT_S)


def api_delete_all():
    """영화/리뷰 전체 삭제(초기화)."""
    return SESSION.delete(f"{API_BASE}/movie", timeout=TIMEOUT_S)


def api_get_movie(movie_id: int):
    """영화 단건 조회."""
    r = SESSION.get(f"{API_BASE}/movie/{movie_id}", timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def api_delete_movie(movie_id: int):
    """영화 단건 삭제."""
    return SESSION.delete(f"{API_BASE}/movie/{movie_id}", timeout=TIMEOUT_S)


def api_post_review(payload: dict):
    """리뷰 등록(백엔드에서 감성/별점 추론 후 결과 반환)."""
    return SESSION.post(f"{API_BASE}/review", json=payload, timeout=TIMEOUT_S)


@st.cache_data(ttl=5)
def api_get_reviews(movie_id: int, limit: int = 10):
    """특정 영화의 최신 리뷰 n개 조회(캐시)."""
    try:
        r = SESSION.get(
            f"{API_BASE}/review",
            params={"movie_id": movie_id, "limit": limit},
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return []


@st.cache_data(ttl=5)
def api_search_reviews(params: dict):
    """리뷰 검색(제목/내용/작성자/별점 필터; 캐시)."""
    r = SESSION.get(f"{API_BASE}/review", params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def api_delete_review(review_id: int):
    """리뷰 단건 삭제."""
    return SESSION.delete(f"{API_BASE}/review/{review_id}", timeout=TIMEOUT_S)


@st.cache_data(ttl=5)
def api_get_avg_star(movie_id: int):
    """특정 영화의 평균 별점/가중평균 별점 조회(캐시)."""
    r = SESSION.get(f"{API_BASE}/rating/average", params={"movie_id": movie_id}, timeout=TIMEOUT_S)
    r.raise_for_status()
    payload = r.json()
    return payload.get("data", payload)


# -----------------------------
# 렌더링 함수 (재사용 + 가독성)
# -----------------------------
def render_movie_card(m: dict, bordered: bool = False, unique_suffix: str = ""):
    """
    영화 카드 한 장을 렌더링.
    - Review/삭제 버튼을 항상 노출 (포스터가 없어도)
    - 삭제 시 캐시 비우고 목록 갱신
    """
    with st.container(border=bordered):
        st.write(m.get("title", ""))

        # 포스터
        url = m.get("poster_url") or ""
        if url:
            st.image(url, use_container_width=True)
        else:
            st.caption("❌ 포스터 없음")

        # 버튼들 (Review / 삭제)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Review", key=f"sel_{m.get('id')}_{unique_suffix}", use_container_width=True):
                st.session_state.selected_movie_id = int(m["id"])
                st.rerun()

        with c2:
            if st.button("삭제", key=f"del_{m.get('id')}_{unique_suffix}", use_container_width=True):
                try:
                    resp = api_delete_movie(int(m["id"]))
                    resp.raise_for_status()
                    st.success("영화가 삭제되었습니다.")
                    clear_cache()
                    # 혹시 상세 화면에서 삭제될 수 있으니 선택 해제
                    if st.session_state.get("selected_movie_id") == m["id"]:
                        st.session_state.selected_movie_id = None
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"삭제 실패: {e}")

        # 메타/평점
        try:
            avg = api_get_avg_star(int(m["id"]))
            if avg and avg.get("count", 0) > 0:
                st.caption(render_stars(avg.get("avg_star")) + f" · 리뷰 {avg['count']}개")
        except requests.RequestException:
            pass

        st.caption(
            f"📅 {m.get('release_date','')} · 🎬 {m.get('director','')} · "
            f"🏷️ {', '.join(m.get('genre') or [])}"
        )


def render_review_item(rv: dict):
    """
    리뷰 아이템 한 건 렌더링.
    - 백엔드 필드 편차를 get_sentiment_fields로 흡수
    - 확신도(conf)는 존재할 때만 숫자 포맷 표시
    """
    with st.container(border=True):
        created = rv.get("created_at", "")
        author = rv.get("author", "")
        content = rv.get("content", "")
        star = rv.get("star")
        star_conf = rv.get("star_conf")
        lbl, sc = get_sentiment_fields(rv)

        st.caption(f"🕒 {created}")
        st.write(f"**작성자**: {author}")
        st.write(content)
        if star is not None:
            st.caption(f"⭐ {star}/5" + (f" · conf {star_conf:.2f}" if star_conf is not None else ""))
        if (lbl is not None) or (sc is not None):
            st.caption(f"감성: {lbl} · 점수: {sc}")


# -----------------------------
# 사이드바 (컨트롤 + 폼들)
# -----------------------------
with st.sidebar:
    # 전역 컨트롤 (새로고침/전체삭제)
    st.header("⚙️ 컨트롤")
    colA, colB = st.columns(2)
    with colA:
        if st.button("새로고침", use_container_width=True):
            clear_cache()
            st.rerun()
    with colB:
        if st.button("전체 데이터 삭제", type="primary", use_container_width=True):
            try:
                r = api_delete_all()
                r.raise_for_status()
                clear_cache()
                st.success("서버 DB가 초기화되었습니다.")
                st.session_state.selected_movie_id = None
                st.rerun()
            except requests.RequestException as e:
                # 서버에서 JSON 에러 본문을 주는 경우 표시
                try:
                    st.error(r.json())
                except Exception:
                    st.error(f"초기화 실패: {e}")

    st.divider()

    # ----------------- 영화 추가 폼 -----------------
    with st.expander("🎬 영화 추가", expanded=False):
        with st.form("movie_form", clear_on_submit=False):
            # 필수값들은 폼 검증에서 누락 시 안내
            title = st.text_input("제목 *", placeholder="예: 괴물")
            release_date = st.date_input("개봉일 *", value=date.today(), min_value=date(1900, 1, 1), max_value=date(2100, 12, 31), help="달력에서 날짜를 선택하세요 (YYYY-MM-DD).")
            director = st.text_input("감독 *", placeholder="예: 봉준호")
            genre = st.multiselect(
                "장르 *",
                ["액션","코미디","드라마","SF","호러","로맨스","스릴러","모험","범죄","미스터리",
                 "다큐멘터리","애니메이션","판타지","가족","음악","뮤지컬","전쟁","스포츠","재난"],
                placeholder="예: 드라마, 가족",
                help="여러 개를 자유롭게 선택할 수 있어요.",
            )
            poster_url = st.text_input("포스터 URL *", placeholder="예: https://...", help="이미지 주소(URL)만 입력해 주세요.")

            submitted = st.form_submit_button("추가하기", use_container_width=True)
            if submitted:
                missing = []
                if not title.strip(): missing.append("제목")
                if not release_date: missing.append("개봉일")
                if not director.strip(): missing.append("감독")
                if not genre: missing.append("장르")
                if not poster_url.strip(): missing.append("포스터 URL")

                if missing:
                    st.error(f"필수 항목을 모두 입력하세요: {', '.join(missing)}")
                else:
                    # 이미지 URL 유효성(가벼운 네트워크 검증)
                    if not url_points_to_image(poster_url.strip()):
                        st.error("포스터 URL에 문제가 있습니다. 유효한 이미지 주소를 입력하세요.")
                    else:
                        payload = {
                            "title": title.strip(),
                            "release_date": str(release_date),
                            "director": director.strip(),
                            "genre": genre,
                            "poster_url": poster_url.strip(),
                        }
                        try:
                            r = api_post_movie(payload)
                            if r.status_code in (200, 201):
                                st.success("영화가 추가되었습니다.")
                                clear_cache()
                                st.rerun()
                            else:
                                try:
                                    st.error(r.json())
                                except Exception:
                                    st.error(f"추가 실패: HTTP {r.status_code}")
                        except requests.RequestException as e:
                            st.error(f"추가 실패: {e}")

    # ----------------- 검색/필터/삭제 -----------------
    with st.expander("🔎 검색/필터/삭제", expanded=False):
        # 사이드바 상단의 현재 필터 상태 표시를 위해 세션에서 복원
        GENRES = ["전체","액션","코미디","드라마","SF","호러","로맨스","스릴러","모험",
                  "범죄","미스터리","다큐멘터리","애니메이션","판타지","가족","음악","뮤지컬","전쟁","스포츠","재난"]

        cur = st.session_state.get("search_params", {}) or {}
        cur_title    = cur.get("q", "")
        cur_director = cur.get("director", "")
        cur_genre    = cur.get("genre", "전체")
        cur_limit    = int(cur.get("limit", 100))

        def _to_date(v):
            """세션에 저장된 ISO 문자열을 date로 복원."""
            from datetime import date as _d
            try:
                return _d.fromisoformat(v) if isinstance(v, str) else v
            except Exception:
                return None

        rd_from_saved = _to_date(cur.get("release_date_from"))
        rd_to_saved   = _to_date(cur.get("release_date_to"))

        try:
            genre_index = GENRES.index(cur_genre) if cur_genre in GENRES else 0
        except ValueError:
            genre_index = 0

        if cur:
            ran = "-"
            if rd_from_saved or rd_to_saved:
                ran = f"{rd_from_saved or '...'} ~ {rd_to_saved or '...'}"
            st.caption(
                f"현재 필터 → 장르: {cur_genre or '전체'} | 제목: {cur_title or '-'} | "
                f"감독: {cur_director or '-'} | 개봉일: {ran} | 표시 개수: {cur_limit}"
            )

        use_range = st.toggle("개봉일 범위 사용", value=bool(rd_from_saved or rd_to_saved))

        with st.form("search_form", clear_on_submit=False):
            # 제목/감독/장르/날짜범위/표시개수 구성
            c1, c2 = st.columns(2)
            with c1:
                title_q = st.text_input("제목", value=cur_title, placeholder="예: 괴물")
            with c2:
                director_q = st.text_input("감독", value=cur_director, placeholder="예: 봉준호")

            genre_opt = st.selectbox("장르", GENRES, index=genre_index)

            rd_from = rd_to = None
            if use_range:
                from datetime import date as _d
                default_start = rd_from_saved or _d(2000, 1, 1)
                default_end   = rd_to_saved   or _d.today()
                picked = st.date_input("개봉일 범위", value=(default_start, default_end), min_value=_d(1900, 1, 1), max_value=_d(2100, 12, 31))
                if isinstance(picked, tuple) and len(picked) == 2:
                    rd_from, rd_to = picked
                else:
                    rd_from = rd_to = picked

            limit = st.number_input("표시 개수", min_value=1, max_value=500, value=cur_limit, step=10)
            applied = st.form_submit_button("적용", use_container_width=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            if applied:
                params = {}
                if title_q.strip():    params["q"] = title_q.strip()
                if director_q.strip(): params["director"] = director_q.strip()
                if genre_opt != "전체": params["genre"] = genre_opt
                params["limit"] = int(limit)

                if use_range and rd_from and rd_to:
                    if rd_from > rd_to:
                        st.error("개봉일 범위가 올바르지 않습니다. 시작일이 종료일보다 이후일 수 없습니다.")
                        st.stop()
                    params["release_date_from"] = str(rd_from)
                    params["release_date_to"]   = str(rd_to)

                # 검색 파라미터를 세션에 저장 → 메인 영역에서 재사용
                st.session_state["search_params"] = params
                clear_cache()
                st.rerun()

        with cc2:
            if st.button("초기화", use_container_width=True):
                st.session_state.pop("search_params", None)
                clear_cache()
                st.rerun()

    # ----------------- 리뷰 관리(검색+삭제) -----------------
    with st.expander("🧰 리뷰 관리", expanded=False):
        rs = st.session_state.get("review_search_params", {}) or {}
        c1, c2 = st.columns(2)
        with c1:
            q_title   = st.text_input("영화 제목", value=rs.get("q_title",""))
            q_author  = st.text_input("작성자", value=rs.get("q_author",""))
        with c2:
            q_content = st.text_input("리뷰 내용", value=rs.get("q_content",""))

        # 별점 필터: 정확/범위 모드 스위치(세션 복원)
        star_mode_default = rs.get("star_mode", "전체")
        mode = st.radio("별점 필터", ["전체", "정확히", "범위"],
                        index=["전체","정확히","범위"].index(star_mode_default),
                        horizontal=True, key="star_mode")

        star_eq_val = None
        star_min = int(rs.get("star_min", 1))
        star_max = int(rs.get("star_max", 5))

        if mode == "정확히":
            star_eq_val = st.select_slider("별점(정확히)", options=[1,2,3,4,5], value=int(rs.get("star_eq", 5)))
        elif mode == "범위":
            s1, s2 = st.columns(2)
            with s1:
                star_min = st.select_slider("별점 최소", options=[1,2,3,4,5], value=star_min)
            with s2:
                star_max = st.select_slider("별점 최대", options=[1,2,3,4,5], value=star_max)

        limit = st.number_input("표시 개수", min_value=1, max_value=200, value=int(rs.get("limit", 20)), step=5)

        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("검색 실행", use_container_width=True):
                params = {"limit": int(limit)}
                if q_title.strip():   params["q_title"]   = q_title.strip()
                if q_author.strip():  params["q_author"]  = q_author.strip()
                if q_content.strip(): params["q_content"] = q_content.strip()

                # 모드별 별점 파라미터 구성
                params["star_mode"] = mode
                if mode == "정확히" and star_eq_val is not None:
                    params["star_eq"] = int(star_eq_val)
                elif mode == "범위":
                    if int(star_min) > int(star_max):
                        st.error("별점 범위가 올바르지 않습니다. 최소가 최대보다 클 수 없습니다.")
                        st.stop()
                    params["star_min"] = int(star_min)
                    params["star_max"] = int(star_max)

                st.session_state["review_search_params"]  = params
                st.session_state.pop("review_search_results", None)  # 이전 결과 무효화
                st.rerun()

        # 검색 결과 렌더링
        if st.session_state.get("review_search_params"):
            st.divider()

            hdr_l, hdr_r = st.columns([7, 4])
            with hdr_l:
                st.subheader("🧾 리뷰 검색 결과")
            with hdr_r:
                if st.button("⬅️ 뒤로", key="review_back", use_container_width=True):
                    st.session_state.pop("review_search_params", None)
                    st.session_state.pop("review_search_results", None)
                    st.rerun()

            params = st.session_state["review_search_params"]
            try:
                results = st.session_state.get("review_search_results")
                if results is None:
                    with st.spinner("리뷰를 검색 중…"):
                        results = api_search_reviews(params)
                    st.session_state["review_search_results"] = results

                st.caption(f"총 {len(results)}건")
                if not results:
                    st.info("검색 결과가 없습니다.")
                else:
                    for rv in results:
                        render_review_item(rv)
                        # 삭제 이후 캐시/결과 초기화 → 재조회
                        if st.button("삭제", key=f"del_review_{rv['id']}", use_container_width=True):
                            try:
                                resp = api_delete_review(int(rv["id"]))
                                resp.raise_for_status()
                                st.success("리뷰가 삭제되었습니다.")
                                st.session_state.pop("review_search_results", None)
                                clear_cache()
                                st.rerun()
                            except requests.RequestException as e:
                                st.error(f"삭제 실패: {e}")
            except requests.RequestException as e:
                st.error(f"검색 실패: {e}")


# =============================================================================
# 메인 영역: 영화 목록 (전체 vs 검색 결과)
# =============================================================================
st.divider()
params = st.session_state.get("search_params", {}) or {}

def _is_filtered(p: dict) -> bool:
    """현재 검색 파라미터로 필터가 적용되었는지 여부."""
    return any([
        bool(p.get("q")),
        bool(p.get("director")),
        (p.get("genre") and p.get("genre") != "전체"),
        bool(p.get("release_date_from")),
        bool(p.get("release_date_to")),
    ])

is_filtered = _is_filtered(params)

# 목록 화면(선택된 영화가 없을 때만)
if not st.session_state.get("selected_movie_id"):
    st.subheader("🔎 검색 결과" if is_filtered else "📚 전체 영화")

    if is_filtered:
        # 적용된 필터 칩 표시(UX)
        chips = []
        if params.get("q"): chips.append(f"`제목:{params['q']}`")
        if params.get("director"): chips.append(f"`감독:{params['director']}`")
        if params.get("genre") and params["genre"] != "전체": chips.append(f"`장르:{params['genre']}`")
        if params.get("release_date_from") or params.get("release_date_to"):
            chips.append(f"`개봉일:{params.get('release_date_from','...')}~{params.get('release_date_to','...')}`")
        st.caption("적용된 필터: " + (" ".join(chips) if chips else "-"))
        col_reset, _ = st.columns([2, 6])
        with col_reset:
            if st.button("필터 초기화"):
                st.session_state.pop("search_params", None)
                clear_cache()
                st.rerun()

    # 데이터 불러오기 + 카드 렌더링
    try:
        with st.spinner("영화 목록을 불러오는 중…"):
            data = fetch_movies_cached(params)
        st.caption(f"총 {len(data)}건")
        if data:
            num_cols = 2 if is_filtered else 3
            cols = st.columns(num_cols, gap="large")
            for i, m in enumerate(data):
                with cols[i % num_cols]:
                    render_movie_card(m, bordered=is_filtered, unique_suffix=str(i))
        else:
            st.info("영화 데이터가 비었습니다.")
    except requests.RequestException as e:
        st.error(f"목록 조회 실패: {e}")


# =============================================================================
# 상세 화면: 선택된 영화 1개 + 리뷰 작성 + 최신 10개
# =============================================================================
if st.session_state.get("selected_movie_id"):
    st.divider()
    sel_id = st.session_state["selected_movie_id"]

    try:
        with st.spinner("영화 상세를 불러오는 중…"):
            mo = api_get_movie(sel_id)
        # 백엔드가 {"data": {...}} 또는 {...} 중 하나를 줄 수 있어 폴백 처리
        movie = mo["data"] if isinstance(mo, dict) and "data" in mo else mo
    except requests.RequestException as e:
        st.error(f"영화 조회 실패: {e}")
        movie = None

    if movie:
        c1, c2 = st.columns([6, 1])
        with c1:
            st.subheader(f"🎬 {movie.get('title','')} (ID: {movie.get('id')})")
            # 평균 평점(리뷰 수가 0이면 '없음')
            try:
                avg = api_get_avg_star(sel_id)
                if avg and avg.get("count", 0) > 0:
                    st.caption(f"{render_stars(avg.get('avg_star'))} · 리뷰 {avg['count']}개")
                else:
                    st.caption("⭐ 평균 평점: 없음")
            except requests.RequestException:
                st.caption("평균 평점을 불러오지 못했습니다.")
            st.caption(
                f"📅 {movie.get('release_date','')} · 🎬 {movie.get('director','')} · "
                f"🏷️ {', '.join(movie.get('genre') or [])}"
            )
        with c2:
            if st.button("⬅️ 목록으로", use_container_width=True):
                st.session_state.selected_movie_id = None
                clear_cache()
                st.rerun()

        colL, colR = st.columns([2, 3], gap="large")
        with colL:
            if movie.get("poster_url"):
                st.image(movie["poster_url"], use_container_width=True)
        with colR:
            st.markdown("### ✍️ 리뷰 작성")
            with st.form(f"review_form_{sel_id}", clear_on_submit=True):
                # 작성 직후 UI 피드백을 위해 clear_on_submit=True
                author = st.text_input("작성자 *", key=f"author_{sel_id}")
                content = st.text_area("리뷰 내용 *", key=f"content_{sel_id}", height=140, placeholder="감상평을 적어주세요.")
                submitted = st.form_submit_button("등록 및 분석(별점/감성)")
                if submitted:
                    miss = []
                    if not author.strip():  miss.append("작성자")
                    if not content.strip(): miss.append("리뷰 내용")
                    if miss:
                        st.error(f"필수 항목을 입력해주세요: {', '.join(miss)}")
                    else:
                        try:
                            r = api_post_review({"movie_id": int(sel_id), "author": author.strip(), "content": content.strip()})
                            if r.status_code in (200, 201):
                                res = r.json()
                                st.success("리뷰가 등록되었습니다.")
                                data = res.get("data", {})
                                # 등록 직후 백엔드 예측 결과를 바로 보여줌
                                star = data.get("star")
                                star_conf = data.get("star_conf")
                                if star is not None:
                                    st.info(f"예측 별점: ⭐ {star}/5" + (f" (conf {star_conf:.2f})" if star_conf is not None else ""))
                                lbl = data.get("sentiment_label")
                                sc  = data.get("sentiment_score")
                                if lbl is not None or sc is not None:
                                    st.caption(f"감성: {lbl} · 점수: {sc}")
                                clear_cache()
                                st.rerun()
                            else:
                                try:
                                    st.error(r.json())
                                except Exception:
                                    st.error(f"등록 실패: HTTP {r.status_code}")
                        except requests.RequestException as e:
                            st.error(f"등록 실패: {e}")

        # 최신 n개 리뷰 나열
        st.markdown("### 🧾 최신 리뷰 10개")
        try:
            with st.spinner("리뷰를 불러오는 중…"):
                reviews = api_get_reviews(sel_id, limit=10)
            if not reviews:
                st.info("아직 등록된 리뷰가 없습니다.")
            else:
                for rv in reviews:
                    render_review_item(rv)
        except requests.RequestException as e:
            st.error(f"리뷰 조회 실패: {e}")