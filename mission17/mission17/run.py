import os, io, time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageFilter
import onnxruntime as ort

# -----------------------------
# Streamlit 기본 설정
# -----------------------------
st.set_page_config(page_title="MISSION 17 - MNIST ONNX", layout="wide")
st.title("MISSION 17: MNIST 손글씨 인식 (ONNX)")

# 세션 상태 초기화 (갤러리 저장용)
if "gallery" not in st.session_state:
    st.session_state.gallery = []

# -----------------------------
# ① 모델 로드 (캐싱 처리)
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/mnist-12.onnx")

@st.cache_resource(show_spinner=True)
def get_session(model_path: str, prefer_cuda: bool = True):
    """ONNX Runtime 세션을 로드하고 Execution Provider(GPU/CPU)를 선택"""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if prefer_cuda else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    used = sess.get_providers()[0] if sess.get_providers() else "CPUExecutionProvider"
    return sess, used

# 사이드바에서 모델 옵션 선택
with st.sidebar:
    st.header("모델/실행 옵션")
    prefer_cuda = st.toggle("CUDA 우선 사용 (가능하면 GPU)", value=False)  # CPU 환경이면 False 권장
    st.caption(f"MODEL_PATH = `{MODEL_PATH}`")
    try:
        sess, provider = get_session(MODEL_PATH, prefer_cuda)
        st.success(f"Execution Provider: **{provider}**")
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

# -----------------------------
# ② 전처리 함수 정의
# -----------------------------
def preprocess_canvas_to_mnist(img_rgba: np.ndarray) -> Image.Image:
    """캔버스 RGBA 이미지를 → 흑백 변환 + 가우시안 블러 + 대비 자동 조정"""
    pil_gray = Image.fromarray(img_rgba.astype(np.uint8), "RGBA").convert("L")
    pil_gray = pil_gray.filter(ImageFilter.GaussianBlur(radius=0.5))
    pil_gray = ImageOps.autocontrast(pil_gray)
    return pil_gray

def autocrop_pil(pil_gray: Image.Image, pad: int = 2):
    """글자가 포함된 영역을 찾아 잘라내고 여백(pad) 추가"""
    inv = ImageOps.invert(pil_gray)
    bbox = inv.getbbox()
    if bbox is None:
        return pil_gray, (0, 0, pil_gray.width, pil_gray.height)
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(pil_gray.width,  x2 + pad)
    y2 = min(pil_gray.height, y2 + pad)
    return pil_gray.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)

def to_mnist_28x28(pil_gray: Image.Image, invert: bool = True, content_size: int = 22, pad_bg: int = 255):
    """28x28 MNIST 입력 크기로 변환 (가운데 정렬, 흰 배경, 흑색 글씨)"""
    w, h = pil_gray.size
    if max(w, h) == 0:  # 빈 이미지 처리
        canvas = Image.new("L", (28, 28), pad_bg)
        arr = np.array(canvas, dtype=np.float32)/255.0
        return canvas, arr[np.newaxis, np.newaxis, :, :]
    # 비율 유지하며 content_size에 맞게 리사이즈
    scale = content_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = pil_gray.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (28, 28), pad_bg)
    canvas.paste(resized, ((28 - new_w)//2, (28 - new_h)//2))
    if invert:  # 글씨 색 반전
        canvas = ImageOps.invert(canvas)
    arr = np.array(canvas, dtype=np.float32)/255.0
    return canvas, arr[np.newaxis, np.newaxis, :, :]

def _softmax(x: np.ndarray) -> np.ndarray:
    """소프트맥스 함수"""
    x = x.astype(np.float32); x -= x.max()
    e = np.exp(x); return e/(e.sum()+1e-9)

def predict_mnist(session: ort.InferenceSession, x: np.ndarray):
    """ONNX 세션으로 MNIST 추론 수행 → 예측 숫자, 확률 벡터 반환"""
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    logits = session.run([out], {inp: x.astype(np.float32)})[0].squeeze()
    if logits.ndim == 2: logits = logits[0]
    probs = _softmax(logits)
    return int(np.argmax(probs)), probs

def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    """PIL 이미지를 PNG 바이트로 변환 (갤러리 저장용)"""
    buf = io.BytesIO(); pil_img.save(buf, format="PNG"); return buf.getvalue()

# -----------------------------
# ③ 입력 캔버스 영역
# -----------------------------
st.subheader("① 입력 캔버스")
c1, c2, c3 = st.columns([1,1,1])
with c1: stroke_width = st.slider("선 굵기", 1, 30, 10)
with c2: stroke_color = st.color_picker("선 색상", "#000000")
with c3:
    drawing_mode = st.selectbox("그리기 모드", ["freedraw","line","rect","circle","polygon","transform"], index=0)

# Streamlit 그리기 캔버스 위젯
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#FFFFFF",
    update_streamlit=True,
    height=320, width=640,
    drawing_mode=drawing_mode,
    display_toolbar=True,
    key="canvas",
)
st.divider()

# -----------------------------
# ④ 전처리/예측 결과 출력
# -----------------------------
st.subheader("② 전처리 이미지 & ③ 예측 확률(막대 차트)")
col_l, col_r = st.columns([1,1])

# 왼쪽: 전처리 과정 시각화
with col_l:
    st.markdown("**전처리 미리보기**")
    if canvas_result.image_data is None:
        st.info("캔버스 초기화 중…")
    else:
        data = canvas_result.json_data
        if (data is None) or (len(data.get("objects", [])) == 0):
            st.info("아직 그린 내용이 없습니다.")
        else:
            # 캔버스 → 전처리 → 28x28 변환
            gray = preprocess_canvas_to_mnist(canvas_result.image_data)
            cropped, bbox = autocrop_pil(gray, pad=2)
            resized, x_input = to_mnist_28x28(cropped, invert=True)
            st.image(gray, caption="Grayscale", use_container_width=True)
            st.image(cropped, caption=f"Cropped (bbox={bbox})", use_container_width=True)
            st.image(resized, caption="Resized 28×28")

# 오른쪽: 예측 확률 & 결과
with col_r:
    st.markdown("**예측 확률 (0~9)**")
    if st.button("추론 실행", type="primary", use_container_width=True):
        data = canvas_result.json_data
        if (data is None) or (len(data.get("objects", [])) == 0):
            st.warning("숫자를 먼저 그려주세요.")
        else:
            # 추론 실행
            gray = preprocess_canvas_to_mnist(canvas_result.image_data)
            cropped, _ = autocrop_pil(gray, pad=2)
            resized, x_input = to_mnist_28x28(cropped, invert=True)
            t0 = time.time()
            try:
                pred, probs = predict_mnist(sess, x_input)
                st.success(f"예측: **{pred}**  |  EP: **{provider}**  |  {(time.time()-t0)*1000:.1f} ms")
                df = pd.DataFrame({"digit": list(range(10)), "prob": probs.astype(float)}).set_index("digit")
                st.bar_chart(df, height=280)
                # Top-3 확률 표시
                top3 = df["prob"].nlargest(3)
                st.write("**Top-3:**", {int(k): float(v) for k,v in top3.items()})
                # 갤러리에 저장
                st.session_state.gallery.append({
                    "png": pil_to_png_bytes(resized),
                    "pred": int(pred),
                    "probs": [float(p) for p in probs.tolist()],
                    "ts": time.time(),
                })
            except Exception as e:
                st.error(f"추론 오류: {e}")

st.divider()

# -----------------------------
# ⑤ 이미지 저장소 (갤러리)
# -----------------------------
st.subheader("④ 이미지 저장소 (예측 라벨/확률 포함)")
left, _, _ = st.columns([1,6,1])
with left:
    if st.button("갤러리 비우기", type="secondary"):
        st.session_state.gallery = []

# 저장된 추론 결과 출력
if len(st.session_state.gallery) == 0:
    st.info("추론 실행 시 자동 저장됩니다.")
else:
    gallery = sorted(st.session_state.gallery, key=lambda x: x["ts"], reverse=True)
    cols = st.columns(3)
    for i, item in enumerate(gallery):
        with cols[i % 3]:
            st.image(item["png"], caption=f"Pred: {item['pred']}", use_container_width=True)
            probs = np.array(item["probs"])
            top3_idx = probs.argsort()[-3:][::-1]
            summary = ", ".join([f"{int(k)}:{probs[k]:.2f}" for k in top3_idx])
            st.caption(f"Top-3 → {summary}")