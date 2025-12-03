FROM python:3.11-slim

# onnxruntime이 필요로 하는 C/C++ 런타임 라이브러리(libgomp1, libstdc++6, libatomic1)만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 libatomic1 \
    libunwind8 libunwind8-dev \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 설치
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY . /app

# 모델 이미지에 포함
COPY models/mnist-12.onnx /app/models/mnist-12.onnx

ENV MODEL_PATH=/app/models/mnist-12.onnx

EXPOSE 8501
CMD ["streamlit","run", "run.py", "--server.address=0.0.0.0", "--server.port=8501"]

