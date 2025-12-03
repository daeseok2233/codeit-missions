# Python 3.10이 설치된 경량 버전
FROM python:3.10-slim

# 환경 변수 설정
# PIP_DISABLE_PIP_VERSION_CHECK=1 → pip 실행할 때 버전 확인 메시지 안 뜨게 함.
# PYTHONDONTWRITEBYTECODE=1 → .pyc (바이트코드 캐시) 파일 생성하지 않음 → 컨테이너 내부 불필요 파일 줄임.
# PYTHONUNBUFFERED=1 → 로그가 버퍼링 없이 바로 출력되게 함 → 실시간 로그 확인 가능.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \ 
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /app

# 라이브러리 설치
# --no-cache-dir 옵션은 설치 후 캐시를 남기지 않아서 이미지 용량을 줄여줌.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 및 데이터 복사
COPY train.py /app/train.py
COPY data /data

# 기본 실행 명령 (CMD)
# 환경이 제대로 세팅됐는지 확인용 출력 명령
CMD ["bash", "-lc", "python -V && pip -V"]