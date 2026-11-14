# CPU image for linux/amd64. GPU / MPS were not tested in the environment that packaged this repo.
# Demucs 4.1 depends on sphn, which publishes a Linux wheel for x86_64 only.
# For NVIDIA GPUs, prefer an official CUDA runtime base, install Python 3.11+,
# ffmpeg, and the cu132 torch wheel; then run with: docker run --gpus all ...
FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1 \
    UNBED_DATA_DIR=/data
VOLUME ["/data"]

# docker run --rm -v $PWD/data:/data IMAGE https://www.youtube.com/watch?v=... --out /data
# docker run --rm -v $PWD/data:/data -v $PWD/clip.mp4:/clip.mp4 IMAGE /clip.mp4 --out /data
ENTRYPOINT ["python", "-u", "-m", "unbed"]
