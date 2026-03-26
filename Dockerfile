# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile – Priority Vehicle Detection API
# Stack : FastAPI + YOLOv8n (Ultralytics) + EasyOCR
# Image : python:3.10-slim
# Port  : 8000
# ─────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────┐
# │  STAGE 1 – dependency installer                                 │
# │  Pisahkan layer install dari layer copy code agar cache         │
# │  requirements.txt tidak invalidated setiap push kode.           │
# └─────────────────────────────────────────────────────────────────┘
FROM python:3.10-slim AS builder

WORKDIR /install

# Install build tools yang dibutuhkan untuk compile beberapa C-extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy hanya requirements.txt terlebih dahulu.
# Layer ini hanya akan di-rebuild jika requirements.txt berubah,
# bukan setiap kali source code diubah.
COPY requirements.txt .

# Install semua dependensi ke folder /install/packages
# --no-cache-dir  : hemat ruang disk, tidak simpan cache pip
# --prefix        : install ke folder custom (untuk multi-stage copy)
RUN pip install --no-cache-dir --prefix=/install/packages \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install/packages \
        -r requirements.txt


# ┌─────────────────────────────────────────────────────────────────┐
# │  STAGE 2 – runtime image (final, lebih kecil)                   │
# └─────────────────────────────────────────────────────────────────┘
FROM python:3.10-slim AS runtime

# Metadata image
LABEL maintainer="Priority Vehicle Detection Research"
LABEL description="FastAPI backend for priority vehicle detection using YOLOv8n + EasyOCR"
LABEL version="1.0.0"

# Variabel lingkungan runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=2200 \
    MODEL_PATH=model/best.pt \
    OCR_LANG=en \
    LOG_LEVEL=info

WORKDIR /app

# Install runtime library sistem (bukan build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        # curl untuk health-check
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packages hasil install dari builder stage
COPY --from=builder /install/packages /usr/local

# Copy source code project
COPY app/        ./app/
COPY scripts/    ./scripts/
COPY dataset/dataset.yaml ./dataset/dataset.yaml

# Buat folder model (best.pt di-mount via volume atau disalin saat deploy)
RUN mkdir -p model

# Buat non-root user untuk keamanan (best practice container)
RUN groupadd --system appgroup && \
    useradd  --system --gid appgroup --no-create-home appuser && \
    chown -R appuser:appgroup /app

RUN mkdir -p /tmp/easyocr && chmod -R 777 /tmp/easyocr

USER appuser

# Expose port aplikasi
EXPOSE 2200

# Health check: verifikasi API merespons setiap 30 detik
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:2200/health || exit 1

# Entrypoint: jalankan FastAPI dengan uvicorn
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "2200", \
     "--log-level", "info", \
     "--workers", "1"]
