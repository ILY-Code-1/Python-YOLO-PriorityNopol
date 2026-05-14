"""
app/main.py - Entry point FastAPI application
Sistem Deteksi Kendaraan Prioritas + OCR Nomor Polisi (2-Stage Pipeline)
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import LOG_LEVEL, NOPOL_MODEL_PATH, VEHICLE_MODEL_PATH
from app.routes.detect import router as detect_router

# ─── Logging setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Priority Vehicle Detection API",
    description=(
        "API deteksi otomatis kendaraan prioritas (ambulance, police, fire_truck) "
        "menggunakan pipeline 2 tahap YOLOv8n. "
        "Stage 1: deteksi jenis kendaraan. "
        "Stage 2: deteksi plat nomor di dalam crop kendaraan. "
        "Digunakan untuk keperluan penelitian."
    ),
    version="2.0.0",
    root_path="/py-yolo-nopol",
)

# ─── CORS ────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Di production, ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Router ──────────────────────────────────────────────────────────────────

# Import router (yang akan memicu load kedua model sebagai singleton)
app.include_router(detect_router, prefix="/api/v1", tags=["Detection"])

log.info("==============================================")
log.info("  Priority Vehicle Detection API v2.0.0")
log.info("  Pipeline: 2-Stage YOLOv8n + EasyOCR")
log.info("  Vehicle model : %s", VEHICLE_MODEL_PATH)
log.info("  Nopol model   : %s", NOPOL_MODEL_PATH)
log.info("==============================================")


# ─── Health endpoints ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Endpoint health-check sederhana."""
    return {
        "status":  "ok",
        "message": "Priority Vehicle Detection API is running.",
        "version": "2.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint health-check untuk monitoring (load-balancer / Docker healthcheck)."""
    return {"status": "healthy"}
