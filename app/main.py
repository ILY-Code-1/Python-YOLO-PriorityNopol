"""
main.py - Entry point FastAPI application
Sistem Deteksi Kendaraan Prioritas + OCR Nomor Polisi
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.detect import router as detect_router

# Inisialisasi FastAPI app
app = FastAPI(
    title="Priority Vehicle Detection API",
    description=(
        "API untuk deteksi otomatis kendaraan prioritas (ambulance, police, fire_truck) "
        "menggunakan YOLOv8n dan pembacaan nomor polisi menggunakan EasyOCR. "
        "Digunakan untuk keperluan penelitian."
    ),
    version="1.0.0",
    root_path="/py-yolo-nopol"
)

# Konfigurasi CORS agar bisa diakses dari frontend / Postman / klien lain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Di production, ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftarkan router deteksi
app.include_router(detect_router, prefix="/api/v1", tags=["Detection"])


@app.get("/", tags=["Health"])
def root():
    """Endpoint health-check sederhana."""
    return {
        "status": "ok",
        "message": "Priority Vehicle Detection API is running.",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint health-check yang lebih eksplisit untuk monitoring."""
    return {"status": "healthy"}
