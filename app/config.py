"""
app/config.py - Konfigurasi terpusat berbasis environment variable.

Semua nilai dapat di-override lewat .env atau environment container.
Lihat .env.example untuk daftar lengkap variabel yang tersedia.
"""

import os
from pathlib import Path

# ─── Model paths ─────────────────────────────────────────────────────────────

# Stage 1: deteksi jenis kendaraan (ambulance / police / fire_truck)
VEHICLE_MODEL_PATH = Path(
    os.getenv("VEHICLE_MODEL_PATH", "model/vehicle_best.pt")
)

# Stage 2: deteksi plat nomor di dalam crop kendaraan
NOPOL_MODEL_PATH = Path(
    os.getenv("NOPOL_MODEL_PATH", "model/nopol_best.pt")
)

# ─── Confidence thresholds ────────────────────────────────────────────────────

# Stage 1 threshold: lebih tinggi karena kendaraan objek besar, false-positive mahal
VEHICLE_CONF_THRESHOLD = float(os.getenv("VEHICLE_CONF_THRESHOLD", "0.05"))

# Stage 2 threshold: lebih rendah karena plat kecil dan sering terpotong
NOPOL_CONF_THRESHOLD = float(os.getenv("NOPOL_CONF_THRESHOLD", "0.05"))

# ─── Crop padding ─────────────────────────────────────────────────────────────

# Padding (px) saat crop bbox kendaraan sebelum dikirim ke Stage 2.
# Lebih besar = lebih aman; terlalu besar = lebih lambat dan tambah noise.
VEHICLE_CROP_PADDING = int(os.getenv("VEHICLE_CROP_PADDING", "15"))

# Padding (px) saat crop bbox plat sebelum dikirim ke OCR.
PLATE_CROP_PADDING = int(os.getenv("PLATE_CROP_PADDING", "4"))

# ─── OCR ─────────────────────────────────────────────────────────────────────

OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_GPU  = os.getenv("OCR_GPU", "false").lower() == "true"

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
