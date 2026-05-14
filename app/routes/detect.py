"""
routes/detect.py - POST /api/v1/detect
Deteksi kendaraan prioritas menggunakan pipeline 2 tahap.

Flow lengkap:
  1. Upload gambar (multipart/form-data)
  2. Decode bytes -> numpy BGR
  3. Stage 1: vehicle_model pada gambar PENUH -> bbox kendaraan + kelas
  4. Crop kendaraan dengan padding (15px) dari gambar asli
  5. Stage 2: nopol_model pada CROP kendaraan -> bbox plat
  6. Crop plat dari vehicle_crop
  7. OCR pada crop plat -> teks nomor polisi
  8. Return JSON response

Response (mobile-friendly, tanpa gambar/base64):
  {
    "vehicle":        "ambulance" | null,
    "plate_number":   "B1234XYZ"  | "",
    "confidence":     0.92        | 0,
    "plate_detected": true        | false
  }
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import PLATE_CROP_PADDING, VEHICLE_CROP_PADDING
from app.services.ocr_service import OCRService
from app.services.yolo_service import YOLOService
from app.utils.image_utils import crop_region, decode_image

router = APIRouter()
log    = logging.getLogger(__name__)

# ─── Singleton services — di-load SEKALI saat module pertama kali di-import ──
# Jika model tidak ditemukan, FileNotFoundError naik ke atas dan server gagal
# startup dengan pesan error yang jelas.
yolo_service = YOLOService()
ocr_service  = OCRService()

# ─── Response standar saat tidak ada deteksi ─────────────────────────────────
EMPTY_RESPONSE = {
    "vehicle":        None,
    "plate_number":   "",
    "confidence":     0,
    "plate_detected": False,
}


@router.post(
    "/detect",
    summary="Deteksi Kendaraan Prioritas + Baca Nomor Polisi",
    response_description="Hasil deteksi kendaraan prioritas dan nomor polisinya",
)
async def detect(
    file: UploadFile = File(..., description="Gambar kendaraan (JPEG/PNG)"),
):
    """
    Endpoint utama sistem penelitian deteksi kendaraan prioritas.

    **Input:** file gambar multipart/form-data

    **Output JSON:**
    ```json
    {
      "vehicle":        "ambulance",
      "plate_number":   "B1234XYZ",
      "confidence":     0.92,
      "plate_detected": true
    }
    ```

    Jika tidak ada kendaraan terdeteksi:
    ```json
    { "vehicle": null, "plate_number": "", "confidence": 0, "plate_detected": false }
    ```

    Jika kendaraan terdeteksi tapi plat tidak ditemukan:
    ```json
    { "vehicle": "ambulance", "plate_number": "", "confidence": 0.92, "plate_detected": false }
    ```
    """

    # ── 1. Validasi format file ──────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Format tidak didukung. Gunakan JPEG atau PNG.",
        )

    # ── 2. Decode bytes -> numpy BGR ─────────────────────────────────────────
    image_bytes = await file.read()
    image_bgr   = decode_image(image_bytes)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Gagal membaca gambar.")

    log.info("[detect] Gambar diterima: %s  ukuran=%s", file.filename, image_bgr.shape)

    try:
        # ── 3. STAGE 1: deteksi kendaraan di gambar penuh ───────────────────
        vehicle = yolo_service.detect_vehicle(image_bgr)

        if vehicle is None:
            # Tidak ada kendaraan prioritas -> langsung return kosong
            log.info("[detect] Stage 1 tidak menemukan kendaraan.")
            return JSONResponse(content=EMPTY_RESPONSE)

        # ── 4. Crop kendaraan dari gambar ASLI (resolusi penuh) ──────────────
        # Padding 15px agar sisi plat tidak terpotong tepi bbox kendaraan
        vehicle_crop = crop_region(
            image_bgr,
            vehicle["bbox"],
            padding=VEHICLE_CROP_PADDING,
        )

        # ── 5. STAGE 2: deteksi plat di dalam crop kendaraan ────────────────
        plate_number   = ""
        plate_detected = False

        plate = yolo_service.detect_plate(vehicle_crop)

        if plate is not None:
            # ── 6. Crop plat dari vehicle_crop ──────────────────────────────
            # Bbox plate dari Stage 2 sudah relatif terhadap vehicle_crop
            plate_crop = crop_region(
                vehicle_crop,
                plate["bbox"],
                padding=PLATE_CROP_PADDING,
            )

            # ── 7. OCR pada crop plat ────────────────────────────────────────
            if plate_crop is not None:
                plate_number   = ocr_service.read_plate(plate_crop)
                plate_detected = True

        # ── 8. Susun dan kembalikan response ─────────────────────────────────
        response = {
            "vehicle":        vehicle["class_name"],
            "plate_number":   plate_number,
            "confidence":     round(vehicle["confidence"], 4),
            "plate_detected": plate_detected,
        }
        log.info("[detect] Response: %s", response)
        return JSONResponse(content=response)

    except Exception as exc:
        log.exception("[detect] Error selama inferensi: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan internal saat proses deteksi.",
        )
