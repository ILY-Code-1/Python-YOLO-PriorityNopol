"""
routes/detect.py - Route endpoint deteksi kendaraan prioritas
POST /api/v1/detect

Flow:
  Upload gambar → YOLO detect → pilih kendaraan utama → crop plat → OCR → JSON

Response (mobile-friendly, ringan, tanpa gambar/base64):
  {
    "vehicle":        "ambulance" | null,
    "plate_number":   "B1234XYZ"  | "",
    "confidence":     0.92        | 0,
    "plate_detected": true        | false
  }
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.services.yolo_service import YOLOService
from app.services.ocr_service import OCRService
from app.utils.image_utils import decode_image, crop_region

router = APIRouter()

# ─── Singleton services (di-load satu kali saat startup) ────────────────────
yolo_service = YOLOService()
ocr_service  = OCRService()

# ─── Response fallback saat tidak ada deteksi ────────────────────────────────
EMPTY_RESPONSE = {
    "vehicle":        None,
    "plate_number":   "",
    "confidence":     0,
    "plate_detected": False,
}


@router.post(
    "/detect",
    summary="Deteksi Kendaraan Prioritas + Baca Nomor Polisi",
    response_description="Kendaraan prioritas terdeteksi dan nomor polisinya",
)
async def detect(
    file: UploadFile = File(..., description="Gambar kendaraan (JPEG/PNG)")
):
    """
    Endpoint utama sistem penelitian.

    **Input:** file gambar (multipart/form-data)

    **Output JSON:**
    ```json
    {
      "vehicle":        "ambulance",
      "plate_number":   "B1234XYZ",
      "confidence":     0.92,
      "plate_detected": true
    }
    ```

    Jika tidak terdeteksi:
    ```json
    {
      "vehicle":        null,
      "plate_number":   "",
      "confidence":     0,
      "plate_detected": false
    }
    ```
    """

    # ── 1. Validasi format file ──────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Format tidak didukung. Gunakan JPEG atau PNG.",
        )

    # ── 2. Decode bytes → numpy BGR ──────────────────────────────────────────
    image_bytes = await file.read()
    image_bgr   = decode_image(image_bytes)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Gagal membaca gambar.")

    # ── 3. YOLO inference (semua objek) ──────────────────────────────────────
    detections = yolo_service.detect(image_bgr)

    # ── 4. Pilih 1 kendaraan prioritas (confidence tertinggi) ────────────────
    primary_vehicle = yolo_service.get_primary_vehicle(detections)

    if primary_vehicle is None:
        # Tidak ada kendaraan prioritas → kembalikan response kosong
        return JSONResponse(content=EMPTY_RESPONSE)

    # ── 5. Pilih plat nomor terbaik (terdekat dengan kendaraan) ─────────────
    best_plate = yolo_service.get_best_plate(detections, primary_vehicle)

    # ── 6. OCR pada crop plat (jika plat terdeteksi) ─────────────────────────
    plate_number   = ""
    plate_detected = False

    if best_plate is not None:
        plate_crop = crop_region(image_bgr, best_plate["bbox"])
        if plate_crop is not None:
            plate_number   = ocr_service.read_plate(plate_crop)
            plate_detected = bool(plate_number)

    # ── 7. Susun response ────────────────────────────────────────────────────
    return JSONResponse(
        content={
            "vehicle":        primary_vehicle["class_name"],
            "plate_number":   plate_number,
            "confidence":     round(primary_vehicle["confidence"], 4),
            "plate_detected": plate_detected,
        }
    )
