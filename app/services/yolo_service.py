"""
services/yolo_service.py - Pipeline deteksi 2 tahap menggunakan YOLOv8

Stage 1 — vehicle_model (vehicle_best.pt):
  Input  : gambar penuh (full resolution)
  Output : bbox kendaraan + nama kelas (ambulance / police / fire_truck)

Stage 2 — nopol_model (nopol_best.pt):
  Input  : crop kendaraan dari Stage 1 (dengan padding)
  Output : bbox plat nomor (relatif terhadap crop, bukan gambar asli)

Kedua model di-load SEKALI saat startup dan di-reuse di setiap request.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from app.config import (
    NOPOL_CONF_THRESHOLD,
    NOPOL_MODEL_PATH,
    VEHICLE_CONF_THRESHOLD,
    VEHICLE_MODEL_PATH,
)

log = logging.getLogger(__name__)

# Nama kelas Stage 1 — harus sinkron dengan dataset.yaml training vehicle_best.pt
VEHICLE_CLASS_NAMES: Dict[int, str] = {
    0: "ambulance",
    1: "police",
    2: "fire_truck",
}
VEHICLE_CLASS_IDS = set(VEHICLE_CLASS_NAMES.keys())


class YOLOService:
    """
    Service singleton untuk pipeline deteksi 2 tahap.
    Di-inisialisasi sekali saat startup FastAPI; di-reuse tanpa reload.
    """

    def __init__(self) -> None:
        log.info("[YOLOService] Memulai inisialisasi pipeline 2 tahap...")
        self.vehicle_model = self._load_model(VEHICLE_MODEL_PATH, "vehicle (Stage 1)")
        self.nopol_model   = self._load_model(NOPOL_MODEL_PATH,   "nopol   (Stage 2)")
        log.info("[YOLOService] Kedua model berhasil di-load.")

    # ─── Internal: load model ────────────────────────────────────────────────

    def _load_model(self, path: Path, label: str) -> YOLO:
        """
        Load model YOLO dari path.
        Raise FileNotFoundError dengan pesan jelas jika file tidak ada.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"[YOLOService] Model '{label}' tidak ditemukan: {path.resolve()}\n"
                "Pastikan training selesai atau set env:\n"
                "  VEHICLE_MODEL_PATH=model/vehicle_best.pt\n"
                "  NOPOL_MODEL_PATH=model/nopol_best.pt"
            )
        log.info("[YOLOService] Loading %s: %s", label, path)
        model = YOLO(str(path))
        log.info("[YOLOService] %s ready.", label)
        return model

    # ─── Stage 1: deteksi kendaraan di gambar penuh ──────────────────────────

    def detect_vehicle(
        self, image_bgr: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 1: jalankan vehicle_model pada gambar asli resolusi penuh.

        Jika lebih dari satu kendaraan terdeteksi, ambil yang confidence-nya
        paling tinggi.

        Args:
            image_bgr: Gambar asli format BGR numpy array.

        Returns:
            Dict dengan keys: class_id, class_name, confidence, bbox [x1,y1,x2,y2]
            atau None jika tidak ada kendaraan prioritas terdeteksi.
        """
        try:
            results = self.vehicle_model.predict(
                source=image_bgr,
                conf=VEHICLE_CONF_THRESHOLD,
                verbose=False,
            )
        except Exception as exc:
            log.error("[Stage1] Inference error: %s", exc)
            raise

        candidates: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                candidates.append(
                    {
                        "class_id":   class_id,
                        "class_name": VEHICLE_CLASS_NAMES[class_id],
                        "confidence": float(box.conf[0]),
                        "bbox":       [x1, y1, x2, y2],
                    }
                )

        if not candidates:
            log.info("[Stage1] Tidak ada kendaraan prioritas terdeteksi.")
            return None

        best = max(candidates, key=lambda d: d["confidence"])
        log.info(
            "[Stage1] Terdeteksi: %s  conf=%.3f  bbox=%s",
            best["class_name"],
            best["confidence"],
            best["bbox"],
        )
        return best

    # ─── Stage 2: deteksi plat di crop kendaraan ─────────────────────────────

    def detect_plate(
        self, vehicle_crop: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 2: jalankan nopol_model pada crop kendaraan dari Stage 1.

        Bbox yang dikembalikan relatif terhadap vehicle_crop (bukan gambar asli).
        Untuk crop OCR, gunakan koordinat ini pada vehicle_crop.

        Args:
            vehicle_crop: Crop kendaraan (BGR numpy array), sudah termasuk
                          padding dari crop_region().

        Returns:
            Dict dengan keys: confidence, bbox [x1,y1,x2,y2] (relatif ke crop)
            atau None jika tidak ada plat terdeteksi.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            log.warning("[Stage2] vehicle_crop kosong — skip deteksi plat.")
            return None

        try:
            results = self.nopol_model.predict(
                source=vehicle_crop,
                conf=NOPOL_CONF_THRESHOLD,
                verbose=False,
            )
        except Exception as exc:
            log.error("[Stage2] Inference error: %s", exc)
            raise

        plates: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                plates.append(
                    {
                        "confidence": float(box.conf[0]),
                        "bbox":       [x1, y1, x2, y2],
                    }
                )

        if not plates:
            log.info("[Stage2] Tidak ada plat terdeteksi di crop kendaraan.")
            return None

        best = max(plates, key=lambda p: p["confidence"])
        log.info(
            "[Stage2] Plat terdeteksi  conf=%.3f  bbox=%s",
            best["confidence"],
            best["bbox"],
        )
        return best
