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

        # ── Filter: plat asli punya rasio aspek & ukuran yang khas, sehingga
        #    bbox yang sekadar berskor tinggi tapi tak proporsional (mis.
        #    teks bodi besar) dibuang DULU sebelum ambil confidence tertinggi.
        crop_h, crop_w = vehicle_crop.shape[:2]

        valid_plates: List[Dict[str, Any]] = []
        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            pw = x2 - x1
            ph = y2 - y1

            if ph == 0:
                continue

            aspect  = pw / ph
            w_ratio = pw / crop_w
            h_ratio = ph / crop_h

            if (1.5 <= aspect  <= 6.0  and
                0.05 <= w_ratio <= 0.20 and
                0.03 <= h_ratio <= 0.20):
                valid_plates.append(plate)
                log.info(
                    "[Stage2] Valid plate candidate: conf=%.3f bbox=%s "
                    "aspect=%.2f w_ratio=%.2f h_ratio=%.2f",
                    plate["confidence"], plate["bbox"],
                    aspect, w_ratio, h_ratio,
                )

        # Scoring: confidence × position_bonus × size_penalty.
        # — Plat asli umumnya berada di paruh BAWAH crop kendaraan
        #   (cy > 50% tinggi crop) DAN KECIL relatif terhadap lebar kendaraan.
        def plate_score(p, crop_w, crop_h):
            x1, y1, x2, y2 = p["bbox"]
            pw = x2 - x1
            ph = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            conf = p["confidence"]

            # Prefer plates in lower half of vehicle (cy > 50% of crop height)
            position_bonus = 1.2 if cy > crop_h * 0.5 else 0.8

            # Prefer smaller plates (real plates are small relative to vehicle).
            # Squared agar gap (1 - pw/crop_w) lebih agresif memberatkan bbox lebar.
            size_penalty = (1.0 - (pw / crop_w)) ** 2   # smaller pw → higher score

            score = conf * position_bonus * size_penalty
            return score

        if valid_plates:
            log.info(
                "[Stage2] Plate scores: %s",
                [(p["bbox"], round(plate_score(p, crop_w, crop_h), 3))
                 for p in valid_plates],
            )
            best = max(
                valid_plates,
                key=lambda p: plate_score(p, crop_w, crop_h),
            )
        else:
            # Fallback: jika tak ada yang lolos filter, ambil bbox TERKECIL
            # (plat asli biasanya kecil dibanding teks bodi besar).
            best = min(
                plates,
                key=lambda p: (p["bbox"][2] - p["bbox"][0])
                              * (p["bbox"][3] - p["bbox"][1]),
            )
            log.warning(
                "[Stage2] No valid plate by ratio filter, using smallest bbox"
            )

        log.info(
            "[Stage2] Plat terdeteksi  conf=%.3f  bbox=%s",
            best["confidence"],
            best["bbox"],
        )
        return best
