"""
services/yolo_service.py - Service untuk YOLOv8n inference
Bertanggung jawab untuk:
  - Load model YOLO dari model/best.pt (atau fallback ke yolov8n.pt pretrained)
  - Jalankan inference pada gambar
  - Pilih 1 kendaraan prioritas dengan confidence tertinggi
  - Pilih plat nomor yang paling dekat dengan kendaraan utama
"""

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from ultralytics import YOLO

# Mapping class ID → nama kelas (sinkron dengan dataset.yaml)
CLASS_NAMES = {
    0: "ambulance",
    1: "police",
    2: "fire_truck",
    3: "license_plate",
}

# Class ID kendaraan prioritas yang diperhitungkan
PRIORITY_VEHICLE_CLASS_IDS = {0, 1, 2}
LICENSE_PLATE_CLASS_ID = 3

# Path model. Fallback ke pretrained COCO jika best.pt belum ada
MODEL_PATH = Path("model/best.pt")
PRETRAINED_FALLBACK = "yolov8n.pt"

# Confidence minimum untuk masuk ke hasil deteksi
CONFIDENCE_THRESHOLD = 0.40


class YOLOService:
    """
    Service singleton untuk YOLOv8n inference.
    Model di-load sekali saat inisialisasi agar tidak reload setiap request.
    """

    def __init__(self):
        self.model = self._load_model()

    # ─── Internal: load model ───────────────────────────────────────────────

    def _load_model(self) -> YOLO:
        """Load model best.pt jika ada, fallback ke pretrained yolov8n.pt."""
        if MODEL_PATH.exists():
            print(f"[YOLOService] Loading trained model: {MODEL_PATH}")
            return YOLO(str(MODEL_PATH))
        print(
            f"[YOLOService] model/best.pt tidak ditemukan, "
            f"menggunakan pretrained: {PRETRAINED_FALLBACK}"
        )
        return YOLO(PRETRAINED_FALLBACK)

    # ─── Public: raw inference ───────────────────────────────────────────────

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Jalankan YOLO inference, kembalikan semua deteksi.

        Returns:
            List of dict:
              - class_id   : int
              - class_name : str
              - confidence : float
              - bbox       : [x1, y1, x2, y2]  ← format list (mobile-friendly)
        """
        results = self.model.predict(
            source=image_bgr,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(
                    {
                        "class_id":   class_id,
                        "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        "confidence": float(box.conf[0]),
                        "bbox":       [x1, y1, x2, y2],
                    }
                )
        return detections

    # ─── Public: pilih kendaraan prioritas utama ─────────────────────────────

    def get_primary_vehicle(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Filter kendaraan prioritas (ambulance/police/fire_truck) lalu
        kembalikan 1 objek dengan confidence tertinggi.

        Returns:
            Dict deteksi kendaraan terpilih, atau None jika tidak ada.
        """
        vehicles = [
            d for d in detections
            if d["class_id"] in PRIORITY_VEHICLE_CLASS_IDS
        ]
        if not vehicles:
            return None
        return max(vehicles, key=lambda d: d["confidence"])

    # ─── Public: pilih plat nomor terbaik ────────────────────────────────────

    def get_best_plate(
        self,
        detections: List[Dict[str, Any]],
        vehicle: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Dari semua deteksi license_plate:
          - Jika hanya ada 1 plat → langsung pilih.
          - Jika ada lebih dari 1 dan ada kendaraan utama →
            pilih plat yang center-nya paling dekat dengan center kendaraan.
          - Jika ada lebih dari 1 tanpa kendaraan → pilih confidence tertinggi.

        Returns:
            Dict deteksi plat terpilih, atau None jika tidak ada.
        """
        plates = [d for d in detections if d["class_id"] == LICENSE_PLATE_CLASS_ID]

        if not plates:
            return None
        if len(plates) == 1:
            return plates[0]

        # Lebih dari satu plat → cari yang paling dekat dengan kendaraan
        if vehicle is not None:
            return min(plates, key=lambda p: self._center_distance(p["bbox"], vehicle["bbox"]))

        # Tidak ada kendaraan rujukan → ambil confidence tertinggi
        return max(plates, key=lambda p: p["confidence"])

    # ─── Helper: hitung jarak antar center dua bbox ──────────────────────────

    @staticmethod
    def _center_distance(bbox_a: List[int], bbox_b: List[int]) -> float:
        """Jarak Euclidean antara titik tengah dua bounding box."""
        cx_a = (bbox_a[0] + bbox_a[2]) / 2
        cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cx_b = (bbox_b[0] + bbox_b[2]) / 2
        cy_b = (bbox_b[1] + bbox_b[3]) / 2
        return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
