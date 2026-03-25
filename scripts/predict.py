"""
scripts/predict.py - Script standalone untuk testing inferensi model
Bisa digunakan untuk test model tanpa menjalankan API server.

Cara menjalankan:
  python scripts/predict.py --image path/ke/gambar.jpg
  python scripts/predict.py --image path/ke/gambar.jpg --show
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Tambahkan root project ke sys.path agar bisa import app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.yolo_service import YOLOService, CLASS_NAMES
from app.services.ocr_service import OCRService
from app.utils.image_utils import crop_region

# Warna per kelas untuk visualisasi bounding box
CLASS_COLORS = {
    0: (0, 255, 255),   # ambulance  → kuning
    1: (255, 0, 0),     # police     → biru
    2: (0, 0, 255),     # fire_truck → merah
    3: (0, 255, 0),     # license_plate → hijau
}

LICENSE_PLATE_CLASS_ID = 3


def draw_detections(image: np.ndarray, detections: list, plate_text: str | None) -> np.ndarray:
    """Gambar bounding box dan label pada gambar."""
    vis = image.copy()
    h, w = vis.shape[:2]

    for det in detections:
        cid  = det["class_id"]
        name = det["class_name"]
        conf = det["confidence"]
        b    = det["bbox"]

        color = CLASS_COLORS.get(cid, (255, 255, 255))
        label = f"{name} {conf:.2f}"

        x1, y1, x2, y2 = b
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Background label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            vis, label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
        )

    # Tampilkan nomor plat di pojok kiri atas
    if plate_text:
        cv2.putText(
            vis, f"Plate: {plate_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )

    return vis


def predict_image(image_path: str, show: bool = False, save: bool = True):
    """
    Jalankan deteksi + OCR pada satu gambar.

    Args:
        image_path : Path ke file gambar.
        show       : Tampilkan hasil di jendela OpenCV.
        save       : Simpan gambar hasil ke disk.
    """
    path = Path(image_path)
    if not path.exists():
        print(f"[ERROR] File tidak ditemukan: {image_path}")
        sys.exit(1)

    print(f"\nMemproses: {path.name}")

    # Load gambar
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        print("[ERROR] Gagal membaca gambar.")
        sys.exit(1)

    # Inisialisasi service
    yolo = YOLOService()
    ocr  = OCRService()

    # Deteksi YOLO (semua objek)
    detections = yolo.detect(image_bgr)
    print(f"  Terdeteksi : {len(detections)} objek")

    # Pilih kendaraan prioritas utama (confidence tertinggi)
    primary_vehicle = yolo.get_primary_vehicle(detections)

    # Pilih plat nomor terbaik (terdekat dengan kendaraan utama)
    best_plate = yolo.get_best_plate(detections, primary_vehicle)

    # OCR pada crop plat terpilih
    plate_text = ""
    if best_plate is not None:
        crop = crop_region(image_bgr, best_plate["bbox"])
        if crop is not None:
            plate_text = ocr.read_plate(crop)

    # Print hasil (format sama dengan response API)
    print("\n  === Hasil Deteksi ===")
    if primary_vehicle:
        print(f"  Kendaraan  : {primary_vehicle['class_name']} (conf={primary_vehicle['confidence']:.4f})")
    else:
        print("  Kendaraan  : Tidak terdeteksi")
    print(f"  Nomor Plat : {plate_text or 'Tidak terbaca'}")

    # Visualisasi
    result_img = draw_detections(image_bgr, detections, plate_text)

    if save:
        out_path = path.parent / f"{path.stem}_result{path.suffix}"
        cv2.imwrite(str(out_path), result_img)
        print(f"\n  Gambar hasil disimpan: {out_path}")

    if show:
        cv2.imshow("Detection Result", result_img)
        print("\n  Tekan sembarang tombol untuk menutup jendela...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections, plate_text


def main():
    parser = argparse.ArgumentParser(
        description="Predict - Deteksi Kendaraan Prioritas + OCR Plat Nomor"
    )
    parser.add_argument("--image", required=True, help="Path ke file gambar input")
    parser.add_argument("--show", action="store_true", help="Tampilkan hasil di jendela")
    parser.add_argument("--no-save", action="store_true", help="Jangan simpan gambar hasil")
    args = parser.parse_args()

    predict_image(
        image_path=args.image,
        show=args.show,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
