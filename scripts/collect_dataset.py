"""
scripts/collect_dataset.py
──────────────────────────
Dataset Collection Script untuk Deteksi Kendaraan Prioritas
Sumber: Bing Image Crawler (icrawler) + filter otomatis
Target : ≥500 gambar (ambulance, police, fire_truck, license_plate)

Cara pakai:
    python scripts/collect_dataset.py
"""

import os
import sys
import json
import time
import hashlib
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
from PIL import Image
from icrawler.builtin import BingImageCrawler

# ─── Konfigurasi logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Direktori ───────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / "dataset" / "_raw"       # Folder sementara per kelas
OUT_DIR   = BASE_DIR / "dataset" / "images"     # Output final (flat)

# ─── Target jumlah gambar per kelas ──────────────────────────────────────────
CLASS_CONFIG: Dict[str, Dict] = {
    "ambulance": {
        "target": 130,
        "queries": [
            "ambulance Indonesia street",
            "mobil ambulans Indonesia",
            "ambulance 118 Indonesia",
            "ambulance vehicle road Indonesia",
            "Indonesian ambulance side view",
        ],
    },
    "police": {
        "target": 130,
        "queries": [
            "mobil polisi Indonesia",
            "police car Indonesia traffic",
            "patroli polisi Indonesia",
            "Indonesian police vehicle",
            "patwal polisi Indonesia",
        ],
    },
    "fire_truck": {
        "target": 130,
        "queries": [
            "fire truck Indonesia",
            "damkar Indonesia",
            "mobil pemadam kebakaran Indonesia",
            "Indonesian fire engine",
            "truk pemadam Indonesia jalan",
        ],
    },
    "license_plate": {
        "target": 160,
        "queries": [
            "plat nomor Indonesia kendaraan",
            "license plate Indonesia motorcycle",
            "nomor polisi mobil Indonesia",
            "Indonesian vehicle license plate close up",
            "plat kendaraan bermotor Indonesia",
            "license plate Indonesia car street",
        ],
    },
}

# ─── Parameter kualitas gambar ────────────────────────────────────────────────
MIN_WIDTH  = 100   # px minimum
MIN_HEIGHT = 100   # px minimum
MAX_WIDTH  = 4096  # px maximum (filter gambar terlalu besar)
MAX_HEIGHT = 4096


# ═══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTILITAS
# ═══════════════════════════════════════════════════════════════════════════════

def file_hash(path: Path) -> str:
    """MD5 hash isi file (untuk deteksi duplikat)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path) -> Tuple[bool, str]:
    """
    Validasi gambar: bisa dibaca, ukuran wajar, tidak korup.
    Returns (valid: bool, reason: str)
    """
    try:
        # Cek dengan PIL
        with Image.open(path) as img:
            img.verify()

        # Cek ulang dengan OpenCV untuk memastikan bisa di-decode
        img_cv = cv2.imread(str(path))
        if img_cv is None:
            return False, "opencv_cannot_read"

        h, w = img_cv.shape[:2]
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return False, f"too_small ({w}x{h})"
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            return False, f"too_large ({w}x{h})"

        return True, "ok"

    except Exception as e:
        return False, str(e)


def download_class(class_name: str, config: dict, raw_class_dir: Path) -> int:
    """
    Download gambar satu kelas menggunakan BingImageCrawler.
    Mencoba setiap query sampai target terpenuhi.

    Returns jumlah gambar berhasil didownload.
    """
    raw_class_dir.mkdir(parents=True, exist_ok=True)
    target    = config["target"]
    queries   = config["queries"]
    collected = list(raw_class_dir.glob("*.jpg")) + list(raw_class_dir.glob("*.png")) + \
                list(raw_class_dir.glob("*.jpeg")) + list(raw_class_dir.glob("*.webp"))

    log.info(f"[{class_name}] Sudah ada {len(collected)} gambar, target {target}")

    for i, query in enumerate(queries):
        existing = len(list(raw_class_dir.iterdir()))
        if existing >= target:
            log.info(f"[{class_name}] Target {target} tercapai, skip query berikutnya.")
            break

        per_query = min(target - existing + 10, 60)  # +10 buffer untuk yang invalid
        log.info(f"[{class_name}] Query {i+1}/{len(queries)}: '{query}' → minta {per_query} gambar")

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(raw_class_dir)},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
            )
            crawler.crawl(
                keyword=query,
                max_num=per_query,
                min_size=(MIN_WIDTH, MIN_HEIGHT),
                file_idx_offset="auto",
            )
        except Exception as e:
            log.warning(f"[{class_name}] Query gagal: {e}")

        time.sleep(2)  # Jeda antar query untuk menghindari rate limit

    # Hitung hasil
    all_files = [f for f in raw_class_dir.iterdir() if f.is_file()]
    return len(all_files)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE UTAMA
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("  DATASET COLLECTION – Kendaraan Prioritas + Plat Nomor")
    log.info("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download_stats: Dict[str, int] = {}

    # ── FASE 1: Download per kelas ──────────────────────────────────────────
    log.info("\n[FASE 1] Mendownload gambar per kelas...")
    for class_name, config in CLASS_CONFIG.items():
        raw_class_dir = RAW_DIR / class_name
        count = download_class(class_name, config, raw_class_dir)
        download_stats[class_name] = count
        log.info(f"[{class_name}] Selesai: {count} file didownload")

    # ── FASE 2: Filter + Deduplikasi ─────────────────────────────────────────
    log.info("\n[FASE 2] Memfilter dan mendeduplikasi gambar...")

    seen_hashes: set = set()
    valid_images: List[Dict] = []
    filter_stats: Dict[str, int] = {"duplicate": 0, "invalid": 0, "ok": 0}

    for class_name in CLASS_CONFIG:
        raw_class_dir = RAW_DIR / class_name
        if not raw_class_dir.exists():
            continue

        files = sorted(raw_class_dir.iterdir())
        class_valid = 0

        for f in files:
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue

            # Cek duplikat
            h = file_hash(f)
            if h in seen_hashes:
                filter_stats["duplicate"] += 1
                continue
            seen_hashes.add(h)

            # Validasi kualitas
            ok, reason = is_valid_image(f)
            if not ok:
                filter_stats["invalid"] += 1
                log.debug(f"  Skip {f.name}: {reason}")
                continue

            valid_images.append({"path": f, "class": class_name})
            class_valid += 1
            filter_stats["ok"] += 1

        log.info(f"  [{class_name}] Valid setelah filter: {class_valid}")

    log.info(f"\n  Total valid  : {filter_stats['ok']}")
    log.info(f"  Duplikat     : {filter_stats['duplicate']}")
    log.info(f"  Invalid      : {filter_stats['invalid']}")

    # ── FASE 3: Rename + Salin ke OUT_DIR ────────────────────────────────────
    log.info("\n[FASE 3] Menyalin dan merename gambar ke dataset/images/...")

    # Hapus konten lama di OUT_DIR (kecuali train/val subdirs yang sudah ada)
    for f in OUT_DIR.glob("img_*.jpg"):
        f.unlink()

    class_counters: Dict[str, int] = {c: 0 for c in CLASS_CONFIG}
    global_counter = 1
    copied_images: List[Dict] = []

    for item in valid_images:
        src        = item["path"]
        class_name = item["class"]
        dst_name   = f"img_{global_counter:05d}.jpg"
        dst_path   = OUT_DIR / dst_name

        # Konversi ke JPG menggunakan PIL (standarisasi format)
        try:
            with Image.open(src) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(dst_path, "JPEG", quality=90)
        except Exception as e:
            log.warning(f"  Gagal konversi {src.name}: {e}")
            continue

        class_counters[class_name] += 1
        copied_images.append({
            "filename": dst_name,
            "class": class_name,
            "original": src.name,
        })
        global_counter += 1

    total_final = global_counter - 1
    log.info(f"\n  Total gambar final : {total_final}")
    for cls, cnt in class_counters.items():
        log.info(f"    {cls:15s} : {cnt}")

    # ── FASE 4: dataset_info.json ─────────────────────────────────────────────
    log.info("\n[FASE 4] Membuat dataset_info.json...")

    dataset_info = {
        "total_images": total_final,
        "target_minimum": 500,
        "target_met": total_final >= 500,
        "sources": [
            {
                "name": "Bing Image Crawler (icrawler)",
                "type": "web_crawl",
                "license": "mixed (public web images – for research/educational use only)",
                "classes_covered": list(CLASS_CONFIG.keys()),
            }
        ],
        "classes": ["ambulance", "police", "fire_truck", "license_plate"],
        "class_distribution": class_counters,
        "has_annotations": False,
        "annotation_note": (
            "Gambar belum dianotasi. Gunakan LabelImg atau Roboflow untuk anotasi manual "
            "sebelum training. Format label: YOLO (class_id x_center y_center width height)."
        ),
        "format": "JPEG, 90% quality, minimum 100x100px",
        "split_recommended": "80% train / 20% val",
        "split_train_count": int(total_final * 0.8),
        "split_val_count":   int(total_final * 0.2),
        "reference_sources_from_pdf": [
            {
                "name": "Roboflow Emergency Vehicle Detection",
                "url": "https://universe.roboflow.com/test-ho5i1/emergency-vehicle-detection-yx3gh",
                "images": "~3900", "license": "CC BY 4.0",
            },
            {
                "name": "Roboflow ambulance-police-firetruck",
                "url": "https://universe.roboflow.com/detection-cars/ambulance-police-firetruck",
                "images": "~1900", "license": "CC BY 4.0",
            },
            {
                "name": "Roboflow Vehicles-OpenImages",
                "url": "https://universe.roboflow.com/roboflow-gw7yv/vehicles-openimages",
                "images": "627", "license": "CC BY 4.0",
            },
            {
                "name": "Roboflow vehicle and license plate (Plat Kendaraan)",
                "url": "https://universe.roboflow.com/plat-kendaraan/vehicle-and-license-plate",
                "images": "~2200", "license": "MIT",
            },
            {
                "name": "Zenodo Indonesian License Plate (YOLOv11 format)",
                "url": "https://zenodo.org/records/15605718",
                "images": "126.8 MB", "license": "CC BY 4.0",
            },
            {
                "name": "Mendeley Central Asia Plates",
                "url": "https://data.mendeley.com/datasets/4c5z833m6m/2",
                "images": "500", "license": "CC BY 4.0",
            },
        ],
        "image_manifest": copied_images,
    }

    out_json = BASE_DIR / "dataset" / "dataset_info.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    log.info(f"  dataset_info.json disimpan: {out_json}")

    # ── LAPORAN FINAL ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  LAPORAN FINAL")
    log.info("=" * 60)
    log.info(f"  Total gambar terkumpul : {total_final}")
    log.info(f"  Target minimum (500)   : {'✓ TERPENUHI' if total_final >= 500 else '✗ BELUM TERPENUHI'}")
    log.info(f"  Duplikat dihapus       : {filter_stats['duplicate']}")
    log.info(f"  Gambar invalid dihapus : {filter_stats['invalid']}")
    log.info(f"\n  Distribusi kelas:")
    for cls, cnt in class_counters.items():
        bar = "█" * (cnt // 5)
        log.info(f"    {cls:15s}: {cnt:4d}  {bar}")
    log.info(f"\n  Sumber digunakan       : Bing Image Crawler")
    log.info(f"  Output folder          : dataset/images/")
    log.info(f"  Manifest               : dataset/dataset_info.json")
    log.info("\n  LANGKAH SELANJUTNYA:")
    log.info("  1. Anotasi manual menggunakan LabelImg:")
    log.info("       pip install labelImg && labelImg")
    log.info("  2. Atau upload ke Roboflow untuk auto-annotation")
    log.info("  3. Setelah anotasi, jalankan: python scripts/train.py")
    log.info("=" * 60)

    return total_final


if __name__ == "__main__":
    total = main()
    sys.exit(0 if total >= 500 else 1)
