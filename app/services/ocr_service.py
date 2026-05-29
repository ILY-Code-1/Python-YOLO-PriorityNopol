"""
services/ocr_service.py - Service OCR menggunakan EasyOCR

Bertanggung jawab untuk:
  - Inisialisasi EasyOCR reader (singleton, load sekali)
  - Preprocess crop plat nomor agar OCR lebih akurat
  - Baca dan bersihkan teks nomor polisi

Format output yang diharapkan: "B1234XYZ" (tanpa spasi, uppercase)
"""

import logging
import re

import cv2
import easyocr
import numpy as np

from app.config import OCR_GPU, OCR_LANG

log = logging.getLogger(__name__)

# Tinggi target (px) saat upscale crop plat sebelum OCR.
# Plat kecil (mis. fire_truck) sering tak terbaca; upscale ke 120px membantu.
OCR_TARGET_HEIGHT = 120

# Kernel sharpening (high-pass) untuk mempertegas tepi karakter setelah CLAHE.
SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

# Token OCR di bawah confidence ini diabaikan.
MIN_TOKEN_CONFIDENCE = 0.10

# Token mirip-tanggal (stiker masa berlaku plat, mis. "07.22" / "J7.2]") punya
# pola digit-pemisah-digit. Token plat bersih ('B', '7564', 'FDA') tidak. Dipakai
# untuk membuang stiker agar tidak ikut tergabung ke nomor plat.
EXPIRY_TOKEN_RE = re.compile(r"\d[.,/\-\[\]]\d")


class OCRService:
    """
    Service OCR untuk membaca nomor polisi dari crop gambar plat.
    EasyOCR di-inisialisasi sekali karena proses load model cukup berat.
    """

    def __init__(self) -> None:
        languages = [OCR_LANG] if isinstance(OCR_LANG, str) else OCR_LANG
        log.info("[OCRService] Inisialisasi EasyOCR  bahasa=%s  gpu=%s", languages, OCR_GPU)
        self.reader = easyocr.Reader(
            languages,
            gpu=OCR_GPU,
            model_storage_directory="/tmp/easyocr",
            user_network_directory="/tmp/easyocr",
        )
        log.info("[OCRService] EasyOCR siap.")

    # ─── Internal: helper resize & threshold ─────────────────────────────────

    def _upscale(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Resize crop plat ke tinggi OCR_TARGET_HEIGHT (lebar proporsional, min 100px).
        Karakter plat yang lebih besar lebih mudah dibaca EasyOCR.
        """
        h, w = image_bgr.shape[:2]
        if h <= 0:
            return image_bgr
        scale = OCR_TARGET_HEIGHT / h
        new_w = max(int(w * scale), 100)
        return cv2.resize(
            image_bgr, (new_w, OCR_TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC
        )

    def _auto_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Binarisasi Otsu dengan auto-deteksi polaritas plat.

        Plat Indonesia bisa hitam-di-putih ATAU putih-di-hitam. Kita coba
        kedua arah threshold lalu pilih yang piksel PUTIH-nya lebih banyak
        dari piksel hitam — sehingga teks (foreground) konsisten untuk OCR.
        """
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white = cv2.countNonZero(binary)
        if white * 2 >= binary.size:
            return binary
        # Lebih banyak hitam → balik agar foreground jadi putih
        _, binary_inv = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary_inv

    # ─── Internal: variasi preprocessing ─────────────────────────────────────

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Variant STANDARD.

        Pipeline:
          1. Upscale ke tinggi 120px (lebar proporsional, min 100px)
          2. Bilateral filter — kurangi noise tapi jaga ketajaman tepi
          3. Grayscale
          4. CLAHE (clipLimit=3.0) — normalisasi kontras adaptif
          5. Sharpening (filter2D high-pass) — pertegas tepi karakter
          6. Threshold Otsu dengan auto-deteksi polaritas plat
        """
        upscaled  = self._upscale(image_bgr)
        denoised  = cv2.bilateralFilter(upscaled, 9, 75, 75)
        gray      = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

        clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced  = clahe.apply(gray)

        sharpened = cv2.filter2D(enhanced, -1, SHARPEN_KERNEL)
        return self._auto_threshold(sharpened)

    def _preprocess_high_contrast(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Variant HIGH CONTRAST.

        Tanpa bilateral filter; CLAHE agresif (clipLimit=5.0) untuk plat
        yang pudar / kontras rendah. Diakhiri threshold Otsu auto-polaritas.
        """
        upscaled = self._upscale(image_bgr)
        gray     = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        clahe    = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return self._auto_threshold(enhanced)

    def _preprocess_raw(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Variant RAW.

        Hanya upscale ke 120px + grayscale, TANPA threshold. Berguna saat
        binarisasi justru merusak karakter (mis. plat dengan bayangan/gradien).
        """
        upscaled = self._upscale(image_bgr)
        return cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # ─── Internal: OCR multi-variant ─────────────────────────────────────────

    def _try_multiple_preprocess(self, image_bgr: np.ndarray) -> str:
        """
        Jalankan OCR pada 3 variant preprocessing dan pilih hasil terbaik.

        Variants:
          1. standard      — upscale → bilateral → CLAHE → sharpen → otsu
          2. high_contrast — upscale → CLAHE(5.0) → otsu (tanpa bilateral)
          3. raw           — upscale → grayscale (tanpa threshold)

        Untuk tiap variant, OCR dijalankan dan token (conf >= MIN_TOKEN_CONFIDENCE)
        dikumpulkan. Variant dengan rata-rata confidence token TERTINGGI dipilih.

        Returns:
            String gabungan token mentah (belum dibersihkan), atau "" jika
            tidak ada variant yang menghasilkan token.
        """
        variants = [
            ("standard",      self._preprocess(image_bgr)),
            ("high_contrast", self._preprocess_high_contrast(image_bgr)),
            ("raw",           self._preprocess_raw(image_bgr)),
        ]

        best_text = ""
        best_avg_conf = -1.0

        for name, processed in variants:
            if processed is None or processed.size == 0:
                continue

            ocr_results = self.reader.readtext(processed, detail=1, paragraph=False)
            tokens = [
                (text, float(conf), int(bbox[0][0]))   # bbox[0][0] = top-left x
                for (bbox, text, conf) in ocr_results
                if conf >= MIN_TOKEN_CONFIDENCE
            ]
            if not tokens:
                continue

            # Urutkan token kiri→kanan sesuai posisi x agar terbaca sesuai urutan
            # pada plat (EasyOCR kadang mengembalikan token tidak berurutan).
            tokens.sort(key=lambda t: t[2])   # sort by x1, left-to-right

            # Buang token stiker masa berlaku (mis. "07.22") — EasyOCR memecah
            # plat jadi beberapa token ('B', '7564', 'FDA'), jadi kita GABUNG
            # token plat tapi singkirkan token mirip-tanggal.
            plate_tokens = [
                (text, conf) for text, conf, x in tokens
                if not EXPIRY_TOKEN_RE.search(text)
            ]
            if not plate_tokens:
                continue

            avg_conf = sum(conf for _, conf in plate_tokens) / len(plate_tokens)
            joined   = " ".join(text for text, _ in plate_tokens)
            log.info(
                "[OCRService] Variant '%s': '%s'  avg_conf=%.3f",
                name, joined, avg_conf,
            )

            if avg_conf > best_avg_conf:
                best_avg_conf = avg_conf
                best_text     = joined

        return best_text

    # ─── Internal: bersihkan teks hasil OCR ─────────────────────────────────

    def _clean_plate_text(self, raw_text: str) -> str:
        """
        Bersihkan teks OCR menjadi format nomor polisi Indonesia standar.

        Contoh:
          "b 1234 xyz"   -> "B1234XYZ"
          " B 1234 XYZ " -> "B1234XYZ"
          "B.1234-XYZ"   -> "B1234XYZ"
        """
        return re.sub(r"[^A-Z0-9]", "", raw_text.upper()).strip()

    # ─── Public: baca plat nomor ─────────────────────────────────────────────

    def read_plate(self, image_crop_bgr: np.ndarray) -> str:
        """
        Baca teks nomor polisi dari crop gambar plat.

        Args:
            image_crop_bgr: Crop plat nomor dalam format BGR numpy array.

        Returns:
            String nomor polisi (contoh: "B1234XYZ"), atau "" jika gagal /
            hasil tidak valid. Tidak pernah return None.
        """
        if image_crop_bgr is None or image_crop_bgr.size == 0:
            return ""

        raw = self._try_multiple_preprocess(image_crop_bgr)
        if not raw:
            return ""

        cleaned = self._clean_plate_text(raw)

        # ── Validasi hasil ───────────────────────────────────────────────────
        if len(cleaned) < 4:
            log.info(
                "[OCRService] Hasil ditolak: terlalu pendek ('%s')", cleaned
            )
            return ""
        if not any(ch.isdigit() for ch in cleaned):
            log.info(
                "[OCRService] Hasil ditolak: tidak ada angka ('%s')", cleaned
            )
            return ""

        log.info("[OCRService] Hasil OCR: '%s'", cleaned)
        return cleaned
