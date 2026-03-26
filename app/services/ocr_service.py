"""
services/ocr_service.py - Service untuk OCR menggunakan EasyOCR
Bertanggung jawab untuk:
  - Inisialisasi EasyOCR reader (singleton)
  - Preprocess crop plat nomor agar OCR lebih akurat
  - Baca dan bersihkan teks nomor polisi

Format output yang diharapkan: "B1234XYZ" (tanpa spasi, uppercase)
"""

import re
import numpy as np
import cv2
import easyocr


class OCRService:
    """
    Service OCR untuk membaca nomor polisi dari crop gambar plat.
    EasyOCR di-inisialisasi sekali karena proses load model cukup berat.
    """

    def __init__(self, languages: list = None):
        """
        Args:
            languages: Kode bahasa EasyOCR. Default ['en'] cukup untuk
                       plat nomor Indonesia (karakter latin alphanumeric).
        """
        if languages is None:
            languages = ["en"]

        print(f"[OCRService] Inisialisasi EasyOCR, bahasa: {languages}")
        # gpu=False → kompatibel semua environment. Set True jika ada CUDA.
        self.reader = easyocr.Reader(
            languages, 
            gpu=False, 
            model_storage_directory="/tmp/easyocr",
            user_network_directory="/tmp/easyocr"
        )

    # ─── Internal: preprocess gambar ────────────────────────────────────────

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Tingkatkan kualitas gambar crop plat sebelum dikirim ke OCR.

        Pipeline:
          1. Resize ke tinggi 64px (lebar proporsional, min 100px)
          2. Grayscale
          3. CLAHE – normalisasi kontras adaptif
          4. Threshold Otsu – binarisasi untuk mempertegas teks
        """
        h, w = image_bgr.shape[:2]
        if h > 0:
            scale  = 64 / h
            new_w  = max(int(w * scale), 100)
            image_bgr = cv2.resize(
                image_bgr, (new_w, 64), interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, thresh = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresh

    # ─── Internal: bersihkan teks hasil OCR ─────────────────────────────────

    def _clean_plate_text(self, raw_text: str) -> str:
        """
        Bersihkan teks OCR menjadi format nomor polisi Indonesia standar.

        Transformasi:
          "b 1234 xyz"   → "B1234XYZ"
          " B 1234 XYZ " → "B1234XYZ"
          "B.1234-XYZ"   → "B1234XYZ"

        Steps:
          1. Uppercase
          2. Hapus semua karakter selain huruf dan angka (termasuk spasi)
          3. Strip sisa whitespace
        """
        # Uppercase lalu buang semua karakter non-alfanumerik (termasuk spasi)
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        return cleaned.strip()

    # ─── Public: baca plat nomor ─────────────────────────────────────────────

    def read_plate(self, image_crop_bgr: np.ndarray) -> str:
        """
        Baca teks nomor polisi dari crop gambar plat.

        Args:
            image_crop_bgr: Crop plat nomor dalam format BGR numpy array.

        Returns:
            String nomor polisi (contoh: "B1234XYZ"), atau string kosong ""
            jika tidak berhasil dibaca. Tidak pernah return None.
        """
        if image_crop_bgr is None or image_crop_bgr.size == 0:
            return ""

        processed = self._preprocess(image_crop_bgr)

        # EasyOCR: detail=1 agar dapat confidence per token
        ocr_results = self.reader.readtext(processed, detail=1, paragraph=False)

        if not ocr_results:
            return ""

        # Kumpulkan token dengan confidence ≥ 0.40
        tokens = [
            text
            for (_, text, conf) in ocr_results
            if conf >= 0.40
        ]

        if not tokens:
            return ""

        # Gabung semua token lalu bersihkan
        cleaned = self._clean_plate_text(" ".join(tokens))
        return cleaned  # sudah dijamin str (bisa "" jika kosong setelah cleaning)
