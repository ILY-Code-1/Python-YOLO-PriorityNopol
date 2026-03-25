"""
utils/image_utils.py - Utilitas pemrosesan gambar
Fungsi-fungsi helper untuk:
  - Decode bytes gambar → numpy array BGR
  - Crop region dari gambar berdasarkan bounding box
  - Resize dengan menjaga aspek rasio
"""

from typing import Optional, List
import numpy as np
import cv2


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode bytes gambar (dari upload) menjadi numpy array format BGR (OpenCV).

    Args:
        image_bytes: Raw bytes dari file gambar (JPEG/PNG).

    Returns:
        numpy array BGR, atau None jika gagal decode.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"[image_utils] Gagal decode gambar: {e}")
        return None


def crop_region(
    image_bgr: np.ndarray,
    bbox: List[int],
    padding: int = 4,
) -> Optional[np.ndarray]:
    """
    Crop area dari gambar berdasarkan bounding box.
    Menambahkan padding kecil agar teks plat tidak terpotong tepi.

    Args:
        image_bgr : Gambar asli numpy array BGR.
        bbox      : List [x1, y1, x2, y2] koordinat piksel.
        padding   : Piksel padding di setiap sisi (default 4).

    Returns:
        Crop numpy array BGR, atau None jika koordinat tidak valid.
    """
    if image_bgr is None:
        return None

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        print(f"[image_utils] Bounding box tidak valid: {bbox}")
        return None

    return image_bgr[y1:y2, x1:x2].copy()


def resize_keep_aspect(
    image_bgr: np.ndarray,
    target_width: int = 640,
) -> np.ndarray:
    """
    Resize gambar ke lebar tertentu dengan menjaga aspek rasio.
    Berguna untuk preprocessing sebelum YOLO inference pada gambar besar.

    Args:
        image_bgr    : Gambar input numpy array BGR.
        target_width : Lebar target dalam piksel (default 640).

    Returns:
        Gambar yang sudah di-resize.
    """
    h, w = image_bgr.shape[:2]
    if w == 0:
        return image_bgr
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image_bgr, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
