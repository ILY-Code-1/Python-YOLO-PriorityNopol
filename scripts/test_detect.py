"""
scripts/test_detect.py - Test script untuk endpoint POST /api/v1/detect

Cara pakai:
  python scripts/test_detect.py --image path/to/image.jpg
  python scripts/test_detect.py --image example_image.jpg --url http://localhost:2200

Jika --url tidak diberikan, default ke http://localhost:8000.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test endpoint deteksi kendaraan prioritas"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path ke file gambar (JPEG/PNG)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Kirim request N kali (untuk benchmark latency)",
    )
    return parser.parse_args()


def send_request(endpoint: str, image_path: Path) -> dict:
    """Kirim gambar ke endpoint dan kembalikan response dict."""
    try:
        import requests
    except ImportError:
        print("ERROR: library 'requests' tidak ditemukan.")
        print("Install dengan: pip install requests")
        sys.exit(1)

    with open(image_path, "rb") as f:
        files   = {"file": (image_path.name, f, "image/jpeg")}
        resp    = requests.post(endpoint, files=files, timeout=60)

    resp.raise_for_status()
    return resp.json()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: File tidak ditemukan: {image_path}")
        sys.exit(1)

    endpoint = args.url.rstrip("/") + "/api/v1/detect"

    print("=" * 55)
    print("  Test Deteksi Kendaraan Prioritas")
    print("=" * 55)
    print(f"  Gambar   : {image_path}")
    print(f"  Endpoint : {endpoint}")
    print(f"  Repeat   : {args.repeat}x")
    print()

    latencies = []

    for i in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"--- Request {i}/{args.repeat} ---")

        t0 = time.perf_counter()
        try:
            result = send_request(endpoint, image_path)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            sys.exit(1)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # Pretty print response
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  Latency: {elapsed*1000:.0f} ms")
        print()

    if args.repeat > 1:
        avg = sum(latencies) / len(latencies)
        mn  = min(latencies)
        mx  = max(latencies)
        print("=" * 55)
        print(f"  Latency avg : {avg*1000:.0f} ms")
        print(f"  Latency min : {mn*1000:.0f} ms")
        print(f"  Latency max : {mx*1000:.0f} ms")
        print("=" * 55)

    # Interpretasi hasil
    v = result.get("vehicle")
    p = result.get("plate_detected")
    n = result.get("plate_number", "")
    c = result.get("confidence", 0)

    print("HASIL:")
    if v is None:
        print("  Tidak ada kendaraan prioritas terdeteksi.")
    else:
        print(f"  Kendaraan : {v}  (conf={c})")
        if p:
            print(f"  Nopol     : {n}  [TERBACA]")
        else:
            print("  Nopol     : tidak terdeteksi / tidak terbaca")


if __name__ == "__main__":
    main()
