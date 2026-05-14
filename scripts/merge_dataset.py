"""
scripts/merge_dataset.py — Merge dataset_aug/ into dataset/

Menjalankan:
  python scripts/merge_dataset.py

Yang dilakukan:
  - Salin semua gambar dan label dari dataset_aug/ ke dataset/
  - Jika nama file sudah ada, tambahkan suffix _aug sebelum ekstensi
  - Cetak ringkasan jumlah file yang disalin dan di-rename
"""

import shutil
from pathlib import Path


# ─── Konfigurasi ─────────────────────────────────────────────────────────────

SRC  = Path("dataset_aug")
DST  = Path("dataset")
SPLITS = ["train", "val"]

# ─────────────────────────────────────────────────────────────────────────────


def safe_copy(src_file: Path, dst_dir: Path) -> tuple[Path, bool]:
    """
    Salin src_file ke dst_dir.
    Jika nama sudah ada, tambahkan suffix _aug sebelum ekstensi.
    Return (dst_path, was_renamed).
    """
    dst_path = dst_dir / src_file.name
    renamed  = False

    if dst_path.exists():
        stem    = src_file.stem + "_aug"
        suffix  = src_file.suffix
        dst_path = dst_dir / (stem + suffix)
        renamed  = True

        # Jika masih konflik setelah _aug, tambahkan counter
        counter = 1
        while dst_path.exists():
            dst_path = dst_dir / (f"{stem}_{counter}{suffix}")
            counter += 1

    shutil.copy2(src_file, dst_path)
    return dst_path, renamed


def merge_split(split: str) -> dict:
    src_img_dir = SRC / "images" / split
    src_lbl_dir = SRC / "labels" / split
    dst_img_dir = DST / "images" / split
    dst_lbl_dir = DST / "labels" / split

    if not src_img_dir.exists():
        print(f"  [{split}] Sumber tidak ada: {src_img_dir} — dilewati.")
        return {"copied": 0, "renamed": 0, "skipped": 0}

    src_images = sorted(src_img_dir.glob("*.jpg"))
    copied = renamed = skipped = 0

    for img_src in src_images:
        lbl_src = src_lbl_dir / (img_src.stem + ".txt")

        # Wajib ada pasangan label
        if not lbl_src.exists():
            skipped += 1
            continue

        img_dst, img_renamed = safe_copy(img_src, dst_img_dir)

        # Label harus menggunakan stem yang sama dengan gambar tujuan
        lbl_dst_name = img_dst.stem + ".txt"
        lbl_dst      = dst_lbl_dir / lbl_dst_name

        # Salin label dengan nama yang sudah disesuaikan
        shutil.copy2(lbl_src, lbl_dst)

        copied  += 1
        renamed += int(img_renamed)

    return {"copied": copied, "renamed": renamed, "skipped": skipped}


def count_files(split: str) -> tuple[int, int]:
    img_count = len(list((DST / "images" / split).glob("*.jpg")))
    lbl_count = len(list((DST / "labels" / split).glob("*.txt")))
    return img_count, lbl_count


def main():
    print("=" * 60)
    print("  Merge dataset_aug/ -> dataset/")
    print("=" * 60)

    total_copied = total_renamed = total_skipped = 0

    for split in SPLITS:
        print(f"\n[{split.upper()}]")
        stats = merge_split(split)
        total_copied  += stats["copied"]
        total_renamed += stats["renamed"]
        total_skipped += stats["skipped"]
        print(f"  Disalin       : {stats['copied']}")
        print(f"  Di-rename     : {stats['renamed']}  (konflik nama, suffix _aug)")
        print(f"  Dilewati      : {stats['skipped']}  (tidak ada pasangan label)")

    print()
    print("-" * 60)
    print(f"  Total disalin    : {total_copied} pasang (gambar + label)")
    print(f"  Total di-rename  : {total_renamed}")
    print(f"  Total dilewati   : {total_skipped}")
    print()
    print("  Dataset akhir setelah merge:")
    for split in SPLITS:
        imgs, lbls = count_files(split)
        match = "OK" if imgs == lbls else "MISMATCH"
        print(f"  [{split}] images={imgs}  labels={lbls}  [{match}]")

    print("=" * 60)


if __name__ == "__main__":
    main()
