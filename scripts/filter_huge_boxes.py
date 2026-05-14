"""
scripts/filter_huge_boxes.py — Hapus bounding box yang terlalu besar

Menjalankan:
  python scripts/filter_huge_boxes.py

Yang dilakukan:
  - Scan semua label .txt di dataset/labels/{train,val}/
  - Hapus baris di mana width > 0.80 ATAU height > 0.80 (normalized)
  - Jika semua label di file terhapus: hapus file label DAN gambarnya
  - Jika masih ada label valid: simpan ulang file label
  - Cetak laporan detail
"""

from pathlib import Path


# --- Konfigurasi -------------------------------------------------------------

DATASET      = Path("dataset")
SPLITS       = ["train", "val"]
HUGE_W_THRESH = 0.80   # bbox dianggap "huge" jika width > nilai ini
HUGE_H_THRESH = 0.80   # ATAU height > nilai ini

# -----------------------------------------------------------------------------

CLASS_NAMES = {0: "ambulance", 1: "police", 2: "fire_truck", 3: "license_plate"}


def filter_split(split: str) -> dict:
    lbl_dir = DATASET / "labels" / split
    img_dir = DATASET / "images" / split

    huge_removed      = 0
    files_modified    = 0
    images_deleted    = 0
    huge_by_class: dict[str, int] = {}

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        text = lbl_path.read_text().strip()
        if not text:
            continue

        valid_lines = []
        removed_lines = []

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                valid_lines.append(line)   # baris format salah — biarkan saja
                continue
            try:
                cid = int(parts[0])
                bw  = float(parts[3])
                bh  = float(parts[4])
            except ValueError:
                valid_lines.append(line)
                continue

            if bw > HUGE_W_THRESH or bh > HUGE_H_THRESH:
                removed_lines.append(line)
                huge_removed += 1
                cn = CLASS_NAMES.get(cid, f"class_{cid}")
                huge_by_class[cn] = huge_by_class.get(cn, 0) + 1
            else:
                valid_lines.append(line)

        if not removed_lines:
            continue   # tidak ada perubahan

        files_modified += 1

        if not valid_lines:
            # Tidak ada label tersisa — hapus label dan gambar
            lbl_path.unlink()
            img_path = img_dir / (lbl_path.stem + ".jpg")
            if img_path.exists():
                img_path.unlink()
                images_deleted += 1
        else:
            # Simpan ulang hanya baris yang valid
            lbl_path.write_text("\n".join(valid_lines))

    return {
        "huge_removed":   huge_removed,
        "files_modified": files_modified,
        "images_deleted": images_deleted,
        "huge_by_class":  huge_by_class,
    }


def count_files(split: str) -> tuple[int, int]:
    imgs = len(list((DATASET / "images" / split).glob("*.jpg")))
    lbls = len(list((DATASET / "labels" / split).glob("*.txt")))
    return imgs, lbls


def main():
    print("=" * 60)
    print("  Filter Huge Bounding Boxes")
    print(f"  Threshold: width > {HUGE_W_THRESH} OR height > {HUGE_H_THRESH}")
    print("=" * 60)

    grand_huge = grand_modified = grand_deleted = 0
    grand_by_class: dict[str, int] = {}

    for split in SPLITS:
        print(f"\n[{split.upper()}]")
        stats = filter_split(split)

        grand_huge     += stats["huge_removed"]
        grand_modified += stats["files_modified"]
        grand_deleted  += stats["images_deleted"]
        for cn, n in stats["huge_by_class"].items():
            grand_by_class[cn] = grand_by_class.get(cn, 0) + n

        imgs, lbls = count_files(split)
        print(f"  Huge boxes removed : {stats['huge_removed']}")
        print(f"  Label files edited : {stats['files_modified']}")
        print(f"  Images deleted     : {stats['images_deleted']}  (semua label terhapus)")
        print(f"  Sisa images        : {imgs}  labels: {lbls}")

    print()
    print("-" * 60)
    print(f"  Total huge boxes removed : {grand_huge}")
    print(f"  Breakdown by class       : {grand_by_class}")
    print(f"  Total label files edited : {grand_modified}")
    print(f"  Total images deleted     : {grand_deleted}")
    print()
    print("  Dataset akhir setelah filter:")
    for split in SPLITS:
        imgs, lbls = count_files(split)
        match = "OK" if imgs == lbls else "MISMATCH"
        print(f"  [{split}] images={imgs}  labels={lbls}  [{match}]")

    print("=" * 60)


if __name__ == "__main__":
    main()
