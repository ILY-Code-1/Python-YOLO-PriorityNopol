"""
scripts/clean_nopol_labels.py - Bersihkan label license_plate yang salah posisi.

Filter label kelas 3 (license_plate) dari dataset/labels/{train,val}/ berdasarkan
aturan geometri. Label kelas 0,1,2 tidak diubah.

Kondisi HAPUS baris license_plate jika ANY benar:
  cy < 0.40     -> plat center terlalu tinggi (reflector/lampu)
  cy > 0.95     -> plat center terlalu rendah (keluar frame)
  width < 0.04  -> terlalu sempit
  width > 0.60  -> terlalu lebar
  height < 0.01 -> terlalu tipis
  height > 0.25 -> terlalu tinggi
  w/h < 1.5     -> aspek ratio salah (bukan plat horizontal)

Cara pakai:
  python scripts/clean_nopol_labels.py
"""

from pathlib import Path

LABEL_DIRS = [
    Path("dataset/labels/train"),
    Path("dataset/labels/val"),
]

LICENSE_PLATE_CLASS_ID = 3

REASONS = [
    ("cy < 0.40",    lambda cy, w, h: cy < 0.40),
    ("cy > 0.95",    lambda cy, w, h: cy > 0.95),
    ("width < 0.04", lambda cy, w, h: w < 0.04),
    ("width > 0.60", lambda cy, w, h: w > 0.60),
    ("height < 0.01",lambda cy, w, h: h < 0.01),
    ("height > 0.25",lambda cy, w, h: h > 0.25),
    ("w/h < 1.5",    lambda cy, w, h: h > 0 and (w / h) < 1.5),
]


def is_bad_plate(parts: list[str]) -> tuple[bool, str]:
    """Return (bad, reason) for a single label line of class 3."""
    try:
        _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except (ValueError, IndexError):
        return True, "parse_error"

    for reason_str, check in REASONS:
        if check(cy, w, h):
            return True, reason_str
    return False, ""


def main():
    reason_counts: dict[str, int] = {}
    total_plate_lines = 0
    removed_plate_lines = 0
    files_modified = 0
    files_all_plates_removed = 0

    for label_dir in LABEL_DIRS:
        if not label_dir.exists():
            print(f"[WARN] Dir tidak ditemukan: {label_dir}")
            continue

        label_files = sorted(label_dir.glob("*.txt"))
        print(f"\n[SCAN] {label_dir}  ({len(label_files)} file)")

        for lbl_path in label_files:
            text = lbl_path.read_text().strip()
            if not text:
                continue

            lines = text.splitlines()
            kept_lines = []
            file_removed = 0

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    kept_lines.append(line)
                    continue

                if int(parts[0]) != LICENSE_PLATE_CLASS_ID:
                    kept_lines.append(line)
                    continue

                total_plate_lines += 1
                bad, reason = is_bad_plate(parts)

                if bad:
                    removed_plate_lines += 1
                    file_removed += 1
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                else:
                    kept_lines.append(line)

            if file_removed > 0:
                files_modified += 1
                lbl_path.write_text("\n".join(kept_lines))

                plate_lines_remaining = sum(
                    1 for l in kept_lines
                    if l.strip().split() and int(l.strip().split()[0]) == LICENSE_PLATE_CLASS_ID
                )
                if plate_lines_remaining == 0:
                    files_all_plates_removed += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  HASIL CLEAN NOPOL LABELS")
    print("=" * 60)
    print(f"  Total baris license_plate ditemukan : {total_plate_lines}")
    print(f"  Baris dihapus (bad bbox)            : {removed_plate_lines}")
    print(f"  Baris dipertahankan                 : {total_plate_lines - removed_plate_lines}")
    print(f"  File label dimodifikasi             : {files_modified}")
    print(f"  File tanpa plat setelah clean       : {files_all_plates_removed}")
    print()
    print("  Breakdown alasan hapus:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = count / total_plate_lines * 100 if total_plate_lines > 0 else 0
        print(f"    {reason:<15} : {count:>5}  ({pct:.1f}%)")

    pct_removed = removed_plate_lines / total_plate_lines * 100 if total_plate_lines > 0 else 0
    print()
    print(f"  {pct_removed:.1f}% label plate dihapus sebagai outlier geometri.")
    print("=" * 60)
    print()
    print("  Langkah berikutnya:")
    print("  1. Jalankan: python scripts/train_nopol.py")
    print("     (Script akan rebuild dataset_nopol/ dari label yang sudah bersih)")
    print("=" * 60)


if __name__ == "__main__":
    main()
