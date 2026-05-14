"""
scripts/verify_dataset.py — Sanity check dataset sebelum training

Menjalankan:
  python scripts/verify_dataset.py

Output:
  - Ringkasan total images per split
  - Rasio train/val
  - Distribusi kelas
  - Missing pairs (gambar tanpa label, label tanpa gambar)
  - Verdict: GO atau NOT READY
"""

from pathlib import Path


# --- Konfigurasi -------------------------------------------------------------

DATASET     = Path("dataset")
SPLITS      = ["train", "val"]
CLASS_NAMES = {0: "ambulance", 1: "police", 2: "fire_truck", 3: "license_plate"}

# Minimum box per kelas untuk verdict GO
MIN_BOXES = {"ambulance": 200, "police": 200, "fire_truck": 200, "license_plate": 400}

# -----------------------------------------------------------------------------


def scan_split(split: str) -> dict:
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split

    img_stems = {p.stem for p in img_dir.glob("*.jpg")}
    lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

    missing_labels = img_stems - lbl_stems   # gambar tanpa label
    orphan_labels  = lbl_stems - img_stems   # label tanpa gambar

    class_boxes  = {i: 0 for i in range(4)}
    class_images = {i: 0 for i in range(4)}
    empty_labels = 0
    bad_lines    = 0

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        text = lbl_path.read_text().strip()
        if not text:
            empty_labels += 1
            continue

        present = set()
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                bad_lines += 1
                continue
            try:
                cid = int(parts[0])
                float(parts[1]); float(parts[2]); float(parts[3]); float(parts[4])
            except ValueError:
                bad_lines += 1
                continue
            if cid in CLASS_NAMES:
                class_boxes[cid] += 1
                present.add(cid)

        for cid in present:
            class_images[cid] += 1

    return {
        "total_images":    len(img_stems),
        "total_labels":    len(lbl_stems),
        "missing_labels":  sorted(missing_labels)[:10],
        "missing_count":   len(missing_labels),
        "orphan_labels":   sorted(orphan_labels)[:10],
        "orphan_count":    len(orphan_labels),
        "empty_labels":    empty_labels,
        "bad_lines":       bad_lines,
        "class_boxes":     {CLASS_NAMES[k]: v for k, v in class_boxes.items()},
        "class_images":    {CLASS_NAMES[k]: v for k, v in class_images.items()},
    }


def main():
    SEP  = "=" * 60
    SEP2 = "-" * 60

    print(SEP)
    print("  Dataset Verification Report")
    print(SEP)

    results = {s: scan_split(s) for s in SPLITS}
    tr = results["train"]
    vl = results["val"]

    total_imgs  = tr["total_images"] + vl["total_images"]
    train_pct   = 100 * tr["total_images"] // max(total_imgs, 1)
    val_pct     = 100 - train_pct

    # -- 1. Overview --------------------------------------------------------
    print()
    print("1. OVERVIEW")
    print(SEP2)
    print(f"  Total images  : {total_imgs}")
    print(f"  Train         : {tr['total_images']}  ({train_pct}%)")
    print(f"  Val           : {vl['total_images']}  ({val_pct}%)")
    split_ok = abs(train_pct - 80) <= 7
    print(f"  Split ratio   : {'OK' if split_ok else 'WARNING: jauh dari 80/20'}")

    # -- 2. Class Distribution ---------------------------------------------
    print()
    print("2. CLASS DISTRIBUTION")
    print(SEP2)
    print(f"  {'Class':15s}  {'Train boxes':>12}  {'Val boxes':>10}  {'Total':>7}  {'Min req':>8}  Status")
    print("  " + "-" * 60)
    all_class_ok = True
    total_boxes_all = 0
    for cn in ["ambulance", "police", "fire_truck", "license_plate"]:
        tr_b = tr["class_boxes"][cn]
        vl_b = vl["class_boxes"][cn]
        tot  = tr_b + vl_b
        req  = MIN_BOXES[cn]
        ok   = tot >= req
        if not ok:
            all_class_ok = False
        flag = "" if ok else "  <- KURANG"
        print(f"  {cn:15s}  {tr_b:>12}  {vl_b:>10}  {tot:>7}  {req:>8}  {'OK' if ok else 'LOW'}{flag}")
        total_boxes_all += tot
    print(f"  {'TOTAL':15s}  {'':>12}  {'':>10}  {total_boxes_all:>7}")

    # -- 3. Label Integrity ------------------------------------------------
    print()
    print("3. LABEL INTEGRITY")
    print(SEP2)
    integrity_ok = True
    for split in SPLITS:
        r = results[split]
        spl = split.upper()
        print(f"  [{spl}]")
        print(f"    Images without label : {r['missing_count']}", end="")
        if r["missing_count"]:
            integrity_ok = False
            print(f"  <- PROBLEM: {r['missing_labels'][:3]}...")
        else:
            print("  OK")

        print(f"    Labels without image : {r['orphan_count']}", end="")
        if r["orphan_count"]:
            integrity_ok = False
            print(f"  <- PROBLEM: {r['orphan_labels'][:3]}...")
        else:
            print("  OK")

        print(f"    Empty label files    : {r['empty_labels']}  (background samples)")
        print(f"    Malformed lines      : {r['bad_lines']}", end="")
        if r["bad_lines"]:
            integrity_ok = False
            print("  <- PROBLEM")
        else:
            print("  OK")

    # -- 4. Co-occurrence Check --------------------------------------------
    print()
    print("4. CO-OCCURRENCE CHECK (train)")
    print(SEP2)
    lbl_dir = DATASET / "labels" / "train"
    co_both = co_v_only = co_lp_only = 0
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        text = lbl_path.read_text().strip()
        if not text:
            continue
        cids = {int(l.split()[0]) for l in text.splitlines()
                if len(l.split()) == 5 and l.split()[0].isdigit()}
        has_v  = bool(cids & {0, 1, 2})
        has_lp = 3 in cids
        if has_v and has_lp:
            co_both += 1
        elif has_v:
            co_v_only += 1
        elif has_lp:
            co_lp_only += 1

    total_lbl = co_both + co_v_only + co_lp_only
    pct_both = 100 * co_both // max(total_lbl, 1)
    co_ok = co_both > 0

    print(f"  Vehicle + plate  : {co_both:5d}  ({pct_both}%)")
    print(f"  Vehicle only     : {co_v_only:5d}")
    print(f"  Plate only       : {co_lp_only:5d}")
    if not co_ok:
        print("  WARNING: Masih 0 gambar dengan kedua anotasi!")
        integrity_ok = False
    else:
        print(f"  OK: {co_both} gambar co-annotated ({pct_both}%)")

    # -- Verdict -----------------------------------------------------------
    print()
    print(SEP)
    go = integrity_ok and all_class_ok and split_ok and co_ok
    if go:
        print("  VERDICT: GO — Dataset siap untuk training!")
        print("  Jalankan: python scripts/train_vehicle.py")
    else:
        print("  VERDICT: NOT READY — Ada masalah yang perlu diperbaiki:")
        if not integrity_ok:
            print("    - Label integrity: ada missing pair atau baris rusak")
        if not all_class_ok:
            print("    - Class distribution: ada kelas kurang dari minimum")
        if not split_ok:
            print("    - Split ratio: tidak sesuai 80/20")
        if not co_ok:
            print("    - Co-annotation: jalankan scripts/augment_nopol.py")
    print(SEP)


if __name__ == "__main__":
    main()
