"""
scripts/fix_kpt_shape.py
─────────────────────────────────────────────────────────────────────────────
Fix nhãn YOLO-Pose bị export từ Roboflow với 3 vấn đề:
  1. kpt_shape sai ([6,3] thay vì [4,3]) → 2 keypoint thừa "0 0 0"
  2. flip_idx sai ([0,1,2,3] → cần [1,0,3,2] cho 4 góc thẻ)
  3. class_id theo Roboflow index cần remap về class_id theo định nghĩa mentor

Cách dùng:
    # Preview không ghi file
    python scripts/fix_kpt_shape.py --data data_multiclasses/MultipleCard-Detect.yolo26 --dry_run

    # Fix thật (sau khi đã preview OK)
    python scripts/fix_kpt_shape.py --data data_multiclasses/MultipleCard-Detect.yolo26
"""

import sys
import argparse
from pathlib import Path

import yaml


# ── Class mapping: Roboflow index → Mentor class_id ──────────────────────────
# Đọc names từ data.yaml của Roboflow, map sang class_id đúng theo mentor spec.
# Key   = tên class trên Roboflow (chính xác chữ thường/hoa)
# Value = class_id đúng theo mentor (src/detector/base.py → CLASS_NAMES)
ROBOFLOW_NAME_TO_MENTOR_ID = {
    "cmnd_ms_c":  0,
    "cmnd_mt_c":  1,
    "cccd_ms_c":  2,
    "cccd_mt_c":  3,
    "cccd_ms_m":  4,
    "cccd_mt_m":  5,
    "blx_ms":     6,
    "blx_mt":     7,
    "hc_mt":      8,
    "other":      9,
    "print":      10,
    "hc_nn":      11,
    "sg_mt":      12,
    "sgp_mt":     12,   # alias Roboflow dùng
    "sg_ms":      13,
    "sgp_ms":     13,   # alias Roboflow dùng
}

CORRECT_FLIP_IDX = [1, 0, 3, 2]   # TL<->TR (0<->1), BL<->BR (3<->2)
KEEP_KPTS        = 4               # Số keypoint cần giữ
BBOX_FIELDS      = 5               # class_id + cx + cy + w + h


def build_remap_table(roboflow_names: list) -> dict:
    """Tạo bảng remap: {roboflow_index → mentor_class_id}.

    roboflow_names: list tên class theo thứ tự trong data.yaml Roboflow.
    """
    remap = {}
    unmapped = []
    for rb_idx, rb_name in enumerate(roboflow_names):
        mentor_id = ROBOFLOW_NAME_TO_MENTOR_ID.get(rb_name.lower())
        if mentor_id is None:
            mentor_id = ROBOFLOW_NAME_TO_MENTOR_ID.get(rb_name)
        if mentor_id is not None:
            remap[rb_idx] = mentor_id
        else:
            unmapped.append((rb_idx, rb_name))
    return remap, unmapped


def fix_label_file(txt_path: Path, remap: dict,
                   keep_kpts: int, dry_run: bool) -> str:
    """Fix 1 file label: remap class_id + strip keypoints thừa.

    Returns: 'ok' | 'fixed' | 'error'
    """
    fields_needed = BBOX_FIELDS + keep_kpts * 3
    lines_in  = txt_path.read_text(encoding="utf-8").strip().splitlines()
    lines_out = []
    changed   = False

    for line in lines_in:
        parts = line.split()
        if not parts:
            continue

        # ── Bước 1: Remap class_id ────────────────────────────────────────
        try:
            rb_class = int(parts[0])
            new_class = remap.get(rb_class, rb_class)  # giữ nguyên nếu không có map
            if new_class != rb_class:
                parts[0] = str(new_class)
                changed = True
        except ValueError:
            pass

        # ── Bước 2: Strip keypoints thừa ─────────────────────────────────
        if len(parts) > fields_needed:
            parts = parts[:fields_needed]
            changed = True
        elif len(parts) < fields_needed and len(parts) >= BBOX_FIELDS:
            print(f"  WARNING {txt_path.name}: {len(parts)} truong < {fields_needed} can thiet")

        lines_out.append(" ".join(parts))

    if not changed:
        return "ok"

    if not dry_run:
        txt_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return "fixed"


def fix_data_yaml(yaml_path: Path, roboflow_names: list,
                  remap: dict, keep_kpts: int, dry_run: bool) -> None:
    """Sửa data.yaml: kpt_shape, flip_idx, và names theo mentor class_id."""
    if not yaml_path.exists():
        print(f"  WARNING: Khong tim thay {yaml_path}")
        return

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    changes = []

    # Sửa kpt_shape
    if data.get("kpt_shape") != [keep_kpts, 3]:
        changes.append(f"kpt_shape: {data.get('kpt_shape')} -> [{keep_kpts}, 3]")
        data["kpt_shape"] = [keep_kpts, 3]

    # Sửa flip_idx
    if data.get("flip_idx") != CORRECT_FLIP_IDX:
        changes.append(f"flip_idx: {data.get('flip_idx')} -> {CORRECT_FLIP_IDX}")
        data["flip_idx"] = CORRECT_FLIP_IDX

    # Rebuild names dict theo mentor class_id
    # Chỉ include các class có trong dataset này
    mentor_names = {}
    for rb_idx, rb_name in enumerate(roboflow_names):
        mentor_id = remap.get(rb_idx)
        if mentor_id is not None:
            mentor_names[mentor_id] = rb_name
    if mentor_names:
        data["names"] = mentor_names
        data["nc"] = max(mentor_names.keys()) + 1
        changes.append(f"names remapped -> {mentor_names}")

    if not changes:
        print("  data.yaml da dung, khong thay doi")
        return

    for c in changes:
        print(f"  -> {c}")

    if not dry_run:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True,
                      default_flow_style=None, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Fix YOLO-Pose labels: kpt_shape, flip_idx, class_id remap"
    )
    parser.add_argument(
        "--data", required=True,
        help="Thu muc goc dataset Roboflow (chua data.yaml va train/test/valid)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Preview ma khong ghi file"
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: Khong tim thay thu muc: {data_dir}")
        sys.exit(1)

    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"ERROR: Khong tim thay data.yaml tai {yaml_path}")
        sys.exit(1)

    # Đọc names hiện tại từ Roboflow data.yaml
    with open(yaml_path, encoding="utf-8") as f:
        rb_data = yaml.safe_load(f)

    rb_names_raw = rb_data.get("names", [])
    if isinstance(rb_names_raw, dict):
        rb_names = [rb_names_raw[k] for k in sorted(rb_names_raw.keys())]
    else:
        rb_names = list(rb_names_raw)

    remap, unmapped = build_remap_table(rb_names)

    print(f"\n{'='*60}")
    print(f"  Fix YOLO-Pose Labels")
    if args.dry_run:
        print(f"  [DRY RUN] - khong ghi file")
    print(f"  Dataset : {data_dir}")
    print(f"\n  Roboflow classes -> Mentor class_id:")
    for rb_idx, rb_name in enumerate(rb_names):
        mentor_id = remap.get(rb_idx, "???")
        print(f"    [{rb_idx}] {rb_name:<16} -> class_id {mentor_id}")
    if unmapped:
        print(f"\n  WARNING: Cac class chua duoc map:")
        for rb_idx, rb_name in unmapped:
            print(f"    [{rb_idx}] {rb_name}")
    print(f"{'='*60}\n")

    # Fix data.yaml
    print("[data.yaml]")
    fix_data_yaml(yaml_path, rb_names, remap, KEEP_KPTS, args.dry_run)

    # Fix tất cả .txt label files
    stats = {"ok": 0, "fixed": 0, "total": 0}
    label_dirs = sorted(data_dir.rglob("labels"))
    if not label_dirs:
        label_dirs = [data_dir]

    for ldir in label_dirs:
        txt_files = sorted(ldir.glob("*.txt"))
        if not txt_files:
            continue
        split_name = ldir.parent.name  # train / test / valid
        print(f"\n[Labels/{split_name}]  ({len(txt_files)} files)")
        for txt_path in txt_files:
            status = fix_label_file(txt_path, remap, KEEP_KPTS, args.dry_run)
            stats[status] = stats.get(status, 0) + 1
            stats["total"] += 1
            if status == "fixed":
                print(f"  Fixed: {txt_path.name}")

    # Báo cáo
    print(f"\n{'='*60}")
    print(f"  Ket qua: {stats['total']} files kiem tra")
    print(f"    Da dung: {stats.get('ok', 0)}")
    print(f"    Da fix : {stats.get('fixed', 0)}")
    if args.dry_run:
        print(f"\n  Chay lai KHONG co --dry_run de ghi that")
    else:
        print(f"\n  HOAN THANH! kpt_shape -> [4, 3], class_id da remap")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
