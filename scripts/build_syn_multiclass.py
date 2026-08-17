"""
scripts/build_syn_multiclass.py
═══════════════════════════════════════════════════════════════════════════════
Sinh synthetic dataset multi-class từ clean templates đã extract.

Phân chia train/val theo TEMPLATE (không random split):
  - Train: synthetic từ template 1,2,3
  - Val:   synthetic từ template 4 (nội dung thẻ khác → tránh data leakage)

Cấu hình cứng (hard-coded) theo yêu cầu cụ thể:
  ┌──────────────┬────────┬───────┬──────────────────────┐
  │ Class        │ Train  │ Val   │ Samples/cat          │
  ├──────────────┼────────┼───────┼──────────────────────┤
  │ cccd_ms_c(2) │ 3 tpl  │ 1 tpl │ 25/cat → 150/tpl     │
  │ cccd_mt_c(3) │ 3 tpl  │ 1 tpl │ 25/cat → 150/tpl     │
  │ sg_ms   (13) │ 3 tpl  │ 1 tpl │ 25/cat → 150/tpl     │
  │ sg_mt   (12) │ 3 tpl  │ 1 tpl │ 25/cat → 150/tpl     │
  │ other    (9) │ 4 tpl  │ 4 tpl │ train:25 / val:10/cat│
  └──────────────┴────────┴───────┴──────────────────────┘

Cách dùng:
    python scripts/build_syn_multiclass.py
    python scripts/build_syn_multiclass.py --dry_run
"""

import sys
import json
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_testset import generate_comprehensive_dataset
from scripts.convert_gt_to_yolo import corners_to_pose_label

# ═══════════════════════════════════════════════════════════════════════════════
# Cấu hình dataset
# ═══════════════════════════════════════════════════════════════════════════════

TEMPLATES_DIR = PROJECT_ROOT / "assets" / "templates"
OUTPUT_DIR    = PROJECT_ROOT / "data_multiclasses" / "synthetic_multiclass"
SEED          = 42

# Mapping class_name → mentor class_id (khớp src/detector/base.py)
CLASS_ID_MAP = {
    "cccd_ms_c": 2,
    "cccd_mt_c": 3,
    "other":     9,
    "sg_mt":     12,
    "sg_ms":     13,
}

# ─── Phân bổ template cho train/val ──────────────────────────────────────────
# Key: class_name
# Value: dict với train_indices, val_indices (0-indexed theo sorted file list),
#        train_samples_per_cat, val_samples_per_cat
SPLIT_CONFIG = {
    "cccd_ms_c": {
        "train_indices": [0, 1, 2],
        "val_indices":   [3],
        "train_samples": 25,
        "val_samples":   25,
    },
    "cccd_mt_c": {
        "train_indices": [0, 1, 2],
        "val_indices":   [3],
        "train_samples": 25,
        "val_samples":   25,
    },
    "sg_ms": {
        "train_indices": [0, 1, 2],
        "val_indices":   [3],
        "train_samples": 25,
        "val_samples":   25,
    },
    "sg_mt": {
        "train_indices": [0, 1, 2],
        "val_indices":   [3],
        "train_samples": 25,
        "val_samples":   25,
    },
    "other": {
        "train_indices": [0, 1, 2, 3],
        "val_indices":   [4, 5, 6, 7],
        "train_samples": 25,
        "val_samples":   10,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Hàm tiện ích
# ═══════════════════════════════════════════════════════════════════════════════

def get_image_size(img_path: Path):
    """Trả (width, height) của ảnh."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 640, 640
    return img.shape[1], img.shape[0]


def collect_items_from_gt(gt_json_path: Path, class_id: int, split: str):
    """Đọc gt_annotations.json, trả list (img_path, corners, class_id, split)."""
    if not gt_json_path.exists():
        return []
    with open(gt_json_path, encoding="utf-8") as f:
        gt = json.load(f)
    items = []
    base_dir = gt_json_path.parent
    for rel_key, entry in gt.items():
        img_path = base_dir / rel_key
        if img_path.exists():
            items.append((img_path, entry["corners"], class_id, split))
    return items


def write_yolo_dataset(all_items: list, output_dir: Path,
                       class_names: dict, dry_run: bool):
    """Ghi toàn bộ ảnh + label YOLO-Pose và tạo dataset.yaml."""
    stats = {"train": {}, "val": {}}

    if not dry_run:
        for split in ["train", "val"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counter = {}  # Đếm file theo class để đặt tên unique

    for img_path, corners, class_id, split in all_items:
        cname = class_names.get(class_id, f"class_{class_id}")
        key = f"{cname}_{split}"
        counter[key] = counter.get(key, 0) + 1
        idx = counter[key]

        new_name = f"cls{class_id:02d}_{cname}_{split}_{idx:04d}"
        dst_img  = output_dir / "images" / split / f"{new_name}.jpg"
        dst_lbl  = output_dir / "labels" / split / f"{new_name}.txt"

        stats[split][cname] = stats[split].get(cname, 0) + 1

        if dry_run:
            continue

        shutil.copy2(img_path, dst_img)

        img_w, img_h = get_image_size(img_path)
        label_line = corners_to_pose_label(corners, img_w, img_h, class_id=class_id)
        dst_lbl.write_text(label_line + "\n", encoding="utf-8")

    # In thống kê
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │             THỐNG KÊ DATASET                │")
    print("  ├──────────────┬──────────┬───────────────────┤")
    print("  │ Class        │  Train   │  Val              │")
    print("  ├──────────────┼──────────┼───────────────────┤")
    total_train = 0
    total_val = 0
    for cname in sorted(set(list(stats["train"].keys()) + list(stats["val"].keys()))):
        t = stats["train"].get(cname, 0)
        v = stats["val"].get(cname, 0)
        total_train += t
        total_val += v
        print(f"  │ {cname:<12} │ {t:>6}   │ {v:>6}            │")
    print("  ├──────────────┼──────────┼───────────────────┤")
    print(f"  │ TỔNG         │ {total_train:>6}   │ {total_val:>6}            │")
    print("  └──────────────┴──────────┴───────────────────┘")

    if dry_run:
        return

    # Ghi dataset.yaml
    nc = max(class_names.keys()) + 1
    yaml_data = {
        "path": output_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": {int(k): v for k, v in sorted(class_names.items())},
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True,
                  default_flow_style=None, sort_keys=False)
    print(f"\n  dataset.yaml -> {yaml_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Sinh synthetic multi-class dataset tu clean templates"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview thong ke, khong sinh anh")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dry_run = args.dry_run

    print("=" * 60)
    print("  BUILD SYNTHETIC MULTI-CLASS DATASET")
    if dry_run:
        print("  [DRY RUN] — chi preview, khong ghi file")
    print(f"  Templates : {TEMPLATES_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("=" * 60)

    # Thư mục tạm cho synthetic trung gian
    tmp_dir = PROJECT_ROOT / "data_multiclasses" / "_syn_tmp"

    all_items = []  # list of (img_path, corners, class_id, split)

    # ── Duyệt từng class ──────────────────────────────────────────────────────
    for class_name, cfg in SPLIT_CONFIG.items():
        class_id = CLASS_ID_MAP[class_name]
        tpl_dir  = TEMPLATES_DIR / class_name

        if not tpl_dir.exists():
            print(f"\n  [SKIP] {class_name}: thu muc template khong ton tai")
            continue

        # Lấy danh sách template sorted
        tpl_files = sorted(tpl_dir.glob("*.jpg"))
        print(f"\n  [{class_name}] class_id={class_id}  ({len(tpl_files)} templates)")

        train_indices = cfg["train_indices"]
        val_indices   = cfg["val_indices"]
        train_samples = cfg["train_samples"]
        val_samples   = cfg["val_samples"]

        # Kiểm tra indices hợp lệ
        max_idx = max(max(train_indices), max(val_indices))
        if max_idx >= len(tpl_files):
            print(f"    WARNING: chi co {len(tpl_files)} templates nhung can index {max_idx}")
            train_indices = [i for i in train_indices if i < len(tpl_files)]
            val_indices   = [i for i in val_indices if i < len(tpl_files)]

        # ── Sinh train ────────────────────────────────────────────────────
        for idx in train_indices:
            tpl = tpl_files[idx]
            tag = f"{class_name}_train_tpl{idx}"
            out = tmp_dir / tag
            print(f"    [train] tpl[{idx}] {tpl.name[:40]}...  ({train_samples}/cat)")

            if not dry_run:
                generate_comprehensive_dataset(str(tpl), str(out),
                                               samples_per_category=train_samples)
            gt_path = out / "gt_annotations.json"
            items = collect_items_from_gt(gt_path, class_id, "train")
            all_items.extend(items)
            if not dry_run:
                print(f"           -> {len(items)} anh")

        # ── Sinh val ──────────────────────────────────────────────────────
        for idx in val_indices:
            tpl = tpl_files[idx]
            tag = f"{class_name}_val_tpl{idx}"
            out = tmp_dir / tag
            print(f"    [val]   tpl[{idx}] {tpl.name[:40]}...  ({val_samples}/cat)")

            if not dry_run:
                generate_comprehensive_dataset(str(tpl), str(out),
                                               samples_per_category=val_samples)
            gt_path = out / "gt_annotations.json"
            items = collect_items_from_gt(gt_path, class_id, "val")
            all_items.extend(items)
            if not dry_run:
                print(f"           -> {len(items)} anh")

    # ── Ghi YOLO dataset ──────────────────────────────────────────────────────
    class_names_dict = {v: k for k, v in CLASS_ID_MAP.items()}
    print(f"\n  [GHI DATASET -> {OUTPUT_DIR.name}/]")
    write_yolo_dataset(all_items, OUTPUT_DIR, class_names_dict, dry_run)

    # ── Dọn thư mục tạm ──────────────────────────────────────────────────────
    if not dry_run and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"  Don thu muc tam: {tmp_dir.name}/")

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if dry_run:
        print("  [DRY RUN] hoan thanh — preview only")

        # Tính dự kiến
        print("\n  DU KIEN (6 categories x samples/cat x num_templates):")
        total_t = 0
        total_v = 0
        for cn, cfg in SPLIT_CONFIG.items():
            t = len(cfg["train_indices"]) * cfg["train_samples"] * 6
            v = len(cfg["val_indices"]) * cfg["val_samples"] * 6
            total_t += t
            total_v += v
            print(f"    {cn:<12}: train={t:>5}  val={v:>4}")
        print(f"    {'TONG':<12}: train={total_t:>5}  val={total_v:>4}")
        print(f"    Tong cong: {total_t + total_v} anh")
    else:
        print(f"  HOAN THANH!")
        print(f"  Dataset: {OUTPUT_DIR}")
        print(f"\n  Buoc tiep theo:")
        print(f"    python scripts/train_yolo.py --task pose \\")
        print(f"      --pose_data {OUTPUT_DIR}/dataset.yaml \\")
        print(f"      --epochs 100 --model nano")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
