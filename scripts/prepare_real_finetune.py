"""
scripts/prepare_real_finetune.py
═══════════════════════════════════════════════════════════════════════════════
Chuẩn bị dataset fine-tune từ ảnh real đã gán nhãn YOLO-Pose.

Lấy 30 ảnh real test (đã fix kpt_shape + class_id remap) và chia:
  - 24 ảnh → train (stratified theo class nếu có thể)
  -  6 ảnh → val

Output structure:
    data_multiclasses/real_finetune/
        images/train/   (24 ảnh)
        images/val/     (6 ảnh)
        labels/train/   (24 nhãn)
        labels/val/     (6 nhãn)
        dataset.yaml

Cách dùng:
    python scripts/prepare_real_finetune.py
    python scripts/prepare_real_finetune.py --src data_multiclasses/MultipleCard-Detect.yolo26/test
    python scripts/prepare_real_finetune.py --val_ratio 0.2  # 20% làm val
"""

import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


# Mapping class_id → class_name (theo mentor spec)
CLASS_NAMES = {
    0: "cmnd_ms_c", 1: "cmnd_mt_c",
    2: "cccd_ms_c", 3: "cccd_mt_c",
    4: "cccd_ms_m", 5: "cccd_mt_m",
    6: "blx_ms",    7: "blx_mt",
    8: "hc_mt",     9: "other",
    10: "print",   11: "hc_nn",
    12: "sg_mt",   13: "sg_ms",
}


def read_class_id(lbl_path: Path) -> int:
    """Đọc class_id từ dòng đầu tiên của file label."""
    try:
        first_line = lbl_path.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(first_line.split()[0])
    except Exception:
        return -1


def stratified_split(items: list, val_ratio: float, seed: int):
    """Chia train/val giữ tỷ lệ class đều nhau (stratified).

    items: list of (img_path, lbl_path, class_id)
    """
    rng = random.Random(seed)

    # Nhóm theo class
    by_class = defaultdict(list)
    for item in items:
        by_class[item[2]].append(item)

    train_set, val_set = [], []

    for class_id, class_items in by_class.items():
        rng.shuffle(class_items)
        n_val = max(1, round(len(class_items) * val_ratio))
        n_val = min(n_val, len(class_items) - 1)  # Ít nhất 1 ảnh trong train
        val_set.extend(class_items[:n_val])
        train_set.extend(class_items[n_val:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    return train_set, val_set


def copy_split(items: list, dst_img_dir: Path, dst_lbl_dir: Path):
    """Copy ảnh + label vào thư mục đích, trả về số file đã copy."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path, lbl_path, _ in items:
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
        count += 1
    return count


def write_dataset_yaml(out_dir: Path, nc: int, names: dict):
    """Tạo dataset.yaml cho YOLO training."""
    data = {
        "path": out_dir.resolve().as_posix(),
        "train": "images/train",
        "val":   "images/val",
        "nc": nc,
        "names": {int(k): v for k, v in sorted(names.items())},
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True,
                  default_flow_style=None, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dataset fine-tune từ ảnh real đã gán nhãn"
    )
    parser.add_argument(
        "--src", default="data_multiclasses/MultipleCard-Detect.yolo26/test",
        help="Thư mục nguồn chứa images/ và labels/"
    )
    parser.add_argument(
        "--out", default="data_multiclasses/real_finetune",
        help="Thư mục output dataset"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="Tỷ lệ ảnh dùng làm val (mặc định: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
    )
    args = parser.parse_args()

    src_dir     = PROJECT_ROOT / args.src
    img_dir     = src_dir / "images"
    lbl_dir     = src_dir / "labels"
    out_dir     = PROJECT_ROOT / args.out

    if not img_dir.exists():
        print(f"❌ Không tìm thấy: {img_dir}")
        sys.exit(1)
    if not lbl_dir.exists():
        print(f"❌ Không tìm thấy: {lbl_dir}")
        sys.exit(1)

    # Thu thập tất cả ảnh có label tương ứng
    items = []
    missing_labels = []
    for img_path in sorted(p for p in img_dir.iterdir()
                           if p.suffix.lower() in IMG_EXTS):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            missing_labels.append(img_path.name)
            continue
        class_id = read_class_id(lbl_path)
        items.append((img_path, lbl_path, class_id))

    print(f"\n{'='*58}")
    print(f"  Prepare Real Fine-tune Dataset")
    print(f"  Source  : {src_dir}")
    print(f"  Output  : {out_dir}")
    print(f"  Total   : {len(items)} ảnh có label ({len(missing_labels)} thiếu label)")
    print(f"  Val     : {args.val_ratio*100:.0f}%")

    # Phân bố class
    print(f"\n  Phân bố class trong nguồn:")
    class_counts = defaultdict(int)
    for _, _, cid in items:
        class_counts[cid] += 1
    for cid in sorted(class_counts):
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        print(f"    [{cid}] {cname:<12}: {class_counts[cid]} ảnh")

    # Stratified split
    train_items, val_items = stratified_split(items, args.val_ratio, args.seed)

    print(f"\n  Chia kết quả (stratified):")
    print(f"    Train: {len(train_items)} ảnh")
    print(f"    Val  : {len(val_items)} ảnh")

    # Phân bố class sau split
    print(f"\n  Phân bố class sau split:")
    t_counts, v_counts = defaultdict(int), defaultdict(int)
    for _, _, cid in train_items: t_counts[cid] += 1
    for _, _, cid in val_items:   v_counts[cid] += 1
    all_cls = sorted(set(list(t_counts.keys()) + list(v_counts.keys())))
    print(f"    {'Class':<14} {'Train':>6} {'Val':>5}")
    print(f"    {'─'*28}")
    for cid in all_cls:
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        print(f"    {cname:<14} {t_counts.get(cid,0):>6} {v_counts.get(cid,0):>5}")
    print(f"{'='*58}\n")

    # Xóa output cũ nếu có
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Copy files
    copy_split(train_items,
               out_dir / "images" / "train",
               out_dir / "labels" / "train")
    copy_split(val_items,
               out_dir / "images" / "val",
               out_dir / "labels" / "val")

    # Ghi dataset.yaml
    # nc = max class_id + 1 (theo mentor 14 classes)
    yaml_path = write_dataset_yaml(out_dir, nc=14, names=CLASS_NAMES)

    print(f"  ✅ Dataset sẵn sàng tại: {out_dir}")
    print(f"  📄 Config: {yaml_path}")
    print(f"\n  Fine-tune command:")
    print(f"    python scripts/train_yolo.py --task pose \\")
    print(f"      --family yolo26 --model xlarge \\")
    print(f"      --pose_data {out_dir / 'dataset.yaml'} \\")
    print(f"      --epochs 30 --finetune")
    print()


if __name__ == "__main__":
    main()
