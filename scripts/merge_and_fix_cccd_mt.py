"""
scripts/merge_and_fix_cccd_mt.py
════════════════════════════════════════════════════════════════════════
Fix + gộp data cccd_mt_real mới với real test cũ thành dataset finetune.

Fix cần làm trên cccd_mt_real:
  1. Remap class_id: 0 → 3 (cccd_mt_c theo mentor spec)
  2. Fix flip_idx: [0,1,2,3] → [1,0,3,2]
  3. Loại bỏ label có tọa độ ngoài [0, 1] (gán nhãn lỗi)

Gộp với:
  - data_multiclasses/MultipleCard-Detect.yolo26/test/ (30 ảnh cũ, đã fix)

Output: data_multiclasses/real_finetune_v2/
  images/train/ (48 ảnh)
  images/val/   (12 ảnh)
  labels/train/
  labels/val/
  dataset.yaml
"""

import sys, shutil, random
from pathlib import Path
from collections import defaultdict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

# Mentor class spec
CLASS_NAMES = {
    0:"cmnd_ms_c", 1:"cmnd_mt_c", 2:"cccd_ms_c", 3:"cccd_mt_c",
    4:"cccd_ms_m", 5:"cccd_mt_m", 6:"blx_ms",    7:"blx_mt",
    8:"hc_mt",     9:"other",    10:"print",     11:"hc_nn",
    12:"sg_mt",   13:"sg_ms",
}

CCCD_MT_REMAP = {0: 3}   # Roboflow export dùng class 0, cần đổi thành 3


def fix_label_line(line: str, class_remap: dict = None) -> str | None:
    """
    Fix 1 dòng label:
      - Remap class_id nếu cần
      - Trả về None nếu có tọa độ ngoài [0, 1] (label lỗi)
    """
    parts = line.strip().split()
    if not parts or len(parts) < 17:
        return None

    # Remap class
    cls = int(parts[0])
    if class_remap:
        cls = class_remap.get(cls, cls)

    # Kiểm tra tọa độ bbox (cx cy w h) và keypoints (kx ky) trong [0,1]
    bbox_coords = [float(x) for x in parts[1:5]]
    kpt_xy = [float(parts[i]) for i in [5, 6, 8, 9, 11, 12, 14, 15]]
    all_coords = bbox_coords + kpt_xy
    if any(c < -0.01 or c > 1.01 for c in all_coords):
        return None  # Label lỗi, loại bỏ

    # Đảm bảo visibility = 2 cho các keypoints hợp lệ
    kpt_parts = list(parts[5:])
    fixed_kpts = []
    for i in range(4):
        kx = float(kpt_parts[i*3])
        ky = float(kpt_parts[i*3+1])
        # v = float(kpt_parts[i*3+2])  # giữ nguyên v từ roboflow
        v = float(kpt_parts[i*3+2])
        fixed_kpts.extend([f"{kx:.8f}", f"{ky:.8f}", f"{v:.0f}"])

    return f"{cls} " + " ".join(parts[1:5]) + " " + " ".join(fixed_kpts)


def load_source(img_dir: Path, lbl_dir: Path,
                class_remap: dict = None) -> list:
    """
    Load tất cả ảnh + label từ 1 thư mục.
    Trả về list of (img_path, fixed_label_text, class_id).
    """
    items = []
    skipped = 0
    for img_path in sorted(p for p in img_dir.iterdir()
                           if p.suffix.lower() in IMG_EXTS):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            skipped += 1
            continue
        raw = lbl_path.read_text(encoding="utf-8").strip()
        lines = [l for l in raw.splitlines() if l.strip()]
        if not lines:
            skipped += 1
            continue

        # Chỉ lấy detection đầu tiên (1 thẻ/ảnh)
        fixed = fix_label_line(lines[0], class_remap)
        if fixed is None:
            print(f"    ⚠️  Label lỗi, bỏ qua: {img_path.name}")
            skipped += 1
            continue

        cls_id = int(fixed.split()[0])
        items.append((img_path, fixed, cls_id))

    if skipped:
        print(f"    (Bỏ qua {skipped} ảnh thiếu label hoặc tọa độ lỗi)")
    return items


def stratified_split(items: list, val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for item in items:
        by_class[item[2]].append(item)

    train_set, val_set = [], []
    for cls_items in by_class.values():
        rng.shuffle(cls_items)
        n_val = max(1, round(len(cls_items) * val_ratio))
        n_val = min(n_val, len(cls_items) - 1)
        val_set.extend(cls_items[:n_val])
        train_set.extend(cls_items[n_val:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    return train_set, val_set


def write_items(items: list, img_dst: Path, lbl_dst: Path):
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img_path, fixed_lbl, _ in items:
        # Tránh trùng tên file nếu 2 nguồn có tên giống nhau
        dst_img = img_dst / img_path.name
        dst_lbl = lbl_dst / (img_path.stem + ".txt")
        counter = 1
        while dst_img.exists():
            stem = f"{img_path.stem}_{counter}"
            dst_img = img_dst / f"{stem}{img_path.suffix}"
            dst_lbl = lbl_dst / f"{stem}.txt"
            counter += 1

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text(fixed_lbl, encoding="utf-8")


def main():
    out_dir = PROJECT_ROOT / "data_multiclasses" / "real_finetune_v2"

    print(f"\n{'='*60}")
    print(f"  Merge & Fix → real_finetune_v2")
    print(f"{'='*60}\n")

    # ── Nguồn 1: 30 ảnh real test cũ (đã fix class + flip_idx) ──
    print("  [Nguồn 1] data_multiclasses/MultipleCard-Detect.yolo26/test")
    src1_img = PROJECT_ROOT / "data_multiclasses/MultipleCard-Detect.yolo26/test/images"
    src1_lbl = PROJECT_ROOT / "data_multiclasses/MultipleCard-Detect.yolo26/test/labels"
    items1 = load_source(src1_img, src1_lbl, class_remap=None)  # đã đúng class_id
    print(f"    → {len(items1)} ảnh hợp lệ\n")

    # ── Nguồn 2: 30 ảnh cccd_mt_real mới ──
    print("  [Nguồn 2] data_multiclasses/cccd_mt_real/train")
    src2_img = PROJECT_ROOT / "data_multiclasses/cccd_mt_real/train/images"
    src2_lbl = PROJECT_ROOT / "data_multiclasses/cccd_mt_real/train/labels"
    items2 = load_source(src2_img, src2_lbl, class_remap=CCCD_MT_REMAP)
    print(f"    → {len(items2)} ảnh hợp lệ\n")

    # ── Gộp và split ──
    all_items = items1 + items2
    print(f"  Tổng: {len(all_items)} ảnh")

    # Phân bố class
    class_dist = defaultdict(int)
    for _, _, cls in all_items:
        class_dist[cls] += 1
    print("\n  Phân bố class:")
    for cls_id in sorted(class_dist):
        print(f"    [{cls_id}] {CLASS_NAMES.get(cls_id,'?'):<12}: {class_dist[cls_id]} ảnh")

    train_items, val_items = stratified_split(all_items, val_ratio=0.2, seed=SEED)
    print(f"\n  Split 80/20 stratified:")
    print(f"    Train: {len(train_items)} ảnh")
    print(f"    Val  : {len(val_items)} ảnh")

    # ── Ghi output ──
    if out_dir.exists():
        shutil.rmtree(out_dir)
    write_items(train_items, out_dir/"images"/"train", out_dir/"labels"/"train")
    write_items(val_items,   out_dir/"images"/"val",   out_dir/"labels"/"val")

    # dataset.yaml
    yaml_data = {
        "path": out_dir.resolve().as_posix(),
        "train": "images/train",
        "val":   "images/val",
        "nc": 14,
        "names": {int(k): v for k, v in sorted(CLASS_NAMES.items())},
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True,
                  default_flow_style=None, sort_keys=False)

    print(f"\n  ✅ Dataset: {out_dir}")
    print(f"  📄 Config : {yaml_path}")
    print(f"\n  Fine-tune command:")
    print(f"    python scripts/train_yolo.py --task pose \\")
    print(f"      --family yolo26 --model xlarge \\")
    print(f"      --pose_data {out_dir}/dataset.yaml \\")
    print(f"      --pose_project runs/pose/real_finetune_v2 \\")
    print(f"      --epochs 50 --finetune")
    print()


if __name__ == "__main__":
    main()
