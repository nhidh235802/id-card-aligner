"""
scripts/build_multiclass_dataset.py
═══════════════════════════════════════════════════════════════════════════════
Xây dựng dataset YOLO-Pose multi-class từ nhiều nguồn template khác nhau.

Workflow:
  1. Đọc configs/multiclass_sources.yaml để biết mỗi class cần template nào
  2. Với mỗi source được bật (enabled=true):
       a. Sinh synthetic data từ clean template (6 categories × N ảnh)
       b. Convert nhãn GT → YOLO-Pose format với đúng class_id
  3. Gộp tất cả vào train/ và val/ theo tỷ lệ
  4. Ghi dataset.yaml đầy đủ 14 class

Cách dùng:
    # Dùng config mặc định
    python scripts/build_multiclass_dataset.py

    # Chỉ build một nguồn cụ thể
    python scripts/build_multiclass_dataset.py --only cccd_magnetic

    # Preview không ghi file
    python scripts/build_multiclass_dataset.py --dry_run
"""

import sys
import json
import shutil
import argparse
import random
from pathlib import Path

import cv2
import yaml
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_testset import generate_comprehensive_dataset
from scripts.convert_gt_to_yolo import corners_to_pose_label


# ═══════════════════════════════════════════════════════════════════════════════
# Hàm tiện ích
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Đọc file cấu hình YAML multiclass_sources."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_template(path: str, name: str) -> bool:
    """Kiểm tra file template có tồn tại và đọc được không."""
    p = PROJECT_ROOT / path
    if not p.exists():
        print(f"  ⚠️  [{name}] Template không tìm thấy: {path}")
        return False
    img = cv2.imread(str(p))
    if img is None:
        print(f"  ⚠️  [{name}] Không đọc được file ảnh: {path}")
        return False
    h, w = img.shape[:2]
    print(f"  ✔  [{name}] Template OK: {path}  ({w}x{h}px)")
    return True


def get_image_size(img_path: Path):
    """Trả (width, height) của ảnh."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 640, 640
    return img.shape[1], img.shape[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Sinh + convert một nguồn đơn lẻ
# ═══════════════════════════════════════════════════════════════════════════════

def build_one_side(template_path: str,
                   class_id: int,
                   side_name: str,
                   tmp_dir: Path,
                   samples_train: int,
                   samples_val: int,
                   dry_run: bool = False) -> list:
    """Sinh synthetic từ 1 template.

    Trả list (img_path, corners, class_id, split).
    """
    abs_template = PROJECT_ROOT / template_path
    train_out = tmp_dir / side_name / "train_raw"
    val_out   = tmp_dir / side_name / "val_raw"

    if not dry_run:
        print(f"    -> Sinh {samples_train} anh/cat train...")
        generate_comprehensive_dataset(
            str(abs_template), str(train_out), samples_per_category=samples_train
        )
        print(f"    -> Sinh {samples_val} anh/cat val...")
        generate_comprehensive_dataset(
            str(abs_template), str(val_out), samples_per_category=samples_val
        )

    items = []
    for split, out_dir in [("train", train_out), ("val", val_out)]:
        gt_file = out_dir / "gt_annotations.json"
        if not gt_file.exists():
            if not dry_run:
                print(f"    WARNING: Khong tim thay gt_annotations.json tai {gt_file}")
            continue
        with open(gt_file, encoding="utf-8") as f:
            gt = json.load(f)
        for rel_key, entry in gt.items():
            img_path = out_dir / rel_key
            if img_path.exists():
                items.append((img_path, entry["corners"], class_id, split))

    return items


# ═══════════════════════════════════════════════════════════════════════════════
# Gộp tất cả sources thành YOLO dataset
# ═══════════════════════════════════════════════════════════════════════════════

def build_yolo_dataset(all_items: list,
                       output_dir: Path,
                       class_names_dict: dict,
                       dry_run: bool = False) -> None:
    """Ghi toàn bộ anh + label YOLO va tao dataset.yaml."""
    stats = {"train": {}, "val": {}}

    if not dry_run:
        for split in ["train", "val"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_path, corners, class_id, split in all_items:
        new_stem = f"cls{class_id:02d}__{img_path.parent.name}__{img_path.stem}"
        dst_img  = output_dir / "images" / split / f"{new_stem}.jpg"
        dst_lbl  = output_dir / "labels" / split / f"{new_stem}.txt"

        cname = class_names_dict.get(class_id, f"class_{class_id}")
        stats[split][cname] = stats[split].get(cname, 0) + 1

        if dry_run:
            continue

        shutil.copy2(img_path, dst_img)

        img_w, img_h = get_image_size(img_path)
        label_line = corners_to_pose_label(corners, img_w, img_h, class_id=class_id)
        dst_lbl.write_text(label_line + "\n", encoding="utf-8")

    # Thong ke
    print("\n  Dataset statistics:")
    for split in ["train", "val"]:
        total = sum(stats[split].values())
        print(f"    {split.upper()} ({total} anh):")
        for cname, count in sorted(stats[split].items()):
            bar = "#" * (count // 10)
            print(f"      {cname:<16} {count:4d}  {bar}")

    if dry_run:
        return

    # Ghi dataset.yaml
    nc = max(class_names_dict.keys()) + 1
    names_yaml = "\n".join(
        f"  {cid}: {cname}"
        for cid, cname in sorted(class_names_dict.items())
    )

    yaml_content = f"""# YOLO-Pose Multi-class Dataset — ID Card Aligner
# Generated by scripts/build_multiclass_dataset.py

path: {output_dir.resolve().as_posix()}
train: images/train
val:   images/val

nc: {nc}
names:
{names_yaml}

kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
"""
    (output_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\n  dataset.yaml -> {output_dir / 'dataset.yaml'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Xay dung YOLO-Pose multi-class dataset tu nhieu template the"
    )
    parser.add_argument(
        "--config", default="configs/multiclass_sources.yaml",
        help="Duong dan file cau hinh (mac dinh: configs/multiclass_sources.yaml)"
    )
    parser.add_argument(
        "--only", default=None,
        help="Chi build 1 source theo ten (vd: --only cccd_magnetic)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Preview thong ke ma khong ghi file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (mac dinh: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Đọc config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Khong tim thay file config: {config_path}")
        sys.exit(1)

    cfg           = load_config(str(config_path))
    output_dir    = PROJECT_ROOT / cfg["output_dir"]
    samples_train = cfg.get("samples_train", 60)
    samples_val   = cfg.get("samples_val", 20)
    class_names   = {int(k): v for k, v in cfg["class_names"].items()}
    sources       = cfg.get("sources", [])
    tmp_dir       = PROJECT_ROOT / "data" / "_multiclass_tmp"

    print("=" * 60)
    print("  Build Multi-class YOLO-Pose Dataset")
    if args.dry_run:
        print("  [DRY RUN] - khong ghi file thuc su")
    print(f"  Output : {output_dir}")
    print(f"  Config : {config_path.name}")
    print(f"  Samples: {samples_train} train / {samples_val} val / category")
    print("=" * 60)

    # Kiểm tra templates
    print("\n[Kiem tra templates]")
    valid_sources = []
    for src in sources:
        name    = src["name"]
        enabled = src.get("enabled", True)

        if args.only and name != args.only:
            continue
        if not enabled:
            print(f"  SKIP [{name}] (enabled: false)")
            continue

        ok = True
        if src.get("template_front"):
            ok &= check_template(src["template_front"], f"{name}/front")
        if src.get("template_back"):
            ok &= check_template(src["template_back"], f"{name}/back")

        if ok:
            valid_sources.append(src)
        else:
            print(f"  ERROR [{name}] Bo qua do template bi thieu")

    if not valid_sources:
        print("\nERROR: Khong co source nao hop le.")
        sys.exit(1)

    # Sinh synthetic và thu thập items
    all_items = []
    for src in valid_sources:
        name = src["name"]
        print(f"\n[{name}] Dang xu ly...")

        sides = []
        if src.get("template_front") and src.get("class_id_front") is not None:
            sides.append(("front", src["template_front"], int(src["class_id_front"])))
        if src.get("template_back") and src.get("class_id_back") is not None:
            sides.append(("back",  src["template_back"],  int(src["class_id_back"])))

        for side, tpl_path, cid in sides:
            cname = class_names.get(cid, f"class_{cid}")
            print(f"  [{side}] class_id={cid} ({cname})")

            items = build_one_side(
                template_path = tpl_path,
                class_id      = cid,
                side_name     = f"{name}_{side}",
                tmp_dir       = tmp_dir,
                samples_train = samples_train,
                samples_val   = samples_val,
                dry_run       = args.dry_run,
            )
            all_items.extend(items)
            if not args.dry_run:
                print(f"    OK: {len(items)} anh ({cname})")

    # Gộp thành YOLO dataset
    print(f"\n[Ghi YOLO dataset -> {output_dir.name}/]")
    build_yolo_dataset(
        all_items        = all_items,
        output_dir       = output_dir,
        class_names_dict = class_names,
        dry_run          = args.dry_run,
    )

    # Dọn thư mục tạm
    if not args.dry_run and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  DRY RUN hoan thanh")
    else:
        print(f"  Dataset san sang tai: {output_dir}")
        print(f"  Buoc tiep theo:")
        print(f"    python scripts/train_yolo.py --task pose \\")
        print(f"      --pose_data {output_dir}/dataset.yaml")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
