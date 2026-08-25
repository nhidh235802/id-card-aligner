"""
scripts/build_mixed_dataset.py  (v2 — offline augmentation)
════════════════════════════════════════════════════════════════════════
Tạo dataset gộp Synthetic + Real (với augmentation đa dạng) để train 1 lần.

Chiến lược:
  Train = 2,400 synthetic  +  47 real × 20 biến thể = 3,340 ảnh
  Val   =   840 synthetic  +  12 real  × 1 (gốc)    =   852 ảnh

  Mỗi biến thể real được tạo bằng cách kết hợp ngẫu nhiên:
    Geometric : HorizontalFlip, Rotate ±20°, Perspective warp
    Color     : Brightness/Contrast, HSV shift, GaussianBlur, MotionBlur

  Keypoints + BBox được biến đổi theo đúng phép biến đổi hình học.
  Khi flip ngang: keypoints được hoán đổi theo flip_idx = [1,0,3,2].

Output: data_multiclasses/mixed_train_v1/
  images/train/  (~3,340 ảnh)
  images/val/    (  852 ảnh)
  labels/train/
  labels/val/
  dataset.yaml
"""

import sys, shutil, random
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Nguồn ─────────────────────────────────────────────────────────────
SRC_SYN_TRAIN_IMG  = PROJECT_ROOT / "data_multiclasses/synthetic_multiclass/images/train"
SRC_SYN_TRAIN_LBL  = PROJECT_ROOT / "data_multiclasses/synthetic_multiclass/labels/train"
SRC_SYN_VAL_IMG    = PROJECT_ROOT / "data_multiclasses/synthetic_multiclass/images/val"
SRC_SYN_VAL_LBL    = PROJECT_ROOT / "data_multiclasses/synthetic_multiclass/labels/val"

SRC_REAL_TRAIN_IMG = PROJECT_ROOT / "data_multiclasses/real_finetune_v2/images/train"
SRC_REAL_TRAIN_LBL = PROJECT_ROOT / "data_multiclasses/real_finetune_v2/labels/train"
SRC_REAL_VAL_IMG   = PROJECT_ROOT / "data_multiclasses/real_finetune_v2/images/val"
SRC_REAL_VAL_LBL   = PROJECT_ROOT / "data_multiclasses/real_finetune_v2/labels/val"

OUT_DIR         = PROJECT_ROOT / "data_multiclasses/mixed_train_v1"
REAL_OVERSAMPLE = 20   # Số biến thể augment cho mỗi ảnh real (trong train)
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FLIP_IDX        = [1, 0, 3, 2]  # TL↔TR, BL↔BR khi flip ngang

CLASS_NAMES = {
    0:"cmnd_ms_c", 1:"cmnd_mt_c",  2:"cccd_ms_c", 3:"cccd_mt_c",
    4:"cccd_ms_m", 5:"cccd_mt_m",  6:"blx_ms",    7:"blx_mt",
    8:"hc_mt",     9:"other",     10:"print",     11:"hc_nn",
   12:"sg_mt",    13:"sg_ms",
}

# ══════════════════════════════════════════════════════════════════════
#  PARSE / SERIALIZE YOLO POSE LABEL
# ══════════════════════════════════════════════════════════════════════

def parse_label(text: str, img_w: int, img_h: int):
    """
    Đọc 1 dòng YOLO-Pose label, trả về:
      cls_id  : int
      kpts_px : list of [x, y] (pixel coords), 4 điểm
      vis     : list of float (visibility per keypoint)
    """
    parts = text.strip().split()
    cls_id = int(parts[0])
    # Keypoints (bỏ qua bbox — sẽ tính lại sau augment)
    kpts_px, vis = [], []
    for i in range(4):
        kx = float(parts[5 + i*3]) * img_w
        ky = float(parts[5 + i*3 + 1]) * img_h
        v  = float(parts[5 + i*3 + 2])
        kpts_px.append([kx, ky])
        vis.append(v)
    return cls_id, kpts_px, vis


def build_label(cls_id: int, kpts_px: list, vis: list,
                img_w: int, img_h: int) -> str:
    """
    Tính lại bbox từ 4 keypoints (thêm padding 5px) và tạo chuỗi label YOLO-Pose.
    """
    pts = np.array(kpts_px, dtype=np.float32)
    x_min = max(0.0, pts[:, 0].min() - 5)
    y_min = max(0.0, pts[:, 1].min() - 5)
    x_max = min(float(img_w), pts[:, 0].max() + 5)
    y_max = min(float(img_h), pts[:, 1].max() + 5)

    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    parts = [f"{cls_id}", f"{cx:.8f}", f"{cy:.8f}", f"{bw:.8f}", f"{bh:.8f}"]
    for i, (kx, ky) in enumerate(kpts_px):
        parts += [f"{kx/img_w:.8f}", f"{ky/img_h:.8f}", f"{vis[i]:.0f}"]
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRIC TRANSFORMS (ảnh + keypoints đồng thời)
# ══════════════════════════════════════════════════════════════════════

def apply_flip(img, kpts_px):
    """Flip ngang, hoán đổi keypoints theo FLIP_IDX."""
    h, w = img.shape[:2]
    flipped = cv2.flip(img, 1)
    new_kpts = [[w - kx, ky] for kx, ky in kpts_px]
    new_kpts = [new_kpts[i] for i in FLIP_IDX]
    return flipped, new_kpts


def apply_rotation(img, kpts_px, angle_deg):
    """Xoay ảnh quanh tâm, biến đổi keypoints bằng affine matrix."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    pts = np.array(kpts_px, dtype=np.float32)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])          # (4, 3)
    new_pts = (M @ pts_h.T).T              # (4, 2)
    new_pts[:, 0] = np.clip(new_pts[:, 0], 0, w - 1)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0, h - 1)
    return rotated, new_pts.tolist()


def apply_perspective(img, kpts_px, rng: random.Random, strength: float):
    """
    Biến dạng phối cảnh nhẹ, biến đổi keypoints bằng homography.
    strength: % kích thước ảnh dịch chuyển các góc (0.02 ~ 0.10).
    """
    h, w = img.shape[:2]
    mg = int(min(w, h) * strength)
    if mg < 1:
        return img, kpts_px

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.randint(0, mg), rng.randint(0, mg)],
        [w - rng.randint(0, mg), rng.randint(0, mg)],
        [w - rng.randint(0, mg), h - rng.randint(0, mg)],
        [rng.randint(0, mg), h - rng.randint(0, mg)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    pts = np.array(kpts_px, dtype=np.float32).reshape(-1, 1, 2)
    new_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    new_pts[:, 0] = np.clip(new_pts[:, 0], 0, w - 1)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0, h - 1)
    return warped, new_pts.tolist()


# ══════════════════════════════════════════════════════════════════════
#  COLOR TRANSFORMS (chỉ ảnh, không ảnh hưởng keypoints)
# ══════════════════════════════════════════════════════════════════════

def apply_color_aug(img, rng: random.Random):
    """Áp dụng biến đổi màu sắc / ánh sáng ngẫu nhiên."""
    # Brightness & Contrast
    alpha = rng.uniform(0.65, 1.35)   # contrast
    beta  = rng.uniform(-40,  40)     # brightness
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # HSV shift
    if rng.random() < 0.6:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + rng.randint(-15, 15), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.randint(-30, 30), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.randint(-30, 30), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Gaussian blur
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Motion blur
    if rng.random() < 0.20:
        k = rng.randint(3, 7)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        if rng.random() < 0.5:
            kernel = kernel.T   # vertical motion
        img = cv2.filter2D(img, -1, kernel)

    return img


# ══════════════════════════════════════════════════════════════════════
#  PER-IMAGE AUGMENTATION LOOP
# ══════════════════════════════════════════════════════════════════════

def augment_and_save(img_path: Path, lbl_path: Path,
                     dst_img: Path, dst_lbl: Path,
                     n_variants: int, base_seed: int, prefix: str):
    """
    Tạo n_variants biến thể augment cho 1 ảnh real và lưu xuống dst.
    Trả về số biến thể đã lưu thành công.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"    ⚠️  Không đọc được: {img_path.name}")
        return 0

    h, w = img.shape[:2]
    label_raw = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    if not label_raw:
        return 0

    cls_id, kpts_px, vis = parse_label(label_raw[0], w, h)
    saved = 0

    for vi in range(n_variants):
        rng = random.Random(base_seed + vi)
        aug_img   = img.copy()
        aug_kpts  = [list(k) for k in kpts_px]

        # ── Geometric ────────────────────────────────────────────
        if rng.random() < 0.5:
            aug_img, aug_kpts = apply_flip(aug_img, aug_kpts)

        angle = rng.uniform(-20, 20)
        aug_img, aug_kpts = apply_rotation(aug_img, aug_kpts, angle)

        if rng.random() < 0.5:
            strength = rng.uniform(0.02, 0.10)
            aug_img, aug_kpts = apply_perspective(aug_img, aug_kpts, rng, strength)

        # ── Color ─────────────────────────────────────────────────
        aug_img = apply_color_aug(aug_img, rng)

        # ── Kiểm tra tính hợp lệ ─────────────────────────────────
        pts = np.array(aug_kpts)
        if pts[:, 0].max() - pts[:, 0].min() < 5 or \
           pts[:, 1].max() - pts[:, 1].min() < 5:
            continue  # Keypoints bị dồn lại → bỏ qua

        # ── Lưu ──────────────────────────────────────────────────
        stem = f"{prefix}{img_path.stem}_r{vi:02d}"
        cv2.imwrite(str(dst_img / f"{stem}{img_path.suffix}"), aug_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        label_str = build_label(cls_id, aug_kpts, vis, w, h)
        (dst_lbl / f"{stem}.txt").write_text(label_str, encoding="utf-8")
        saved += 1

    return saved


# ══════════════════════════════════════════════════════════════════════
#  COPY HELPERS (synthetic — không augment)
# ══════════════════════════════════════════════════════════════════════

def copy_folder(src_img: Path, src_lbl: Path,
                dst_img: Path, dst_lbl: Path, prefix: str = "") -> int:
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in sorted(p for p in src_img.iterdir()
                           if p.suffix.lower() in IMG_EXTS):
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        stem = f"{prefix}{img_path.stem}"
        shutil.copy2(img_path, dst_img / f"{stem}{img_path.suffix}")
        shutil.copy2(lbl_path, dst_lbl / f"{stem}.txt")
        count += 1
    return count


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def write_yaml(out_dir: Path):
    data = {
        "path": out_dir.resolve().as_posix(),
        "train": "images/train",
        "val":   "images/val",
        "nc": 14,
        "names": {int(k): v for k, v in sorted(CLASS_NAMES.items())},
        "kpt_shape": [4, 3],
        "flip_idx":  [1, 0, 3, 2],
    }
    p = out_dir / "dataset.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True,
                  default_flow_style=None, sort_keys=False)
    return p


def main():
    print(f"\n{'='*62}")
    print(f"  BUILD MIXED DATASET v2 — Offline Augmentation")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*62}\n")

    if OUT_DIR.exists():
        print("  Xóa thư mục cũ...")
        shutil.rmtree(OUT_DIR)

    dst_ti = OUT_DIR / "images" / "train"
    dst_tl = OUT_DIR / "labels" / "train"
    dst_vi = OUT_DIR / "images" / "val"
    dst_vl = OUT_DIR / "labels" / "val"

    for d in [dst_ti, dst_tl, dst_vi, dst_vl]:
        d.mkdir(parents=True, exist_ok=True)

    # ── TRAIN: Synthetic (copy thẳng) ────────────────────────────
    print("  [TRAIN — Synthetic]")
    n_syn_train = copy_folder(SRC_SYN_TRAIN_IMG, SRC_SYN_TRAIN_LBL,
                               dst_ti, dst_tl, prefix="syn_")
    print(f"    {n_syn_train:>5} ảnh copied\n")

    # ── TRAIN: Real (augment ×20) ─────────────────────────────────
    print(f"  [TRAIN — Real × {REAL_OVERSAMPLE} augment variants]")
    real_imgs = sorted(p for p in SRC_REAL_TRAIN_IMG.iterdir()
                       if p.suffix.lower() in IMG_EXTS)
    n_real_train = 0
    for idx, img_path in enumerate(real_imgs):
        lbl_path = SRC_REAL_TRAIN_LBL / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        saved = augment_and_save(img_path, lbl_path,
                                 dst_ti, dst_tl,
                                 n_variants=REAL_OVERSAMPLE,
                                 base_seed=idx * 1000,
                                 prefix="real_")
        n_real_train += saved
        if (idx + 1) % 10 == 0:
            print(f"    ... xong {idx+1}/{len(real_imgs)} ảnh ({n_real_train} biến thể)")

    print(f"    {len(real_imgs)} ảnh gốc → {n_real_train} biến thể\n")

    # ── VAL: Synthetic (copy thẳng) ──────────────────────────────
    print("  [VAL — Synthetic]")
    n_syn_val = copy_folder(SRC_SYN_VAL_IMG, SRC_SYN_VAL_LBL,
                             dst_vi, dst_vl, prefix="syn_")
    print(f"    {n_syn_val:>5} ảnh copied\n")

    # ── VAL: Real (copy thẳng — không oversample) ─────────────────
    print("  [VAL — Real (gốc, không augment)]")
    n_real_val = copy_folder(SRC_REAL_VAL_IMG, SRC_REAL_VAL_LBL,
                              dst_vi, dst_vl, prefix="real_")
    print(f"    {n_real_val:>5} ảnh copied\n")

    # ── YAML ─────────────────────────────────────────────────────
    yaml_path = write_yaml(OUT_DIR)

    # ── Tóm tắt ──────────────────────────────────────────────────
    n_train = n_syn_train + n_real_train
    n_val   = n_syn_val + n_real_val
    print(f"{'='*62}")
    print(f"  ✅ HOÀN THÀNH")
    print(f"  Train : {n_train:>5} ảnh  (syn={n_syn_train}, real={n_real_train})")
    print(f"  Val   : {n_val:>5} ảnh  (syn={n_syn_val}, real={n_real_val})")
    print(f"  Real% : {n_real_train/n_train*100:.1f}% của train")
    print(f"  Config: {yaml_path}")
    print(f"\n  Lệnh train:")
    print(f"    python scripts/train_yolo.py --task pose \\")
    print(f"      --family yolo26 --model xlarge \\")
    print(f"      --pose_data {OUT_DIR}/dataset.yaml \\")
    print(f"      --pose_project runs/pose/mixed_v1 \\")
    print(f"      --epochs 100")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
