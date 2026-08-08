"""
debug_alignment.py – Vẽ 4 góc detected + ảnh aligned cạnh nhau để debug.
Lưu ảnh side-by-side vào outputs/debug_align/
"""
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligner.perspective_aligner import PerspectiveAligner
from src.utils.corner_utils import order_corners

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_obb_detector():
    with open("configs/obb_detector.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["weights"] = "runs/obb/runs/train/obb_finetune/weights/best.pt"
    from src.detector.obb_detector import OBBDetector
    return OBBDetector(cfg).load_model()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()

    folder = Path(args.folder)
    out_dir = Path("outputs/debug_align")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])[:args.num]

    detector = load_obb_detector()
    aligner = PerspectiveAligner(target_width=856, target_height=540)

    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255)]

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = detector.detect(image)
        if result.confidence < 0.3:
            print(f"  SKIP {img_path.name} (conf={result.confidence:.2f})")
            continue

        # Raw corners from detector (already order_corners'd inside detector)
        corners = result.corners

        # ── Vẽ ảnh gốc + 4 góc ──
        vis = image.copy()
        pts = corners.astype(np.int32)
        cv2.polylines(vis, [pts.reshape(-1,1,2)], True, (0,200,0), 3)
        for i, (pt, label, color) in enumerate(zip(pts, labels, colors)):
            cv2.circle(vis, tuple(pt), 12, color, -1)
            cv2.putText(vis, f"{label}({i})", (pt[0]+15, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ── Log tọa độ ──
        print(f"\n{img_path.name} (conf={result.confidence:.2f})")
        for i, (label, pt) in enumerate(zip(labels, corners)):
            print(f"  {label}: ({pt[0]:.0f}, {pt[1]:.0f})")

        # Tính cạnh ngang vs dọc
        w_top = np.linalg.norm(corners[1] - corners[0])
        w_bot = np.linalg.norm(corners[2] - corners[3])
        h_left = np.linalg.norm(corners[3] - corners[0])
        h_right = np.linalg.norm(corners[2] - corners[1])
        print(f"  Width (TL-TR): {w_top:.0f}, (BL-BR): {w_bot:.0f}")
        print(f"  Height(TL-BL): {h_left:.0f}, (TR-BR): {h_right:.0f}")
        print(f"  Ratio W/H: {max(w_top,w_bot)/max(h_left,h_right):.2f} (expected ~1.59)")

        # ── Align ──
        aligned = aligner.align(image, corners)

        # ── Ghép side-by-side ──
        h_orig = vis.shape[0]
        h_ali = aligned.shape[0]
        scale = h_orig / h_ali
        aligned_resized = cv2.resize(aligned, (int(aligned.shape[1]*scale), h_orig))

        divider = np.full((h_orig, 6, 3), 255, dtype=np.uint8)
        combined = np.hstack([vis, divider, aligned_resized])

        save_path = out_dir / f"debug_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Saved: {save_path}")

    print(f"\nDone! Check: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
