import sys
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import các module trong src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligner.perspective_aligner import PerspectiveAligner
from src.utils.vis_utils import draw_detection_result

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Trọng số của phiên bản cũ (OLD)
OLD_WEIGHTS = {
    "obb": "runs/obb/runs/train/obb_finetune/weights/best.pt",
    "pose": "runs/pose/runs/train/pose_finetune/weights/best.pt",
}

# Trọng số của phiên bản mới (NEW)
NEW_WEIGHTS = {
    "obb": "runs/obb/runs_new/obb_finetune/weights/best.pt",
    "pose": "runs/pose/runs_new/pose_finetune/weights/best.pt",
}


def load_detector(detector_type: str, weights_path: str):
    """Khởi tạo và nạp trọng số cho bộ phát hiện góc thẻ (Classical, OBB hoặc Pose)."""
    if detector_type == "classical":
        with open("configs/classical_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        from src.detector.classical_detector import ClassicalDetector
        return ClassicalDetector(cfg).load_model()
    elif detector_type == "obb":
        with open("configs/obb_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["weights"] = weights_path
        from src.detector.obb_detector import OBBDetector
        return OBBDetector(cfg).load_model()
    elif detector_type == "pose":
        with open("configs/pose_detector.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["weights"] = weights_path
        from src.detector.pose_detector import PoseDetector
        return PoseDetector(cfg).load_model()
    else:
        raise ValueError(f"Detector không hợp lệ: {detector_type}")


def main():
    """Hàm chính điều khiển quá trình visual debug alignment."""
    # Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Visual Debug Alignment: Vẽ 4 góc detected + ảnh aligned cạnh nhau")
    parser.add_argument("--folder", default="data_new/real_test", help="Thư mục ảnh cần debug")
    parser.add_argument("--ver", choices=["new", "old"], default="new", help="Chọn phiên bản model: new hoặc old")
    parser.add_argument("--detector", choices=["obb", "pose", "classical"], default="obb", help="Loại detector")
    parser.add_argument("--weights", default=None, help="Ghi đè đường dẫn weights")
    parser.add_argument("--num", type=int, default=30, help="Số lượng ảnh tối đa")
    args = parser.parse_args()

    version = args.ver.lower()

    # Xác định đường dẫn file trọng số weights
    if args.weights:
        weights_path = args.weights
    elif version == "new":
        weights_path = NEW_WEIGHTS.get(args.detector)
        if args.folder == "data_new/real_test" and not Path(args.folder).exists():
            args.folder = "data/real_test"
    else:
        weights_path = OLD_WEIGHTS.get(args.detector)
        if args.folder == "data_new/real_test":
            args.folder = "data/real_test"

    # Kiểm tra sự tồn tại của file weights
    if args.detector in ["obb", "pose"] and weights_path and not Path(weights_path).exists():
        print(f"\n❌ Không tìm thấy weights ({weights_path})!")
        print(f"👉 Hãy train model trước hoặc chỉ định --ver old / --weights\n")
        return

    folder = Path(args.folder)
    out_dir = Path(f"outputs/debug_align_{version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách đường dẫn các ảnh cần chạy debug
    images = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS])[:args.num]

    print(f"\n🔍 Visual Debug Alignment | Detector: {args.detector.upper()} ({version.upper()})")
    print(f"📁 Folder   : {folder}")
    if weights_path:
        print(f"🎯 Weights  : {weights_path}")
    print(f"📂 Save dir : {out_dir.resolve()}\n")

    # Khởi tạo mô hình detector và đối tượng nắn thẳng aligner
    detector = load_detector(args.detector, weights_path)
    aligner  = PerspectiveAligner(target_width=856, target_height=540)

    # Duyệt và xử lý từng ảnh
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Phát hiện 4 góc thẻ
        result = detector.detect(image)
        if result.confidence < 0.3:
            print(f"  ❌ SKIP {img_path.name} (conf={result.confidence:.2f} < 0.3)")
            continue

        corners = result.corners

        # ── Khối 1: Vẽ kết quả detection theo đúng format mentor ──
        #    (class_name + confidence + bbox + 4 keypoints)
        vis = draw_detection_result(image, result, show_corner_labels=True)

        # ── Khối 2: Thực hiện warp perspective để nắn thẳng thẻ ──
        aligned = aligner.align(image, corners)

        # ── Khối 3: Căn chỉnh chiều cao và ghép ảnh gốc + ảnh nắn thẳng cạnh nhau ──
        h_orig = vis.shape[0]
        h_ali  = aligned.shape[0]
        scale  = h_orig / h_ali
        aligned_resized = cv2.resize(aligned, (int(aligned.shape[1] * scale), h_orig))

        divider  = np.full((h_orig, 6, 3), 255, dtype=np.uint8)
        combined = np.hstack([vis, divider, aligned_resized])

        # Lưu ảnh kết quả debug ghép cạnh nhau
        save_path = out_dir / f"debug_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  ✅ {img_path.name}  class={result.class_name}  conf={result.confidence:.2f}")

    print(f"\n🎉 Hoàn thành! Kiểm tra ảnh debug tại: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
