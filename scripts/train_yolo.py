"""
train_yolo.py – Fine-tune YOLO-OBB và YOLO-Pose trên dataset thẻ CCCD.

Cách dùng:
  # Fine-tune cả 2:
  python scripts/train_yolo.py --task both

  # Chỉ OBB:
  python scripts/train_yolo.py --task obb

  # Chỉ Pose:
  python scripts/train_yolo.py --task pose

  # Tuỳ chỉnh epochs và model size:
  python scripts/train_yolo.py --task both --epochs 50 --model nano
"""

import argparse
from pathlib import Path


# ── Model presets ──────────────────────────────────────────────────────────────
# Ultralytics tự download weights nếu chưa có
MODEL_MAP = {
    "obb": {
        "nano":   "yolo11n-obb.pt",
        "small":  "yolo11s-obb.pt",
        "medium": "yolo11m-obb.pt",
    },
    "pose": {
        "nano":   "yolo11n-pose.pt",
        "small":  "yolo11s-pose.pt",
        "medium": "yolo11m-pose.pt",
    },
}


def train_obb(epochs: int, model_size: str, data_yaml: str, project: str, imgsz: int):
    from ultralytics import YOLO

    weights = MODEL_MAP["obb"][model_size]
    print(f"\n{'='*55}")
    print(f"  Fine-tune YOLO-OBB | model={weights} | epochs={epochs}")
    print(f"{'='*55}")

    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        patience=10,           # Early stopping nếu không cải thiện sau 10 epoch
        project=project,
        name="obb_finetune",
        exist_ok=True,
        verbose=True,
        # Augmentation nhẹ — data synthetic đã đa dạng sẵn rồi
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        flipud=0.0,
        fliplr=0.5,
    )
    best_weights = Path(project) / "obb_finetune" / "weights" / "best.pt"
    print(f"\n✅ OBB training xong! Best weights: {best_weights}")
    return best_weights


def train_pose(epochs: int, model_size: str, data_yaml: str, project: str, imgsz: int):
    from ultralytics import YOLO

    weights = MODEL_MAP["pose"][model_size]
    print(f"\n{'='*55}")
    print(f"  Fine-tune YOLO-Pose | model={weights} | epochs={epochs}")
    print(f"{'='*55}")

    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        patience=10,
        project=project,
        name="pose_finetune",
        exist_ok=True,
        verbose=True,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        fliplr=0.5,
    )
    best_weights = Path(project) / "pose_finetune" / "weights" / "best.pt"
    print(f"\n✅ Pose training xong! Best weights: {best_weights}")
    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO-OBB và YOLO-Pose cho thẻ CCCD")
    parser.add_argument("--task",      choices=["obb", "pose", "both"], default="both")
    parser.add_argument("--model",     choices=["nano", "small", "medium"], default="nano",
                        help="Kích thước model (nano=nhanh nhất, medium=chính xác nhất)")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--obb_data",  default="data/yolo_obb/dataset.yaml")
    parser.add_argument("--pose_data", default="data/yolo_pose/dataset.yaml")
    parser.add_argument("--project",   default="runs/train",
                        help="Thư mục lưu kết quả training (checkpoints, logs)")
    args = parser.parse_args()

    if args.task in ("obb", "both"):
        train_obb(args.epochs, args.model, args.obb_data, args.project, args.imgsz)

    if args.task in ("pose", "both"):
        train_pose(args.epochs, args.model, args.pose_data, args.project, args.imgsz)
