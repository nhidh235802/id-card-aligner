import argparse
from pathlib import Path

MODEL_MAP = {
    "yolo11": {
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
    },
    "yolo26": {
        "obb": {
            "nano":   "yolo26n-obb.pt",
            "small":  "yolo26s-obb.pt",
            "medium": "yolo26m-obb.pt",
            "large":  "yolo26l-obb.pt",
            "xlarge": "yolo26x-obb.pt",
        },
        "pose": {
            "nano":   "yolo26n-pose.pt",
            "small":  "yolo26s-pose.pt",
            "medium": "yolo26m-pose.pt",
            "large":  "yolo26l-pose.pt",
            "xlarge": "yolo26x-pose.pt",
        },
    },
}


def train_obb(epochs: int, model_size: str, data_yaml: str, project: str, imgsz: int, family: str = "yolo11"):
    """Huấn luyện tinh chỉnh (fine-tune) mô hình YOLO-OBB cho nhiệm vụ phát hiện bounding box định hướng của thẻ."""
    from ultralytics import YOLO

    # Nạp weights tương ứng từ preset
    weights = MODEL_MAP[family]["obb"][model_size]
    print(f"\n{'='*55}")
    print(f"  Fine-tune YOLO-OBB | model={weights} | epochs={epochs}")
    print(f"{'='*55}")

    # Khởi tạo mô hình và chạy huấn luyện với các siêu tham số
    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        patience=10,           # Dừng sớm (early stopping) nếu không cải thiện sau 10 epoch
        project=project,
        name="obb_finetune",
        exist_ok=True,
        verbose=True,
        # Cấu hình augmentation nhẹ cho dữ liệu
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


def train_pose(epochs: int, model_size: str, data_yaml: str, project: str, imgsz: int, family: str = "yolo11"):
    """Huấn luyện tinh chỉnh (fine-tune) mô hình YOLO-Pose cho nhiệm vụ phát hiện 4 keypoints góc thẻ."""
    from ultralytics import YOLO

    # Nạp weights tương ứng từ preset
    weights = MODEL_MAP[family]["pose"][model_size]
    print(f"\n{'='*55}")
    print(f"  Fine-tune YOLO-Pose | model={weights} | epochs={epochs}")
    print(f"{'='*55}")

    # Khởi tạo mô hình và chạy huấn luyện với các siêu tham số
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
    # Tự động chọn đường dẫn mặc định cho bộ dữ liệu cũ hoặc bộ dữ liệu data_new mới
    obb_default  = "data_new/yolo_obb/dataset.yaml" if Path("data_new/yolo_obb/dataset.yaml").exists() else "data/yolo_obb/dataset.yaml"
    pose_default = "data_new/yolo_pose/dataset.yaml" if Path("data_new/yolo_pose/dataset.yaml").exists() else "data/yolo_pose/dataset.yaml"
    
    obb_proj_default  = "runs/obb/runs_new" if Path("data_new").exists() else "runs/obb/runs/train"
    pose_proj_default = "runs/pose/runs_new" if Path("data_new").exists() else "runs/pose/runs/train"

    # Đọc tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Fine-tune YOLO-OBB và YOLO-Pose cho thẻ CCCD")
    parser.add_argument("--task",        choices=["obb", "pose", "both"], default="both")
    parser.add_argument("--family",      choices=["yolo11", "yolo26"], default="yolo26",
                        help="Họ model (yolo11 hoặc yolo26, mặc định: yolo26)")
    parser.add_argument("--model",       choices=["nano", "small", "medium", "large", "xlarge"], default="nano",
                        help="Kích thước model (nano→xlarge, mặc định: nano)")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--imgsz",       type=int, default=640)
    parser.add_argument("--obb_data",    default=obb_default)
    parser.add_argument("--pose_data",   default=pose_default)
    parser.add_argument("--obb_project",  default=obb_proj_default)
    parser.add_argument("--pose_project", default=pose_proj_default)
    args = parser.parse_args()

    # Kiểm tra model size hợp lệ cho family
    if args.model in ("large", "xlarge") and args.family == "yolo11":
        print(f"WARNING: yolo11 chi co nano/small/medium. Tu dong chuyen sang medium.")
        args.model = "medium"

    # Điều hướng thực thi huấn luyện mô hình được lựa chọn
    if args.task in ("obb", "both"):
        train_obb(args.epochs, args.model, args.obb_data, args.obb_project, args.imgsz, args.family)

    if args.task in ("pose", "both"):
        train_pose(args.epochs, args.model, args.pose_data, args.pose_project, args.imgsz, args.family)
