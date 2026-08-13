import sys
import argparse
import subprocess
from pathlib import Path

# ── Cấu hình môi trường và đường dẫn dự án ─────────────────────────────────────
# Đảm bảo PYTHONPATH trỏ đúng vào thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sử dụng trình thực thi Python từ môi trường ảo .venv nếu tồn tại
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# ── Cấu hình đường dẫn cố định cho phiên bản OLD (cũ) và NEW (mới) ──────────────

# Bản đồ thư mục dữ liệu đánh giá cho mô hình cũ
OLD_DATA_MAP = {
    "real": "data/real_test",
    "front": "data/synthetic_testset_front",
    "back": "data/synthetic_testset_back",
}

# Bản đồ thư mục dữ liệu đánh giá cho mô hình mới
NEW_DATA_MAP = {
    "real": "data_new/real_test",
    "front": "data_new/synthetic_testset_front",
    "back": "data_new/synthetic_testset_back",
}

# Đường dẫn file trọng số (weights) của mô hình cũ
OLD_WEIGHTS = {
    "obb": "runs/obb/runs/train/obb_finetune/weights/best.pt",
    "pose": "runs/pose/runs/train/pose_finetune/weights/best.pt",
}

# Đường dẫn file trọng số (weights) của mô hình mới
NEW_WEIGHTS = {
    "obb": "runs/obb/runs_new/obb_finetune/weights/best.pt",
    "pose": "runs/pose/runs_new/pose_finetune/weights/best.pt",
}


def handle_build_data(args):
    """Xử lý lệnh tạo bộ dữ liệu tổng hợp mới."""
    # Khởi chạy script build_data_new.py qua tiến trình con
    cmd = [PYTHON, "scripts/build_data_new.py"]
    print("\n🚀 Đang khởi chạy quy trình tạo bộ dữ liệu mới data_new/...\n")
    subprocess.run(cmd)


def handle_benchmark(args):
    """Xử lý lệnh chạy benchmark đánh giá hiệu năng các phương pháp."""
    version = args.ver.lower()

    # Phân loại cấu hình dữ liệu và weights dựa trên phiên bản được chọn
    if version == "new":
        data_map = NEW_DATA_MAP
        obb_w = args.obb_weights or NEW_WEIGHTS["obb"]
        pose_w = args.pose_weights or NEW_WEIGHTS["pose"]
        ver_label = "MỚI (data_new & runs_new)"
    else:
        data_map = OLD_DATA_MAP
        obb_w = args.obb_weights or OLD_WEIGHTS["obb"]
        pose_w = args.pose_weights or OLD_WEIGHTS["pose"]
        ver_label = "CŨ / LEGACY (data & runs)"

    testset_path = data_map.get(args.target, args.target)

    # Kiểm tra sự tồn tại của file weights trước khi thực thi benchmark mô hình mới
    if version == "new":
        missing_weights = []
        if "obb" in args.methods and not Path(obb_w).exists():
            missing_weights.append(f"OBB ({obb_w})")
        if "pose" in args.methods and not Path(pose_w).exists():
            missing_weights.append(f"Pose ({pose_w})")

        if missing_weights:
            print(f"\n⚠️  CẢNH BÁO: Bạn chọn --ver new nhưng chưa train mô hình mới!")
            print(f"    Không tìm thấy file weights: {', '.join(missing_weights)}")
            print(f"    👉 Hãy chạy lệnh train trước: 'python main.py train obb' hoặc 'python main.py train pose'\n")
            return

    # Xây dựng lệnh gọi script run_benchmark.py
    cmd = [
        PYTHON, "scripts/run_benchmark.py",
        "--testset", testset_path,
        "--methods", *args.methods,
    ]

    if "obb" in args.methods:
        cmd.extend(["--obb_weights", obb_w])
    if "pose" in args.methods:
        cmd.extend(["--pose_weights", pose_w])

    if args.save:
        out_json = f"outputs/benchmark/cli_{version}_{args.target}.json"
        cmd.extend(["--save_json", out_json])

    print(f"\n🚀 Đang chạy Benchmark ({ver_label}) trên tập '{args.target}' ({testset_path})...\n")
    subprocess.run(cmd)


def handle_debug(args):
    """Xử lý lệnh kiểm tra và vẽ trực quan kết quả (debug)."""
    # Phân nhánh debug 1 ảnh cụ thể hoặc debug toàn bộ thư mục
    if args.image:
        cmd = [PYTHON, "scripts/debug_classical.py", "--img", args.image]
        print(f"\n🔍 Visual Debug Classical trên 1 ảnh: {args.image}\n")
    else:
        cmd = [
            PYTHON, "scripts/debug_alignment.py",
            "--ver", args.ver,
            "--detector", args.detector,
            "--num", str(args.num),
        ]
        if args.folder:
            cmd.extend(["--folder", args.folder])
        print(f"\n🔍 Visual Debug Alignment | ver={args.ver.upper()} | detector={args.detector.upper()}...\n")
    subprocess.run(cmd)


def handle_verify(args):
    """Xử lý lệnh kiểm tra và vẽ đè nhãn Ground Truth 4 góc."""
    # Xác định đường dẫn tập dữ liệu kiểm tra
    data_path = "data_new/real_test" if args.ver == "new" else "data/real_test"
    cmd = [
        PYTHON, "scripts/verify_gt.py",
        "--data", data_path,
        "--num", str(args.num),
        "--save"
    ]
    print(f"\n🖼️  Xác minh Ground Truth cho {args.num} ảnh tại '{data_path}'...\n")
    subprocess.run(cmd)


def handle_align(args):
    """Xử lý lệnh nắn phẳng ảnh thẻ căn cước (align)."""
    version = args.ver.lower()
    obb_w  = args.obb_weights  or (NEW_WEIGHTS["obb"]  if version == "new" else OLD_WEIGHTS["obb"])
    pose_w = args.pose_weights or (NEW_WEIGHTS["pose"] if version == "new" else OLD_WEIGHTS["pose"])
    ver_label = "MỚI (runs_new)" if version == "new" else "CŨ (runs)"

    # Xây dựng các đối số cho script run_alignment.py
    cmd = [PYTHON, "scripts/run_alignment.py"]

    if args.img:
        cmd.extend(["--img", args.img])
    elif args.folder:
        cmd.extend(["--folder", args.folder])

    cmd.extend(["--detector", args.detector])
    cmd.extend(["--obb_weights",  obb_w])
    cmd.extend(["--pose_weights", pose_w])

    if args.save:
        cmd.append("--save")
    if args.show:
        cmd.append("--show")
    if args.comparison:
        cmd.append("--show-comparison")

    label = args.img or args.folder
    print(f"\n[ALIGN] Từ '{label}' dùng detector={args.detector.upper()} ({ver_label})...\n")
    subprocess.run(cmd)


def handle_train(args):
    """Xử lý lệnh huấn luyện mô hình YOLO."""
    # Gọi script train_yolo.py với các tham số tương ứng
    cmd = [
        PYTHON, "scripts/train_yolo.py",
        "--task", args.task,
        "--epochs", str(args.epochs),
        "--model", args.model
    ]
    print(f"\n🏋️  Bắt đầu huấn luyện YOLO-{args.task.upper()} ({args.epochs} epochs) trên data_new/...\n")
    subprocess.run(cmd)


def main():
    """Khởi tạo giao diện dòng lệnh (CLI) và điều hướng lệnh."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="🎯 ID Card Aligner CLI – Công cụ dòng lệnh báo cáo & thử nghiệm",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Danh sách lệnh")

    # ── Đăng ký lệnh build_data ───────────────────────────────────────────────
    p_build = subparsers.add_parser("build_data", help="Tạo bộ dữ liệu data_new (60/20/20 train/val/test)")
    p_build.set_defaults(func=handle_build_data)

    # ── Đăng ký lệnh benchmark ────────────────────────────────────────────────
    p_bench = subparsers.add_parser("benchmark", help="Chạy đánh giá số liệu benchmark")
    p_bench.add_argument("target", choices=["real", "front", "back"], default="real", nargs="?",
                         help="Tập test cần chạy: 'real' (30 ảnh), 'front' (120 ảnh), 'back' (120 ảnh)")
    p_bench.add_argument("--ver", choices=["new", "old"], default="new",
                         help="Phiên bản: 'new' (data_new & runs_new), 'old' (data & runs cũ). Mặc định: new")
    p_bench.add_argument("--methods", nargs="+", default=["classical", "obb", "pose"],
                         choices=["classical", "obb", "pose"], help="Các phương pháp cần benchmark")
    p_bench.add_argument("--obb_weights", default=None, help="Ghi đè file weights cho OBB")
    p_bench.add_argument("--pose_weights", default=None, help="Ghi đè file weights cho Pose")
    p_bench.add_argument("--save", action="store_true", default=True, help="Tự động lưu kết quả file JSON")
    p_bench.set_defaults(func=handle_benchmark)

    # ── Đăng ký lệnh debug ────────────────────────────────────────────────────
    p_debug = subparsers.add_parser("debug", help="Vẽ 4 góc detected + ảnh aligned cạnh nhau để debug")
    p_debug.add_argument("--image", default=None, help="Đường dẫn đến 1 ảnh cụ thể (tùy chọn)")
    p_debug.add_argument("--folder", default=None, help="Đường dẫn đến folder ảnh (mặc định: data_new/real_test)")
    p_debug.add_argument("--ver", choices=["new", "old"], default="new", help="Chọn phiên bản model: new hoặc old")
    p_debug.add_argument("--detector", choices=["obb", "pose", "classical"], default="obb", help="Detector cần debug")
    p_debug.add_argument("--num", type=int, default=30, help="Số lượng ảnh debug (mặc định: 30)")
    p_debug.set_defaults(func=handle_debug)

    # ── Đăng ký lệnh verify GT ────────────────────────────────────────────────
    p_verify = subparsers.add_parser("verify", help="Vẽ kiểm tra nhãn Ground Truth 4 góc")
    p_verify.add_argument("--ver", choices=["new", "old"], default="new", help="Chọn data_new hoặc data cũ")
    p_verify.add_argument("--num", type=int, default=5, help="Số ảnh muốn vẽ kiểm tra")
    p_verify.set_defaults(func=handle_verify)

    # ── Đăng ký lệnh align ────────────────────────────────────────────────────
    p_align = subparsers.add_parser("align", help="Detect 4 góc → Warp → Ảnh thẻ phẳng chuẩn ISO")
    align_input = p_align.add_mutually_exclusive_group(required=True)
    align_input.add_argument("--img",    type=str, help="Đường dẫn 1 ảnh đơn lẻ")
    align_input.add_argument("--folder", type=str, help="Thư mục chứa nhiều ảnh")
    p_align.add_argument("--ver", choices=["new", "old"], default="new",
                         help="Phiên bản weights: 'new' (runs_new) hoặc 'old' (runs cũ)")
    p_align.add_argument("--detector",    default="obb",
                         choices=["classical", "obb", "pose"],
                         help="Phương pháp detect (mặc định: obb)")
    p_align.add_argument("--obb_weights",  default=None)
    p_align.add_argument("--pose_weights", default=None)
    p_align.add_argument("--save",       action="store_true", help="Lưu ảnh aligned vào outputs/aligned/")
    p_align.add_argument("--show",       action="store_true", help="Hiển thị cửa sổ ảnh kết quả")
    p_align.add_argument("--comparison", action="store_true", help="Hiển thị side-by-side: gốc + aligned")
    p_align.set_defaults(func=handle_align)

    # ── Đăng ký lệnh train ────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Huấn luyện YOLO-OBB hoặc YOLO-Pose trên data_new")
    p_train.add_argument("task", choices=["obb", "pose", "both"], help="Mô hình cần train")
    p_train.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện")
    p_train.add_argument("--model", default="nano", choices=["nano", "small", "medium"], help="Kích thước mô hình")
    p_train.set_defaults(func=handle_train)

    # Parse tham số truyền vào từ giao diện dòng lệnh
    args = parser.parse_args()

    # Gọi hàm xử lý tương ứng với lệnh được chọn
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
