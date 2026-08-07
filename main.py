"""
main.py – Giao diện dòng lệnh (CLI) chính cho ID Card Aligner

Cách dùng đơn giản nhất:
  # 1. Chạy Benchmark số liệu (tự động điền weights tốt nhất)
  python main.py benchmark real      # Chạy trên 30 ảnh thực tế
  python main.py benchmark front     # Chạy trên 120 ảnh synthetic mặt trước
  python main.py benchmark back      # Chạy trên 120 ảnh synthetic mặt sau

  # 2. Debug vẽ 4 góc trên 1 ảnh cụ thể
  python main.py debug data/real_test/13-7-2_1.jpg

  # 3. Kiểm tra nhãn Ground Truth
  python main.py verify --data data/real_test --num 5

  # 4. Huấn luyện mô hình (GPU/CPU)
  python main.py train obb --epochs 50
  python main.py train pose --epochs 50
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Đảm bảo PYTHONPATH đúng
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Luôn dùng python trong .venv (có đủ thư viện cv2, ultralytics,...)
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# Đường dẫn mặc định chuẩn
DEFAULT_OBB_WEIGHTS = "runs/obb/runs/train/obb_finetune/weights/best.pt"
DEFAULT_POSE_WEIGHTS = "runs/pose/runs/train/pose_finetune/weights/best.pt"

TESTSET_MAP = {
    "real": "data/real_test",
    "front": "data/synthetic_testset_front",
    "back": "data/synthetic_testset_back",
}


def handle_benchmark(args):
    testset_path = TESTSET_MAP.get(args.target, args.target)

    cmd = [
        PYTHON, "scripts/run_benchmark.py",
        "--testset", testset_path,
        "--methods", *args.methods,
    ]

    if "obb" in args.methods:
        cmd.extend(["--obb_weights", args.obb_weights])
    if "pose" in args.methods:
        cmd.extend(["--pose_weights", args.pose_weights])

    if args.save:
        out_json = f"outputs/benchmark/cli_{args.target}.json"
        cmd.extend(["--save_json", out_json])

    print(f"\n🚀 Đang chạy Benchmark trên tập '{args.target}' ({testset_path})...\n")
    subprocess.run(cmd)


def handle_debug(args):
    cmd = [PYTHON, "scripts/debug_classical.py", "--img", args.image]
    print(f"\n🔍 Visual Debug trên ảnh: {args.image}\n")
    subprocess.run(cmd)


def handle_verify(args):
    cmd = [
        PYTHON, "scripts/verify_gt.py",
        "--data", args.data,
        "--num", str(args.num),
        "--save"
    ]
    print(f"\n🖼️  Xác minh Ground Truth cho {args.num} ảnh tại '{args.data}'...\n")
    subprocess.run(cmd)


def handle_align(args):
    cmd = [PYTHON, "scripts/run_alignment.py"]

    if args.img:
        cmd.extend(["--img", args.img])
    elif args.folder:
        cmd.extend(["--folder", args.folder])

    cmd.extend(["--detector", args.detector])
    cmd.extend(["--obb_weights",  args.obb_weights])
    cmd.extend(["--pose_weights", args.pose_weights])

    if args.save:
        cmd.append("--save")
    if args.show:
        cmd.append("--show")
    if args.comparison:
        cmd.append("--show-comparison")

    label = args.img or args.folder
    print(f"\n[ALIGN] Tu '{label}' dung detector={args.detector.upper()}...\n")
    subprocess.run(cmd)


def handle_train(args):
    cmd = [
        PYTHON, "scripts/train_yolo.py",
        "--task", args.task,
        "--epochs", str(args.epochs),
        "--model", args.model
    ]
    print(f"\n🏋️  Bắt đầu huấn luyện YOLO-{args.task.upper()} ({args.epochs} epochs)...\n")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="🎯 ID Card Aligner CLI – Công cụ dòng lệnh báo cáo & thử nghiệm",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Danh sách lệnh")

    # ── Lệnh benchmark ────────────────────────────────────────────────────────
    p_bench = subparsers.add_parser("benchmark", help="Chạy đánh giá số liệu benchmark")
    p_bench.add_argument("target", choices=["real", "front", "back"], default="real", nargs="?",
                         help="Tập test cần chạy: 'real' (30 ảnh), 'front' (120 ảnh), 'back' (120 ảnh)")
    p_bench.add_argument("--methods", nargs="+", default=["classical", "obb", "pose"],
                         choices=["classical", "obb", "pose"], help="Các phương pháp cần benchmark")
    p_bench.add_argument("--obb_weights", default=DEFAULT_OBB_WEIGHTS, help="Đường dẫn file best.pt cho OBB")
    p_bench.add_argument("--pose_weights", default=DEFAULT_POSE_WEIGHTS, help="Đường dẫn file best.pt cho Pose")
    p_bench.add_argument("--save", action="store_true", default=True, help="Tự động lưu kết quả file JSON")
    p_bench.set_defaults(func=handle_benchmark)

    # ── Lệnh debug ────────────────────────────────────────────────────────────
    p_debug = subparsers.add_parser("debug", help="Vẽ 4 góc thử nghiệm trên 1 ảnh")
    p_debug.add_argument("image", help="Đường dẫn đến file ảnh cần debug")
    p_debug.set_defaults(func=handle_debug)

    # ── Lệnh verify GT ────────────────────────────────────────────────────────
    p_verify = subparsers.add_parser("verify", help="Vẽ kiểm tra nhãn Ground Truth 4 góc")
    p_verify.add_argument("--data", default="data/real_test", help="Thư mục testset chứa gt_annotations.json")
    p_verify.add_argument("--num", type=int, default=5, help="Số ảnh muốn vẽ kiểm tra")
    p_verify.set_defaults(func=handle_verify)

    # ── Lệnh align ────────────────────────────────────────────────────────────
    p_align = subparsers.add_parser("align", help="Detect 4 góc → Warp → Ảnh thẻ phẳng chuẩn ISO")
    align_input = p_align.add_mutually_exclusive_group(required=True)
    align_input.add_argument("--img",    type=str, help="Đường dẫn 1 ảnh đơn lẻ")
    align_input.add_argument("--folder", type=str, help="Thư mục chứa nhiều ảnh")
    p_align.add_argument("--detector",    default="obb",
                         choices=["classical", "obb", "pose"],
                         help="Phương pháp detect (mặc định: obb)")
    p_align.add_argument("--obb_weights",  default=DEFAULT_OBB_WEIGHTS)
    p_align.add_argument("--pose_weights", default=DEFAULT_POSE_WEIGHTS)
    p_align.add_argument("--save",       action="store_true", help="Lưu ảnh aligned vào outputs/aligned/")
    p_align.add_argument("--show",       action="store_true", help="Hiển thị cửa sổ ảnh kết quả")
    p_align.add_argument("--comparison", action="store_true", help="Hiển thị side-by-side: gốc + aligned")
    p_align.set_defaults(func=handle_align)

    # ── Lệnh train ────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Huấn luyện YOLO-OBB hoặc YOLO-Pose")
    p_train.add_argument("task", choices=["obb", "pose", "both"], help="Mô hình cần train")
    p_train.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện")
    p_train.add_argument("--model", default="nano", choices=["nano", "small", "medium"], help="Kích thước mô hình")
    p_train.set_defaults(func=handle_train)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
