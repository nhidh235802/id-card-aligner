import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# Đảm bảo PYTHONPATH trỏ đúng vào thư mục gốc của dự án
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detector.classical_detector import ClassicalDetector

# Đọc các đối số truyền vào từ dòng lệnh
parser = argparse.ArgumentParser()
parser.add_argument("--img", required=True)
parser.add_argument("--canny_low",  type=int, default=50)
parser.add_argument("--canny_high", type=int, default=150)
args = parser.parse_args()

# Đọc ảnh gốc và lấy kích thước khung hình
image = cv2.imread(args.img)
h, w = image.shape[:2]
min_area = h * w * 0.10  # Ngưỡng diện tích tối thiểu (10% diện tích ảnh)

# ── Khối tiền xử lý ảnh: Chuyển ảnh xám, làm mờ Gaussian và tách biên Canny ──
gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges   = cv2.Canny(blurred, args.canny_low, args.canny_high)
kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges   = cv2.dilate(edges, kernel, iterations=1)

# ── Khối trích xuất contours từ bản đồ cạnh ──
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Tổng contours tìm được: {len(contours)}")
print(f"Image size: {w}x{h}, min_area ngưỡng: {min_area:.0f} px²")

# ── Khối lọc các đường contour thành hình đa giác 4 cạnh (quadrilateral) ──
vis = image.copy()
quads = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue
    peri  = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    print(f"  Contour area={area:.0f}  sides={len(approx)}")
    if len(approx) == 4:
        quads.append((area, approx))
        cv2.drawContours(vis, [approx], -1, (0, 255, 0), 3)
        for pt in approx.reshape(4, 2):
            cv2.circle(vis, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
    else:
        cv2.drawContours(vis, [approx], -1, (200, 200, 0), 1)

# ── Khối đánh giá ứng viên contour lớn nhất và tính độ tin cậy confidence ──
if quads:
    best = max(quads, key=lambda x: x[0])
    print(f"\n→ Best quad area = {best[0]:.0f} → confidence = {min(best[0]/(h*w), 1.0):.3f}")
else:
    print("\n→ Không tìm thấy contour 4 cạnh nào phù hợp!")

# ── Khối thu nhỏ kích thước hiển thị và xuất ra màn hình OpenCV Window ──
scale = min(800/w, 700/h)
vis_r  = cv2.resize(vis,   (int(w*scale), int(h*scale)))
edges_r = cv2.resize(edges, (int(w*scale), int(h*scale)))

cv2.imshow("Edges (Canny)", edges_r)
cv2.imshow("Detected Contours", vis_r)
print("\nNhấn phím bất kỳ để đóng cửa sổ...")
cv2.waitKey(0)
cv2.destroyAllWindows()
