"""
debug_classical.py – Xem trực tiếp Canny + contours trên 1 ảnh test
để hiểu tại sao ClassicalDetector không detect được thẻ.

Cách dùng:
    python scripts/debug_classical.py --img data/synthetic_testset_front/1_easy_standard/1_easy_standard_001.jpg
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detector.classical_detector import ClassicalDetector

parser = argparse.ArgumentParser()
parser.add_argument("--img", required=True)
parser.add_argument("--canny_low",  type=int, default=50)
parser.add_argument("--canny_high", type=int, default=150)
args = parser.parse_args()

image = cv2.imread(args.img)
h, w = image.shape[:2]
min_area = h * w * 0.10  # 10%

gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges   = cv2.Canny(blurred, args.canny_low, args.canny_high)
kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges   = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Tổng contours tìm được: {len(contours)}")
print(f"Image size: {w}x{h}, min_area ngưỡng: {min_area:.0f} px²")

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

if quads:
    best = max(quads, key=lambda x: x[0])
    print(f"\n→ Best quad area = {best[0]:.0f} → confidence = {min(best[0]/(h*w), 1.0):.3f}")
else:
    print("\n→ Không tìm thấy contour 4 cạnh nào phù hợp!")

# Resize cho vừa màn hình
scale = min(800/w, 700/h)
vis_r  = cv2.resize(vis,   (int(w*scale), int(h*scale)))
edges_r = cv2.resize(edges, (int(w*scale), int(h*scale)))

cv2.imshow("Edges (Canny)", edges_r)
cv2.imshow("Detected Contours", vis_r)
print("\nNhấn phím bất kỳ để đóng cửa sổ...")
cv2.waitKey(0)
cv2.destroyAllWindows()
