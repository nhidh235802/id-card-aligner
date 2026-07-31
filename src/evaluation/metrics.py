"""
metrics.py – Đo lường độ chính xác để benchmark các phương pháp.
"""

import numpy as np


def corner_distance_error(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Tính lỗi Euclidean distance giữa 4 góc predict và ground truth.

    Args:
        pred : (4, 2) float32
        gt   : (4, 2) float32

    Returns:
        dict với mean_err, max_err, per_corner_err (4,)
    """
    dists = np.linalg.norm(pred - gt, axis=1)  # (4,)
    return {
        "mean_err_px": float(dists.mean()),
        "max_err_px": float(dists.max()),
        "per_corner_err_px": dists.tolist(),
    }


def angle_error(pred_angle: float, gt_angle: float) -> float:
    """Lỗi góc xoay (độ), có tính đến wrap-around ±180°."""
    err = abs(pred_angle - gt_angle)
    return min(err, 360 - err)


def compute_iou_polygon(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Tính IoU giữa 2 polygon (4 điểm).
    Dùng Sutherland-Hodgman clipping thông qua OpenCV.
    """
    import cv2
    pred_i = pred.astype(np.float32).reshape(-1, 1, 2)
    gt_i   = gt.astype(np.float32).reshape(-1, 1, 2)

    inter_area = cv2.intersectConvexConvex(pred_i, gt_i)[0]
    pred_area  = cv2.contourArea(pred_i)
    gt_area    = cv2.contourArea(gt_i)
    union_area = pred_area + gt_area - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def benchmark_summary(results: list[dict]) -> dict:
    """
    Tổng hợp kết quả benchmark từ list các dict metrics.

    Args:
        results : list of dicts từ corner_distance_error + angle_error

    Returns:
        summary dict
    """
    mean_errs   = [r["mean_err_px"] for r in results]
    angle_errs  = [r.get("angle_err_deg", 0) for r in results]
    ious        = [r.get("iou", 0) for r in results]

    return {
        "n_samples": len(results),
        "corner_mean_err_px": float(np.mean(mean_errs)),
        "corner_std_err_px":  float(np.std(mean_errs)),
        "angle_mean_err_deg": float(np.mean(angle_errs)),
        "mean_iou":           float(np.mean(ious)),
    }
