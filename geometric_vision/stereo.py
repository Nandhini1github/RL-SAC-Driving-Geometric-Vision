"""Depth in the left optical frame of a horizontally rectified stereo pair."""
import cv2
import numpy as np


def make_matcher():
    block = 5
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=block,
        P1=8 * block**2, P2=32 * block**2, uniquenessRatio=8,
        disp12MaxDiff=1, speckleWindowSize=40, speckleRange=2,
    )


def estimate(left, right, matcher, focal_px, baseline_m):
    """Accept rectified uint8 grayscale images with identical intrinsics."""
    if left.ndim != 2 or left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Expected two equal-size uint8 grayscale images")
    if not all(np.isfinite(v) and v > 0 for v in (focal_px, baseline_m)):
        raise ValueError("Focal length and baseline must be positive and finite")
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    depth = np.full_like(disparity, np.nan)
    np.divide(focal_px * baseline_m, disparity, out=depth, where=disparity > 0)
    return depth


def unproject(u, v, z, fx, fy, cx, cy):
    """Return right, down, forward coordinates in metres."""
    return (u - cx) * z / fx, (v - cy) * z / fy, z
