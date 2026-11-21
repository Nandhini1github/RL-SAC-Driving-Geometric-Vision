"""Calibrated pinhole rig, optical frames x-right y-down z-forward."""
import cv2
import numpy as np


class Rectifier:
    def __init__(self, k_left, k_right, d_left, d_right, rotation, translation, size):
        # X_right = rotation @ X_left + translation, in metres.
        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
            k_left, d_left, k_right, d_right, size, rotation, translation,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.rotation = r1
        self.k = p1[:, :3]
        self.baseline = -p2[0, 3] / p2[0, 0]
        self.maps = [cv2.initUndistortRectifyMap(k, d, r, p, size, cv2.CV_32FC1)
                     for k, d, r, p in ((k_left,d_left,r1,p1), (k_right,d_right,r2,p2))]

    def apply(self, left, right):
        return tuple(cv2.remap(img, *maps, cv2.INTER_LINEAR) for img, maps in zip((left,right), self.maps))


def obstacle_xyz(depth, k, height=1.5):
    """Find the nearest coherent elevated surface in a 2 m wide ego corridor."""
    v, u = np.indices(depth.shape, dtype=np.float32)
    x = (u-k[0,2])*depth/k[0,0]
    y = (v-k[1,2])*depth/k[1,1]
    candidate = (depth>1.5) & (depth<32) & (abs(x)<1.0) & (y<height-0.35) & (y>-0.5)
    # Quantize range before component analysis so surfaces at different depths stay separate.
    for z in range(2,32):
        mask = np.uint8(candidate & (depth>=z-0.5) & (depth<z+0.75))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count > 1:
            component = 1 + np.argmax(stats[1:,cv2.CC_STAT_AREA])
            if stats[component,cv2.CC_STAT_AREA] >= 18:
                selected = labels==component
                return tuple(float(np.median(axis[selected])) for axis in (x, y, depth))
    return None


def obstacle(depth, k, height=1.5):
    point = obstacle_xyz(depth, k, height)
    return None if point is None else (point[0], point[2])
