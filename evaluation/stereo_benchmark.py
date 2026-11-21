"""Seeded frontoparallel plane experiment. Run as a module from the repo root."""
import json
import platform
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from geometric_vision.stereo import estimate, make_matcher


def main():
    rng = np.random.default_rng(2027)
    cv2.setNumThreads(1)
    h, w = 240, 480
    focal, baseline = 360.0, 0.24
    matcher = make_matcher()
    rows = []
    for z in (4.0, 8.0, 16.0):
        errors, times = [], []
        for i in range(11):
            left = rng.integers(0, 256, (h, w), dtype=np.uint8)
            shift = np.float32([[1, 0, -focal * baseline / z], [0, 1, 0]])
            right = cv2.warpAffine(left, shift, (w, h))
            right = np.clip(right + rng.normal(0, 2, (h, w)), 0, 255).astype(np.uint8)
            start = perf_counter()
            depth = estimate(left, right, matcher, focal, baseline)
            elapsed = (perf_counter() - start) * 1000
            if i:  # Exclude one warm-up frame per plane.
                errors.append(depth - z)
                times.append(elapsed)
        err = np.stack(errors)
        valid = err[np.isfinite(err)]
        coverage = valid.size / err.size
        if coverage < 0.75 or np.median(np.abs(valid)) > 0.06 * z:
            raise RuntimeError(f"Depth regression at {z} m")
        rows.append(dict(depth_m=z, coverage=coverage, mae_m=float(np.mean(np.abs(valid))),
                         rmse_m=float(np.sqrt(np.mean(valid**2))), median_ms=float(np.median(times))))
    report = dict(dataset="synthetic rectified textured planes", frames_per_depth=10,
                  width=w, height=h, fx_px=focal, baseline_m=baseline, seed=2027,
                  python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                  platform=platform.system(), rows=rows)
    output = Path("results/stereo_synthetic.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output.read_text())


if __name__ == "__main__":
    main()
