"""Independent reference scoring for visual speed and lane offset on held-out renders."""
import json
from pathlib import Path

import numpy as np
from driving_environment.route import RouteEnv, SCENES, pursuit


def main():
    env = RouteEnv()
    rows = []
    for scene in SCENES:
        speed_errors, lane_errors = [], []
        count = 0
        for seed in range(8000,8004):
            obs, _ = env.reset(seed=seed,options={"scene": scene})
            while True:
                obs, _, term, trunc, _ = env.step(pursuit(obs))
                count += 1
                if env.readable:
                    ref_error, _, _ = env.reference_metrics()
                    speed_errors.append(env.measured_speed-env.speed)
                    lane_errors.append(env.error-ref_error)
                if term or trunc: break
        rows.append(dict(scene=scene,frames=count,valid=len(speed_errors),
                         speed_rmse_mps=float(np.sqrt(np.mean(np.square(speed_errors)))),
                         offset_rmse_m=float(np.sqrt(np.mean(np.square(lane_errors))))))
    Path("results/camera_accuracy.json").write_text(json.dumps(rows,indent=2))
    print(rows)


if __name__=="__main__":
    main()
