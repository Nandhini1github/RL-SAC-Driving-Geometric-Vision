import numpy as np
from driving_environment.route import SCENES


def propose(rows):
    """Upweight one validation stratum. No access to code, rewards or test seeds."""
    scores = {}
    for scene in SCENES:
        cases = [r for r in rows if r["scene"] == scene]
        scores[scene] = float(np.mean([2*r["collision"] + (1-r["success"]) + r["rmse_m"]/3 for r in cases]))
    worst = max(scores, key=scores.get)
    return dict(scores=scores, target=worst, weights=[0.6 if s == worst else 0.2 for s in SCENES])
