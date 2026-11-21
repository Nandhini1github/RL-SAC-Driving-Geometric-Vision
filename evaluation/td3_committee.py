"""Evaluate one frozen median-action controller and matched controls on unused routes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sac_training.experiment import save_csv
from sac_training.residual import rollouts
from sac_training.td3_camera import SEEDS, SETTINGS, load_actor


OUT = Path("results/td3/committee")
TEST = (45000, 200)
ARMS = ("td3_median", "imitation_median", "pursuit", "slower_pursuit", *(f"td3_{s}" for s in SEEDS))
PRIMARY = ("pursuit", "slower_pursuit", "imitation_median")


class MedianActor:
    def __init__(self, branch):
        self.actors = [load_actor(branch, seed) for seed in SEEDS]

    def predict(self, obs, deterministic=True):
        actions = [actor.predict(obs)[0] for actor in self.actors]
        return np.median(actions, axis=0).astype(np.float32), None


def evaluate(arm):
    settings = (4.5, 3.75, 0.) if arm == "slower_pursuit" else SETTINGS
    model = None
    if arm == "td3_median":
        model = MedianActor("final")
    elif arm == "imitation_median":
        model = MedianActor("clone")
    elif arm.startswith("td3_"):
        model = load_actor("final", int(arm.split("_")[-1]))
    rows = rollouts(settings, TEST, model, reference=True, standalone=model is not None)
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUT/f"{arm}.csv", rows)
    print(arm, sum(r["success"] for r in rows), len(rows), flush=True)


def summarize():
    tables = {arm:pd.read_csv(OUT/f"{arm}.csv").set_index(["seed","scene"]).sort_index() for arm in ARMS}
    reference = tables["td3_median"]
    expected = pd.MultiIndex.from_product([range(TEST[0],TEST[0]+TEST[1]), ["bend","obstacle","straight"]],
                                          names=["seed","scene"])
    for table in tables.values():
        assert table.index.equals(expected)
    totals = {arm:dict(finishes=int(t.success.sum()), episodes=len(t), contacts=int(t.collision.sum()),
                       mean_episode_rmse_m=float(t.rmse_m.mean()), camera_loss_endings=int(t.ended_on_missing_input.sum()),
                       finishes_by_scene=t.groupby(level="scene").success.sum().to_dict()) for arm,t in tables.items()}
    rng = np.random.default_rng(37419)
    indices = rng.integers(0,TEST[1],(20000,TEST[1]))
    comparisons = {}
    for arm in PRIMARY:
        other = tables[arm]
        diff = reference.success - other.success
        # All three scenes for a route seed stay in the same bootstrap block.
        gains = 100*diff.groupby(level="seed").mean().to_numpy()[indices].mean(axis=1)
        interval = np.quantile(gains,[.025/len(PRIMARY),1-.025/len(PRIMARY)])
        joint = reference.success.eq(1) & other.success.eq(1)
        comparisons[arm] = dict(gain_pp=float(100*diff.mean()), simultaneous_interval_pp=interval.tolist(),
                                td3_only_finishes=int(diff.eq(1).sum()), other_only_finishes=int(diff.eq(-1).sum()),
                                joint_finishes=int(joint.sum()), paired_arrival_difference_s=float(
                                    (reference.first_arrival_s[joint]-other.first_arrival_s[joint]).mean()))
    passed = all(v["simultaneous_interval_pp"][0]>0 and totals["td3_median"]["contacts"]<=totals[k]["contacts"]
                 for k,v in comparisons.items())
    report = dict(controllers=totals, paired_td3_minus_control=comparisons, meets_criterion=bool(passed))
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("arm",choices=(*ARMS,"summary"))
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.arm=="summary":
        summarize()
    else:
        evaluate(args.arm)
