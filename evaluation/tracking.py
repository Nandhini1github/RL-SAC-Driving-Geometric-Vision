"""Reproduce the paired Gymnasium benchmark for the two lane-history controllers."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from driving_environment.route import RouteEnv, SCENES
from td3_training.routes import save_csv
from td3_training.tracking import TD3, PurePursuit, OUT

CONTROLLERS = {"td3": TD3, "pure_pursuit": PurePursuit}
NAMES = ("TD3", "Pure pursuit")
ROUTES = range(61000, 61200)


def evaluate(name, directory, seeds=ROUTES):
    controller = CONTROLLERS[name]()
    env = RouteEnv()
    rows = []
    try:
        for scene in SCENES:
            for seed in seeds:
                controller.reset()
                obs, _ = env.reset(seed=seed, options={"scene": scene})
                errors = []
                contact = departure = arrived = False
                while True:
                    obs, _, done, timed_out, _ = env.step(controller.predict(obs))
                    error, hit, reached = env.reference_metrics()
                    errors.append(error)
                    contact |= hit
                    departure |= abs(error) > 2.6
                    arrived |= reached
                    if done or timed_out:
                        break
                rows.append(dict(scene=scene, seed=seed, steps=env.steps,
                                 success=int(arrived and not (contact or departure)),
                                 contact=int(contact), rmse_m=np.sqrt(np.mean(np.square(errors)))))
    finally:
        env.close()
    directory.mkdir(parents=True, exist_ok=True)
    save_csv(directory/f"{name}.csv", rows)


def summarize(directory):
    cases = pd.MultiIndex.from_product([SCENES, ROUTES], names=["scene", "seed"])
    tables = []
    totals = {}
    for key, name in zip(CONTROLLERS, NAMES):
        table = pd.read_csv(directory/f"{key}.csv").set_index(["scene", "seed"]).sort_index()
        if not table.index.equals(cases.sort_values()):
            raise ValueError(f"{name} must contain exactly the 600 specified cases")
        tables.append(table)
        totals[name] = dict(routes=len(table), finishes=int(table.success.sum()),
                            contacts=int(table.contact.sum()), lateral_rmse_m=float(table.rmse_m.mean()))
    td3, pursuit = tables
    both = td3.success.eq(1) & pursuit.success.eq(1)
    paired = dict(joint_finishes=int(both.sum()),
                  joint_lateral_rmse_m={name: float(table.rmse_m[both].mean())
                                        for name, table in zip(NAMES, tables)})
    # Keep all three scenes together when resampling a route seed.
    draws = np.random.default_rng(57000).integers(0, len(ROUTES), (20000, len(ROUTES)))
    for metric in ("success", "rmse_m"):
        differences = (td3[metric] - pursuit[metric]).groupby(level="seed").mean().to_numpy()
        paired[metric] = dict(td3_minus_pure_pursuit=float(differences.mean()),
                              adjusted_interval=np.quantile(differences[draws].mean(axis=1),
                                                            [.0125, .9875]).tolist())
    (directory/"summary.json").write_text(json.dumps(dict(controllers=totals, paired=paired), indent=2)+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, metric, title in zip(axes, ("success", "rmse_m"), ("Unfinished routes", "Mean lateral RMSE (cm)")):
        x = np.arange(len(SCENES))
        for table, name, offset, color in zip(tables, NAMES, (-.2, .2), ("#16766f", "#64778f")):
            grouped = table[metric].groupby(level="scene")
            values = (len(ROUTES) - grouped.sum() if metric == "success" else 100*grouped.mean()).loc[list(SCENES)]
            bars = ax.bar(x+offset, values, .38, label=name, color=color)
            ax.bar_label(bars, fmt="%.0f" if metric == "success" else "%.1f", fontsize=9)
        ax.set(xticks=x, xticklabels=["Straight", "Bend", "Obstacle"], title=title)
        ax.margins(y=.18)
    axes[0].legend(frameon=False)
    fig.savefig(directory/"comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[*CONTROLLERS, "summary"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "summary":
        summarize(args.output)
    else:
        evaluate(args.command, args.output)
