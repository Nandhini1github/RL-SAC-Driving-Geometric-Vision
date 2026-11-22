"""Online TD3 on camera measurements from the Gymnasium driving environment."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from driving_environment.route import RouteEnv, SCENES
from td3_training.routes import SEEDS, save_csv
from td3_training.tracking import LaneMemory

PUBLISHED = Path("results/online_td3")
VALIDATION = range(57000, 57008)
HOLDOUT = range(61000, 61200)
STAGE = 16_000
BUDGET = 48_000


class CameraHistory(gym.Wrapper):
    def __init__(self):
        super().__init__(RouteEnv())
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (14,), dtype=np.float32)

    def reset(self, **kwargs):
        self.memory = LaneMemory()
        obs, info = self.env.reset(**kwargs)
        return self.memory.update(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self.memory.update(obs), reward, done, truncated, info


def rollouts(models, seeds, *, reference):
    rows = []
    env = CameraHistory()
    try:
        for scene in SCENES:
            for seed in seeds:
                obs, _ = env.reset(seed=seed, options={"scene": scene})
                errors = []
                contact = departed = reached = False
                while True:
                    actions = [model.predict(obs, deterministic=True)[0] for model in models]
                    action = np.median(actions, axis=0)
                    obs, _, done, truncated, info = env.step(action)
                    if reference:
                        error, hit, finish = env.unwrapped.reference_metrics()
                        outside = abs(error) > 2.6
                    else:
                        error, hit = info["error_m"], info["collision"]
                        finish, outside = info["success"], info["departure"]
                    errors.append(error)
                    contact |= hit
                    departed |= outside
                    reached |= finish
                    if done or truncated:
                        break
                rows.append(dict(scene=scene, seed=seed, steps=len(errors),
                                 success=int(reached and not (contact or departed)),
                                 contact=int(contact), departure=int(departed), reached=int(reached),
                                 rmse_m=float(np.sqrt(np.mean(np.square(errors))))))
    finally:
        env.close()
    return rows


def summarize(rows):
    return dict(routes=len(rows), finishes=sum(row["success"] for row in rows),
                contacts=sum(row["contact"] for row in rows),
                mean_rmse_m=float(np.mean([row["rmse_m"] for row in rows])))


def configure_cpu():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def train(seed, output):
    configure_cpu()
    env = CameraHistory()
    try:
        model = TD3("MlpPolicy", env, device="cpu", seed=seed, learning_rate=0.0003,
                    buffer_size=60_000, learning_starts=2_000, batch_size=128, gamma=.99, tau=.005,
                    train_freq=(4, "step"), gradient_steps=4, policy_delay=2,
                    action_noise=NormalActionNoise(np.zeros(2), np.full(2, .20)),
                    target_policy_noise=.2, target_noise_clip=.5, policy_kwargs={"net_arch": [128, 128]})
        history = []
        best = None
        for steps in range(STAGE, BUDGET + 1, STAGE):
            model.learn(total_timesteps=STAGE, reset_num_timesteps=(steps == STAGE))
            stats = summarize(rollouts([model], VALIDATION, reference=False))
            history.append(dict(steps=steps, **stats))
            rank = (stats["finishes"], -stats["contacts"], -stats["mean_rmse_m"])
            if best is None or rank > best:
                best, selected = rank, steps
                model.save(output / f"online-{seed}.zip")
            log = dict(seed=seed, selected=selected, validation=history)
            (output / f"training-{seed}.json").write_text(json.dumps(log, indent=2))
            print(f"Seed {seed}, {steps} interactions, "
                  f"{stats['finishes']}/{stats['routes']} visual finishes", flush=True)
    finally:
        env.close()


def score_batch(models_dir, seeds):
    configure_cpu()
    models = [TD3.load(models_dir / f"online-{seed}.zip", device="cpu") for seed in SEEDS]
    return rollouts(models, seeds, reference=True)


def plot_comparison(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    expected = pd.MultiIndex.from_product([SCENES, HOLDOUT], names=["scene", "seed"]).sort_values()
    sources = [
        ("Offline TD3", Path("results/tracking/td3.csv"), "#16766f"),
        ("Pure pursuit", Path("results/tracking/pure_pursuit.csv"), "#64778f"),
        ("Online TD3", output / "online_td3.csv", "#aa5715"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.1), layout="constrained")
    x = np.arange(len(SCENES))
    for i, (name, path, color) in enumerate(sources):
        table = pd.read_csv(path).set_index(["scene", "seed"]).sort_index()
        if not table.index.equals(expected):
            raise ValueError(f"{path} has different route IDs from the benchmark")
        groups = table.groupby(level="scene")
        finishes = groups.success.sum().loc[list(SCENES)]
        errors_cm = 100 * groups.rmse_m.mean().loc[list(SCENES)]
        for ax, values, fmt in zip(axes, (finishes, errors_cm), ("%.0f", "%.1f")):
            bars = ax.bar(x + (i - 1)*.26, values, .245, label=name, color=color)
            ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)
    for ax, title in zip(axes, ("Clean finishes out of 200", "Mean lateral tracking RMSE (cm)")):
        ax.set(xticks=x, xticklabels=["Straight", "Bend", "Obstacle bypass"], title=title)
        ax.margins(y=.14)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
               ncol=3, frameon=False)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def score(models_dir, output):
    batches = [range(i, i + 50) for i in HOLDOUT[::50]]
    with ProcessPoolExecutor(len(batches), mp_context=get_context("spawn")) as pool:
        chunks = pool.map(score_batch, [models_dir] * len(batches), batches)
        rows = [row for chunk in chunks for row in chunk]
    rows.sort(key=lambda row: (SCENES.index(row["scene"]), row["seed"]))
    save_csv(output / "online_td3.csv", rows)
    result = dict(all=summarize(rows), by_scene={
        scene: summarize([row for row in rows if row["scene"] == scene]) for scene in SCENES})
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_comparison(output)
    print(json.dumps(result["all"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/online-td3"))
    parser.add_argument("--models", type=Path, default=PUBLISHED, help="Checkpoints for evaluate")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        with ProcessPoolExecutor(len(SEEDS), mp_context=get_context("spawn")) as pool:
            list(pool.map(train, SEEDS, [args.output] * len(SEEDS)))
    else:
        score(args.models, args.output)
