"""Train bounded SAC corrections to a camera-only pure-pursuit controller."""
import argparse
import json
from math import atan
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from driving_environment.route import RouteEnv, SCENES, pursuit
from sac_training.experiment import save_csv


OUT = Path("results/residual")
SEEDS = (37, 83, 109)
STAGE = 8192
DEFAULT = (6.0, 5.0, 0.0)
VALIDATION = (6000, 12)
TEST = (12000, 30)


def base_command(obs, settings):
    lookahead, cruise, slowdown = settings
    if tuple(settings) == DEFAULT:
        return pursuit(obs)
    offsets = obs[3:6] * [5, 8, 12]
    lateral = np.interp(lookahead, [3, 6, 10], offsets)
    steer = atan(2 * 2.6 * lateral / lookahead**2) / 0.52
    # Change in slope between two parts of the visible lane.
    bend = abs((offsets[2] - offsets[1]) / 4 - (offsets[1] - offsets[0]) / 3)
    target = max(3.0, cruise - slowdown * bend)
    return np.clip([steer, (target - 10 * obs[0]) / 2], -1, 1)


class ResidualEnv(gym.Wrapper):
    def __init__(self, settings):
        super().__init__(RouteEnv())
        self.settings = settings
        self.observation_space = spaces.Box(-1.0, 1.0, (13,), dtype=np.float32)

    def features(self, obs):
        route = self.unwrapped
        self.obs = np.asarray([*obs, float(route.speed_available), float(route.lane_available)], dtype=np.float32)
        return self.obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.features(obs), info

    def step(self, residual):
        route = self.unwrapped
        nominal = base_command(self.obs, self.settings)
        action = np.clip(nominal + np.asarray(residual) * [0.20, 0.45], -1, 1)
        progress_before = route.progress
        obs, _, done, timed_out, info = self.env.step(action)
        # All distances, errors and failure cues here come from the rendered cameras.
        reward = (route.progress - progress_before - .12 - .10 * abs(route.error)
                  - .08 * info["steering_change"] - 1.5 * (not route.readable)
                  - .025 * float(np.dot(residual, residual)) + 25 * info["success"]
                  - 25 * (info["departure"] or info["collision"]) - 12 * (timed_out and not done))
        return self.features(obs), float(reward), done, timed_out, info


def rollouts(settings, split, model=None, *, reference=False):
    first, count = split
    env = ResidualEnv(settings)
    zero = np.zeros(2, dtype=np.float32)
    rows = []
    for scene in SCENES:
        for seed in range(first, first + count):
            obs, _ = env.reset(seed=seed, options={"scene": scene})
            arrived = contact = departure = False
            errors = []
            while True:
                correction = zero if model is None else model.predict(obs, deterministic=True)[0]
                obs, _, done, timeout, info = env.step(correction)
                if reference:
                    error, hit, reached = env.unwrapped.reference_metrics()
                    contact |= hit
                    departure |= abs(error) > 2.6
                    arrived |= reached
                else:
                    error = info["error_m"]
                errors.append(error)
                if done or timeout:
                    break
            success = arrived and not (contact or departure) if reference else info["success"]
            collision = contact if reference else info["collision"]
            rows.append(dict(scene=scene, seed=seed, success=int(success), collision=int(collision),
                             rmse_m=float(np.sqrt(np.mean(np.square(errors)))), steps=len(errors)))
    env.close()
    return rows


def rank(rows):
    return (sum(r["success"] for r in rows), -sum(r["collision"] for r in rows),
            -float(np.mean([r["rmse_m"] for r in rows])))


def tune():
    candidates = [(look, speed, gain) for look in (4.5, 6.0, 7.5)
                  for speed, gain in ((5.0, 0.0), (4.0, 0.0), (5.0, 6.0))]
    records = []
    for candidate in candidates:
        scores = rank(rollouts(candidate, VALIDATION))
        records.append(dict(settings=candidate, rank=scores))
        print(candidate, scores, flush=True)
    winner = max(records, key=lambda record: record["rank"])
    (OUT/"baseline.json").write_text(json.dumps(dict(selected=winner["settings"], candidates=records), indent=2))


def train(seed, settings):
    folder = OUT/str(seed)
    folder.mkdir(exist_ok=True)
    env = Monitor(ResidualEnv(settings), str(folder/"episodes"))
    model = SAC("MlpPolicy", env, seed=seed, device="cpu", policy_kwargs=dict(net_arch=[64, 64]),
                buffer_size=65536, batch_size=128, learning_starts=512, train_freq=4,
                gradient_steps=2, ent_coef="auto_0.05", verbose=0)
    best = None
    checkpoints = []
    for stage in range(1, 5):
        model.learn(STAGE, reset_num_timesteps=False)
        score = rank(rollouts(settings, VALIDATION, model))
        checkpoints.append(dict(steps=stage * STAGE, validation_rank=score))
        if best is None or score > best:
            best = score
            model.save(folder/"selected")
        (folder/"validation.json").write_text(json.dumps(checkpoints, indent=2))
        print(seed, stage * STAGE, score, flush=True)
    env.close()


def test(settings):
    tests = []
    for name, config, model in [("original_pursuit", DEFAULT, None), ("tuned_pursuit", settings, None)]:
        tests.extend(dict(controller=name, training_seed="", **r) for r in rollouts(config, TEST, model, reference=True))
    for seed in SEEDS:
        model = SAC.load(OUT/str(seed)/"selected", device="cpu")
        tests.extend(dict(controller="residual_sac", training_seed=seed, **r)
                     for r in rollouts(settings, TEST, model, reference=True))
    save_csv(OUT/"heldout.csv", tests)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tune", "train", "test"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "tune":
        tune()
    else:
        settings = json.loads((OUT/"baseline.json").read_text())["selected"]
        if args.mode == "train":
            train(args.seed, settings)
        else:
            test(settings)
