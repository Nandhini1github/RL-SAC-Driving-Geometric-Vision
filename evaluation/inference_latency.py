"""Measure single-observation CPU policy latency on a shared recorded camera route."""
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import SAC

from driving_environment.route import RouteEnv, pursuit


def camera_observations(seed):
    env = RouteEnv()
    observations = []
    obs, _ = env.reset(seed=seed, options={"scene": "obstacle"})
    while True:
        observations.append(obs)
        obs, _, ended, timed_out, _ = env.step(pursuit(obs))
        if ended or timed_out:
            env.close()
            return observations


def main():
    torch.set_num_threads(1)
    seed = 8002
    inputs = camera_observations(seed)
    arms = ("fixed", "adaptive")
    policies = {arm: SAC.load(Path("results/control") / arm, device="cpu") for arm in arms}
    times = {arm: [] for arm in arms}
    warmup_passes, measured_passes = 2, 20
    for trial in range(warmup_passes + measured_passes):
        for arm in (arms if trial % 2 == 0 else reversed(arms)):
            predict = policies[arm].predict
            for obs in inputs:
                start = perf_counter_ns()
                predict(obs, deterministic=True)
                elapsed = perf_counter_ns() - start
                if trial >= warmup_passes:
                    times[arm].append(elapsed / 1_000_000)

    report = {
        "operation": "Stable-Baselines3 predict on one precomputed camera observation",
        "device": "cpu", "torch_threads": torch.get_num_threads(), "batch_size": 1,
        "platform": platform.platform(), "processor": platform.processor(),
        "torch": str(torch.__version__), "stable_baselines3": stable_baselines3.__version__,
        "training_seed": 37, "route_seed": seed, "observation_source": "pure pursuit replay",
        "observations_per_pass": len(inputs), "warmup_passes": warmup_passes,
        "measured_passes": measured_passes, "policies": {},
    }
    for arm, samples in times.items():
        report["policies"][arm] = {
            "calls": len(samples), "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
        }
    output = json.dumps(report, indent=2)
    Path("results/inference_latency.json").write_text(output + "\n", encoding="utf8")
    print(output)


if __name__ == "__main__":
    main()
