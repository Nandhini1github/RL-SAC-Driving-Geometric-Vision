"""Paired, equal-budget curriculum pilot. Run from repository root."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from driving_environment.route import RouteEnv, SCENES, pursuit
from agentic_supervisor.curriculum import propose

OUT = Path("results/control")
SEED = 37
STAGE = 4096


def evaluate(model, first_seed, per_scene):
    rows = []
    env = RouteEnv()
    for scene in SCENES:
        for seed in range(first_seed, first_seed+per_scene):
            obs, _ = env.reset(seed=seed, options={"scene":scene})
            errors, changes, latency = [], [], []
            total = 0
            while True:
                start = perf_counter_ns()
                action = pursuit(obs) if model is None else model.predict(obs, deterministic=True)[0]
                latency.append((perf_counter_ns()-start)/1e6)
                obs, r, term, trunc, info = env.step(action)
                errors.append(info["error_m"])
                changes.append(info["steering_change"])
                total += r
                if term or trunc:
                    break
            rows.append(dict(scene=scene, seed=seed, success=int(info["success"]), collision=int(info["collision"]),
                             departure=int(info["departure"]), rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                             steering_delta=float(np.mean(changes)), return_sum=total, latency_ms=float(np.median(latency))))
    return rows


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    check_env(RouteEnv())
    base = SAC("MlpPolicy", Monitor(RouteEnv(), str(OUT/"shared")), policy_kwargs=dict(net_arch=[64,64]),
               buffer_size=32768, learning_starts=256, batch_size=128, train_freq=4, gradient_steps=2,
               ent_coef="auto_0.1", seed=SEED, device="cpu")
    base.learn(STAGE)
    base.save(OUT/"shared")
    base.save_replay_buffer(OUT/"shared_buffer.pkl")
    validation = evaluate(base, 700, 4)
    curves, decisions, tests = [], [], []
    for name in ("pursuit", "fixed", "adaptive"):
        model = None
        if name != "pursuit":
            env = RouteEnv()
            model = SAC.load(OUT/"shared", env=Monitor(env,str(OUT/name)), device="cpu")
            model.load_replay_buffer(OUT/"shared_buffer.pkl")
            model.set_random_seed(SEED+1)
            rows = validation
            for stage in range(3):
                curves.extend(dict(arm=name, steps=STAGE*(stage+1), **r) for r in rows)
                if stage == 2:
                    break
                decision = propose(rows) if name == "adaptive" else dict(weights=[1/3]*3)
                decisions.append(dict(arm=name, stage=stage+1, **decision))
                env.weights = decision["weights"]
                model.learn(STAGE, reset_num_timesteps=False)
                rows = evaluate(model, 700, 4)
            model.save(OUT/name)
        tests.extend(dict(controller=name, **r) for r in evaluate(model, 8000, 20))
        print("Finished", name, flush=True)
    save_csv(OUT/"heldout.csv", tests)
    save_csv(OUT/"validation.csv", curves)
    (OUT/"decisions.json").write_text(json.dumps(decisions, indent=2))
    print("Saved held-out comparison", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    SEED = args.seed
    if SEED != 37:
        OUT = Path(f"results/control_seed{SEED}")
    main()
