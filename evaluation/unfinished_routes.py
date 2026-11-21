"""Replay unsuccessful held-out episodes and distinguish the end signal from the score."""
import csv
from collections import Counter, deque
from pathlib import Path

import torch
from stable_baselines3 import SAC

from driving_environment.route import RouteEnv, pursuit


def end_reason(env, info, truncated, recent):
    if info["collision"]:
        return "visual proximity warning"
    if info["departure"]:
        if env.invalid < 3:
            return "visual lane offset over limit"
        if all(lane for speed, lane in recent):
            return "missing visual speed"
        if all(speed for speed, lane in recent):
            return "missing lane fit"
        return "missing speed and lane across three frames"
    if info["success"]:
        return "visual finish"
    if truncated:
        return "time limit"
    raise ValueError("Unrecognized end condition")


def main():
    torch.set_num_threads(1)
    rows = []
    for training_seed in (37, 83, 109):
        folder = Path("results/control" if training_seed == 37 else f"results/control_seed{training_seed}")
        with (folder/"heldout.csv").open(newline="") as source:
            failures = [case for case in csv.DictReader(source) if case["success"] == "0"]
        for arm in ("pursuit", "fixed", "adaptive"):
            if arm == "pursuit" and training_seed != 37:
                continue
            model = None if arm == "pursuit" else SAC.load(folder/arm, device="cpu")
            for case in (r for r in failures if r["controller"] == arm):
                env = RouteEnv()
                obs, _ = env.reset(seed=int(case["seed"]), options={"scene": case["scene"]})
                recent = deque(maxlen=3)
                hit = offroad = arrived = False
                while True:
                    action = pursuit(obs) if model is None else model.predict(obs, deterministic=True)[0]
                    obs, _, term, trunc, info = env.step(action)
                    error, contact, reached = env.reference_metrics()
                    arrived |= reached
                    offroad |= abs(error) > 2.6
                    hit |= contact
                    recent.append((env.speed_available, env.lane_available))
                    if term or trunc:
                        break
                assert not (arrived and not (hit or offroad)), case
                assert (int(hit), int(offroad)) == (int(case["collision"]), int(case["departure"])), case
                rows.append(dict(controller=arm, training_seed=training_seed if model else "",
                                 scene=case["scene"], route_seed=case["seed"],
                                 end_signal=end_reason(env, info, trunc, recent),
                                 final_x_m=round(env.x, 3), visual_progress_m=round(env.progress, 3),
                                 reached_finish=int(arrived), contact=int(hit), lane_departure=int(offroad)))
                env.close()
    with Path("results/unfinished_routes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(Counter((r["controller"],r["end_signal"]) for r in rows))


if __name__ == "__main__":
    main()
