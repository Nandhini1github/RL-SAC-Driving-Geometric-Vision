"""Shared driving teacher and paired camera-policy rollouts."""
import csv
from math import atan
import numpy as np
from driving_environment.route import RouteEnv, SCENES

SEEDS = (37, 83, 109)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def base_command(obs, settings):
    lookahead, cruise, slowdown = settings
    offsets = obs[3:6] * [5, 8, 12]
    lateral = np.interp(lookahead, [3, 6, 10], offsets)
    steer = atan(2 * 2.6 * lateral / lookahead**2) / 0.52
    # Change in slope between two parts of the visible lane.
    bend = abs((offsets[2] - offsets[1]) / 4 - (offsets[1] - offsets[0]) / 3)
    target = max(3.0, cruise - slowdown * bend)
    return np.clip([steer, (target - 10 * obs[0]) / 2], -1, 1)


def rollouts(settings, split, model=None, *, reference=False):
    first, count = split
    env = RouteEnv()
    rows = []
    for scene in SCENES:
        for seed in range(first, first + count):
            obs, _ = env.reset(seed=seed, options={"scene": scene})
            arrived = contact = departure = False
            first_arrival = None
            errors = []
            missing_speed_frames = 0
            while True:
                action = base_command(obs, settings) if model is None else model.predict(obs, deterministic=True)[0]
                obs, _, done, timeout, info = env.step(action)
                missing_speed_frames += int(not env.speed_available)
                if reference:
                    error, hit, reached = env.reference_metrics()
                    contact |= hit
                    departure |= abs(error) > 2.6
                    arrived |= reached
                    if reached and first_arrival is None:
                        first_arrival = round(0.1 * env.steps, 1)
                else:
                    error = info["error_m"]
                errors.append(error)
                if done or timeout:
                    break
            success = arrived and not (contact or departure) if reference else info["success"]
            collision = contact if reference else info["collision"]
            rows.append(dict(scene=scene, seed=seed, success=int(success), collision=int(collision),
                             rmse_m=float(np.sqrt(np.mean(np.square(errors)))), steps=len(errors),
                             first_arrival_s=first_arrival, missing_speed_frames=missing_speed_frames,
                             ended_on_missing_input=int(env.invalid >= 3)))
    env.close()
    return rows


def rank(rows):
    return (sum(r["success"] for r in rows), -sum(r["collision"] for r in rows),
            -float(np.mean([r["rmse_m"] for r in rows])))
