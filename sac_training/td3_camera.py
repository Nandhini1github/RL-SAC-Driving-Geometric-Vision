"""Offline TD3 with a camera-only demonstration prior and a matched imitation control."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from driving_environment.route import RouteEnv, SCENES
from sac_training.experiment import save_csv
from sac_training.residual import SEEDS, base_command, rank, rollouts


OUT = Path("results/td3")
SETTINGS = (4.5, 4.0, 0.0)  # Published validation-selected geometric controller.
VALID = (6700, 10)
TEST = (22000, 60)
WARMUP = 2400
STAGES = 4
UPDATES = 1600


def collect():
    rng = np.random.default_rng(5221)
    env = RouteEnv()
    records = []
    for scene in SCENES:
        for i in range(64):
            obs, _ = env.reset(seed=2000+i, options={"scene": scene})
            # Nominal drives and correlated perturbations cover recovery and speed variation.
            nominal = i % 4 == 0
            settings = SETTINGS if nominal else (rng.uniform(3.5, 7), rng.uniform(3.1, 5.6), 0)
            noise = np.zeros(2)
            while True:
                teacher = base_command(obs, SETTINGS)
                noise = .86 * noise + rng.normal(0, [.055, .13])
                action = teacher if nominal else np.clip(base_command(obs, settings)+noise, -1, 1)
                progress = env.progress
                following, _, done, timeout, info = env.step(action)
                # Camera-measured progress and failure cues. No map, pose or contact labels.
                reward = (env.progress - progress - .12 - .1 * abs(env.error)
                          - .08 * info["steering_change"] - 1.5 * (not env.readable)
                          + 25 * info["success"] - 25 * (info["departure"] or info["collision"])
                          - 12 * (timeout and not done))
                records.append((obs, action, reward, following, done or timeout, teacher))
                obs = following
                if done or timeout:
                    break
    env.close()
    fields = ("obs", "act", "reward", "next_obs", "terminal", "teacher")
    arrays = {key:np.asarray(x, dtype=np.float32) for key,x in zip(fields, zip(*records))}
    np.savez_compressed(OUT/"training.npz", **arrays)
    print(f"Saved {len(records)} transitions from 192 camera-only episodes", flush=True)


class CameraActor(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.net = nn.Sequential(nn.Linear(11, 96), nn.ReLU(), nn.Linear(96, 96),
                                 nn.ReLU(), nn.Linear(96, 2), nn.Tanh())

    def forward(self, obs):
        return self.net(((obs-self.mean)/self.scale).clamp(-8, 8))

    @torch.no_grad()
    def predict(self, obs, deterministic=True):
        return self(torch.as_tensor(obs, dtype=torch.float32)).numpy(), None


def critic():
    return nn.Sequential(nn.Linear(13, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 1))


def scores(actor):
    return rank(rollouts(SETTINGS, VALID, actor, standalone=True))


def train(seed):
    torch.manual_seed(seed)
    folder = OUT/str(seed)
    folder.mkdir(exist_ok=True)
    with np.load(OUT/"training.npz") as archive:
        data = {k:torch.from_numpy(archive[k]) for k in archive.files}
    obs = data["obs"]
    mean, scale = obs.mean(0), obs.std(0).clamp_min(.05)
    actor = CameraActor(mean, scale)
    warm_optimizer = torch.optim.Adam(actor.parameters(), lr=.0008)
    for _ in range(WARMUP):
        idx = torch.randint(len(obs), (256,))
        loss = F.mse_loss(actor(obs[idx]), data["teacher"][idx])
        warm_optimizer.zero_grad()
        loss.backward()
        warm_optimizer.step()

    clone = deepcopy(actor)
    target_actor = deepcopy(actor)
    critics = nn.ModuleList([critic(), critic()])
    targets = deepcopy(critics)
    critic_opt = torch.optim.Adam(critics.parameters(), lr=.0005)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=.00015)
    clone_opt = torch.optim.Adam(clone.parameters(), lr=.00015)
    initial_score = scores(actor)
    best = {name:initial_score for name in ("td3","clone")}
    history = []
    for name in best:
        torch.save(actor.state_dict(), folder/f"{name}.pt")
        history.append(dict(arm=name, critic_updates=0, post_warmup_actor_updates=0,
                            validation_rank=initial_score, selected=True))

    for stage in range(1, STAGES+1):
        for step in range(UPDATES):
            idx = torch.randint(len(obs), (256,))
            current, following = obs[idx], data["next_obs"][idx]
            norm = ((current-mean)/scale).clamp(-8, 8)
            with torch.no_grad():
                noise = (torch.randn(len(idx), 2) * .12).clamp(-.25, .25)
                action = (target_actor(following) + noise).clamp(-1, 1)
                x = torch.cat((((following-mean)/scale).clamp(-8, 8), action), dim=1)
                q = torch.minimum(targets[0](x), targets[1](x)).flatten()
                target = data["reward"][idx] + .99 * (1-data["terminal"][idx]) * q
            x = torch.cat((norm, data["act"][idx]), dim=1)
            loss = sum(F.mse_loss(net(x).flatten(), target) for net in critics)
            critic_opt.zero_grad()
            loss.backward()
            critic_opt.step()
            if step % 2 == 1:
                output = actor(current)
                value = critics[0](torch.cat((norm, output), dim=1))
                teacher = data["teacher"][idx]
                policy_loss = F.mse_loss(output, teacher) - .35*value.mean()/value.detach().abs().mean().clamp_min(1)
                actor_opt.zero_grad()
                policy_loss.backward()
                actor_opt.step()
                clone_opt.zero_grad()
                F.mse_loss(clone(current), teacher).backward()
                clone_opt.step()
                with torch.no_grad():
                    for source, dest in ((actor, target_actor), (critics, targets)):
                        for p, t in zip(source.parameters(), dest.parameters()):
                            t.lerp_(p, .005)
        for name, policy in (("td3", actor), ("clone", clone)):
            score = scores(policy)
            selected = score > best[name]
            if selected:
                best[name] = score
                torch.save(policy.state_dict(), folder/f"{name}.pt")
            history.append(dict(arm=name, critic_updates=stage*UPDATES if name=="td3" else 0,
                                post_warmup_actor_updates=stage*UPDATES//2,
                                validation_rank=score, selected=selected))
        (folder/"validation.json").write_text(json.dumps(history, indent=2))
        print(seed, stage, "selected validation scores", best, flush=True)


def test(arm, seed):
    actor = None
    if arm == "residual":
        from stable_baselines3 import SAC
        actor = SAC.load(Path("results/residual")/str(seed)/"selected", device="cpu")
    elif arm != "pursuit":
        actor = CameraActor(torch.zeros(11), torch.ones(11))
        actor.load_state_dict(torch.load(OUT/str(seed)/f"{arm}.pt", weights_only=True))
        actor.eval()
    rows = rollouts(SETTINGS, TEST, actor, reference=True, standalone=arm in ("td3", "clone"))
    target = OUT/("pursuit.csv" if actor is None else f"{seed}/{arm}.csv")
    save_csv(target, rows)
    print(arm, seed, sum(r["success"] for r in rows), "/", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "train", "test"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=37)
    parser.add_argument("--arm", choices=("pursuit", "td3", "clone", "residual"), default="pursuit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "collect":
        collect()
    elif args.mode == "train":
        train(args.seed)
    else:
        test(args.arm, args.seed)
