"""Rebuild the selected camera-history TD3 steering heads from saved offline transitions."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from td3_training.td3_camera import load_actor, SEEDS
from td3_training.tracking import LaneMemory, steer_net

CACHE = Path(".runtime/history-training.npz")


def prepare():
    with np.load(Path("results/td3/training.npz")) as data:
        raw = dict(data)
    current, following = [], []
    mem = LaneMemory()
    first = True
    for o,n,finished in zip(raw["obs"],raw["next_obs"],raw["terminal"]):
        if first:
            mem = LaneMemory()
            x = mem.update(o)
        xn = mem.update(n)
        current.append(x)
        following.append(xn)
        first = bool(finished)
        x = xn
    np.savez(CACHE, **dict(raw, obs=np.stack(current),next_obs=np.stack(following)))


def train(seed, output):
    torch.manual_seed(seed)
    with np.load(CACHE) as ds:
        t = {k:torch.from_numpy(ds[k]) for k in ds.files}
    dim = 14
    x, xn = t["obs"][:,:dim],t["next_obs"][:,:dim]
    mean, std = x.mean(0), x.std(0).clamp_min(.05)
    steering_labels = (torch.atan(2*2.6*(3*x[:,12:13])/3.5**2)/.52).clamp(-1,1)
    lateral_m = t["next_obs"][:,11]*3
    reward = t["reward"] - lateral_m.square()
    target_speed = load_actor("final", seed)
    with torch.no_grad():
        acc = target_speed(x[:,:11])[:,1:]
        accn = target_speed(xn[:,:11])[:,1:]
    x,xn = ((x-mean)/std).clamp(-8,8), ((xn-mean)/std).clamp(-8,8)
    net=steer_net(dim)
    opt=torch.optim.Adam(net.parameters(),lr=.0007)
    for _ in range(1600):
        ids=torch.randint(len(x),(256,))
        loss=F.mse_loss(net(x[ids]),steering_labels[ids])
        opt.zero_grad(); loss.backward(); opt.step()
    target=deepcopy(net)
    qs=nn.ModuleList([nn.Sequential(nn.Linear(dim+2,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU(),nn.Linear(96,1)) for _ in range(2)])
    qt=deepcopy(qs)
    opt=torch.optim.Adam(net.parameters(),lr=.00012)
    qo=torch.optim.Adam(qs.parameters(),lr=.0005)
    for step in range(3200):
        ids=torch.randint(len(x),(256,))
        with torch.no_grad():
            na=torch.cat((target(xn[ids]),accn[ids]),1)
            na=(na+(torch.randn_like(na)*.12).clamp(-.25,.25)).clamp(-1,1)
            z=torch.cat((xn[ids],na),1)
            backup=reward[ids]+.99*(1-t["terminal"][ids])*torch.minimum(qt[0](z),qt[1](z)).flatten()
        z=torch.cat((x[ids],t["act"][ids]),1)
        loss=sum(F.mse_loss(q(z).flatten(),backup) for q in qs)
        qo.zero_grad(); loss.backward(); qo.step()
        if step%2:
            steer=net(x[ids])
            q=qs[0](torch.cat((x[ids],steer,acc[ids]),1))
            loss=F.mse_loss(steer,steering_labels[ids])-.12*q.mean()/q.detach().abs().mean().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for a,b in ((net,target),(qs,qt)):
                    for ap,bp in zip(a.parameters(),b.parameters()):
                        bp.lerp_(ap,.005)
    torch.save(dict(state=net.state_dict(),mean=mean,std=std),output/f"steering-{seed}.pt")
    print("Trained TD3 steering", seed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".runtime/retrained"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    CACHE.parent.mkdir(exist_ok=True)
    args.output.mkdir(exist_ok=True, parents=True)
    prepare()
    for seed in SEEDS:
        train(seed, args.output)
