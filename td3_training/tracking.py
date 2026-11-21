"""Camera lane history and the two deployed tracking controllers."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from td3_training.td3_camera import MedianActor, SEEDS

OUT = Path("results/tracking")


class LaneMemory:
    """Transport past 6 m camera lane samples using visual speed and previous steering."""
    def __init__(self):
        self.points = np.empty((0, 2))
    def update(self, obs):
        distance = float(obs[0])  # measured speed * 0.1 seconds
        yaw = distance * np.tan(float(obs[9]) * .52) / 2.6
        cs, sn = np.cos(yaw), np.sin(yaw)
        p = self.points
        if len(p):
            p = (p - [distance, 0.]) @ np.array([[cs,-sn],[sn,cs]])
            p = p[(p[:,0] > -1.) & (p[:,0] < 6.5)]
        y6 = float(obs[4])*8
        if abs(float(obs[1])) < .99:
            p = p[p[:,0] < 5.88]
            p = np.vstack((p, [6.,y6]))
        self.points = p
        d = np.array([0., 3.5, 4.])
        left = np.interp(d, [0,3,6], [-obs[1]*3,obs[3]*5,y6])
        if len(p):
            ordered = p[np.argsort(p[:,0])]
            valid = (d >= ordered[0,0]) & (d <= ordered[-1,0])
            left[valid] = np.interp(d[valid], ordered[:,0], ordered[:,1])
        return np.asarray([*obs, *np.clip(left/3.,-1.,1.)],np.float32)

def steer_net(n):
    return nn.Sequential(nn.Linear(n,96),nn.ReLU(),nn.Linear(96,96),nn.ReLU(),nn.Linear(96,1),nn.Tanh())

class TD3:
    def __init__(self):
        self.speed_policy = MedianActor("final")
        self.steering = []
        for seed in SEEDS:
            weights = torch.load(OUT/f"steering-{seed}.pt", weights_only=True)
            net = steer_net(14)
            net.load_state_dict(weights["state"])
            self.steering.append((net.eval(), weights["mean"], weights["std"]))
        self.reset()

    def reset(self):
        self.memory = LaneMemory()

    @torch.no_grad()
    def predict(self, obs):
        features = torch.from_numpy(self.memory.update(obs))
        requests = [net(((features-mean)/scale).clamp(-8,8)).item()
                    for net, mean, scale in self.steering]
        action, _ = self.speed_policy.predict(obs)
        action[0] = np.median(requests)
        return action


class PurePursuit:
    def __init__(self):
        self.reset()

    def reset(self):
        self.memory = LaneMemory()

    def predict(self, obs):
        features = self.memory.update(obs)
        # Historical lane offset is normalized by 3 m; the steering lookahead is 3.5 m.
        steer = np.arctan(2*2.6*3*features[12]/3.5**2)/.52
        longitudinal = (4.0 - 10*obs[0])/2
        return np.clip(np.array([steer, longitudinal], dtype=np.float64), -1, 1)
