"""Short-route simulator. Distances in metres; positive y and steer point left."""
from math import atan, atan2, cos, exp, hypot, sin, tan

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SCENES = ("straight", "bend", "obstacle")


class RouteEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (11,), dtype=np.float32)
        self.weights = np.full(3, 1 / 3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.scene = (options or {}).get("scene") or self.np_random.choice(SCENES, p=self.weights)
        self.side = self.np_random.choice([-1, 1])
        self.ox = self.np_random.uniform(17, 23)
        self.x, self.y = 0.0, self.np_random.uniform(-0.6, 0.6)
        self.yaw = self.np_random.uniform(-0.08, 0.08)
        self.speed = self.np_random.uniform(2, 4)
        self.steps = 0
        self.prev = np.zeros(2)
        return self.observe(), {}

    def path(self, x):
        if self.scene == "obstacle":
            return self.side * 2.5 * exp(-((x - self.ox) / 6.5)**2)
        if self.scene == "bend":
            return self.side * 2.8 * sin(x / 9)
        return 0.0

    def observe(self):
        slope = (self.path(self.x + 0.1) - self.path(self.x - 0.1)) / 0.2
        self.error = (self.y - self.path(self.x)) / hypot(1, slope)
        yaw_error = self.yaw - atan(slope)
        def lateral(d):
            return -sin(self.yaw) * d + cos(self.yaw) * (self.path(self.x+d) - self.y)
        visible = self.scene == "obstacle" and self.x < self.ox
        distance = hypot(self.ox-self.x, self.y) if visible else 40
        bearing = atan2(-self.y, self.ox-self.x) - self.yaw if visible else 0
        values = [self.speed/10, self.error/3, sin(yaw_error), lateral(3)/5, lateral(6)/8,
                  lateral(10)/12, distance/40, sin(bearing), float(visible), *self.prev]
        return np.clip(values, -1, 1).astype(np.float32)

    def step(self, action):
        steer, accel = np.clip(action, -1, 1)
        last_x = self.x
        self.x += 0.1 * self.speed * cos(self.yaw)
        self.y += 0.1 * self.speed * sin(self.yaw)
        self.yaw += 0.1 * self.speed * tan(0.52 * steer) / 2.6
        self.speed = float(np.clip(self.speed + 0.3 * accel, 0, 9))
        delta = abs(float(steer) - self.prev[0])
        self.prev = np.array([steer, accel])
        self.steps += 1
        obs = self.observe()
        collision = self.scene == "obstacle" and hypot(self.x-self.ox, self.y) < 1.35
        departure = abs(self.error) > 2.6
        success = self.x >= 40 and not (collision or departure)
        reward = (self.x - last_x - 0.65*self.error**2 - 0.5*obs[2]**2
                  - 0.055*(self.speed-5)**2 - 0.1*delta - 0.03*accel**2
                  - 25*(collision or departure) + 15*success)
        info = dict(scene=str(self.scene), collision=bool(collision), success=bool(success),
                    departure=bool(departure), error_m=float(self.error), steering_change=float(delta))
        return obs, float(reward), bool(collision or departure or success), self.steps >= 160, info


def pursuit(obs):
    # Body-frame lookahead lateral offset and approximately six-metre chord.
    steer = atan(2 * 2.6 * obs[4] * 8 / 36) / 0.52
    accel = (5 - 10 * obs[0]) / 2
    return np.clip([steer, accel], -1, 1)
