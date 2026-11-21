"""Short-route simulator. Distances in metres; positive y and steer point left."""
from math import atan, atan2, cos, exp, hypot, sin, tan

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
from driving_environment.cameras import RoadCameras
from driving_environment.camera_features import CameraFeatures

SCENES = ("straight", "bend", "obstacle")


class RouteEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (11,), dtype=np.float32)
        cv2.setNumThreads(1)
        self.weights = np.full(3, 1 / 3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.scene = (options or {}).get("scene") or self.np_random.choice(SCENES, p=self.weights)
        self.side = self.np_random.choice([-1, 1])
        self.ox = self.np_random.uniform(17, 23)
        self.x, self.y = 0.0, self.np_random.uniform(-0.6, 0.6)
        self.yaw = self.np_random.uniform(-0.08, 0.08)
        self.speed = 0.
        self.progress = 0.
        self.invalid = 0
        self.cameras = RoadCameras(self.np_random,self.path,self.ox if self.scene=="obstacle" else None)
        self.vision = CameraFeatures()
        self.steps = 0
        self.prev = np.zeros(2)
        left,right=self.cameras.pair(self.x,self.y,self.yaw)
        self.vision.motion.update(left,-0.1)
        return self.observe(left,right), {}

    def path(self, x):
        if self.scene == "obstacle":
            return self.side * 2.5 * exp(-((x - self.ox) / 6.5)**2)
        if self.scene == "bend":
            return self.side * 2.8 * sin(x / 9)
        return 0.0

    def observe(self,left,right):
        speed,lane,point=self.vision.measure(left,right,self.steps*.1)
        self.readable = speed is not None and lane is not None
        self.invalid = 0 if self.readable else self.invalid+1
        self.measured_speed = 0. if speed is None else min(speed,12.)
        self.error,heading,look = lane if lane is not None else (3.,0.,[0.,0.,0.])
        self.near_hazard = False
        distance,bearing = 40.,0.
        if point is not None:
            z,x=point
            distance=hypot(z+1.6,x)
            bearing=atan2(-x,z+1.6)
            self.near_hazard = z<2.4 and abs(x)<1.
        values=[self.measured_speed/10,self.error/3,sin(heading),look[0]/5,look[1]/8,
                look[2]/12,distance/40,sin(bearing),float(point is not None),*self.prev]
        return np.clip(values,-1,1).astype(np.float32)

    def reference_metrics(self):
        """Ground truth for final scoring only. Never called by reset or step."""
        slope=(self.path(self.x+.1)-self.path(self.x-.1))/.2
        err=(self.y-self.path(self.x))/hypot(1,slope)
        collision=self.scene=="obstacle" and hypot(self.x-self.ox,self.y)<1.35
        return float(err),bool(collision),self.x>=39.

    def step(self, action):
        steer, accel = np.clip(action, -1, 1)
        self.x += 0.1 * self.speed * cos(self.yaw)
        self.y += 0.1 * self.speed * sin(self.yaw)
        self.yaw += 0.1 * self.speed * tan(0.52 * steer) / 2.6
        self.speed = float(np.clip(self.speed + 0.3 * accel, 0, 9))
        delta = abs(float(steer) - self.prev[0])
        self.prev = np.array([steer, accel])
        self.steps += 1
        obs = self.observe(*self.cameras.pair(self.x,self.y,self.yaw))
        progress = .1*self.measured_speed if self.readable else 0.
        self.progress += progress
        departure = abs(self.error)>2.6 and self.readable or self.invalid>=3
        hazard = self.near_hazard
        success = self.progress>=40 and not (departure or hazard)
        reward = (progress - .65*self.error**2 - .5*obs[2]**2
                  - .055*(self.measured_speed-5)**2 - .1*delta - .03*accel**2
                  -25*bool(hazard or departure) +15*success)
        info=dict(scene=str(self.scene),collision=bool(hazard),success=bool(success),
                  departure=bool(departure),error_m=float(self.error),steering_change=float(delta))
        return obs,float(reward),bool(hazard or departure or success),self.steps>=160,info


def pursuit(obs):
    # Body-frame lookahead lateral offset and approximately six-metre chord.
    steer = atan(2 * 2.6 * obs[4] * 8 / 36) / 0.52
    accel = (5 - 10 * obs[0]) / 2
    return np.clip([steer, accel], -1, 1)
