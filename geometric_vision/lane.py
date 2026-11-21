"""Ego-relative straight-lane fit using a calibrated, level camera on flat pavement."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Lane:
    right_at_origin: float
    slope: float
    markings: int

    def right(self, forward):
        return self.right_at_origin + self.slope * forward

    def tracking_features(self):
        # Gym uses positive-left coordinates; heading error is ego minus path yaw.
        error = self.right_at_origin / np.hypot(1, self.slope)
        return error, np.arctan(self.slope), [-self.right(d) for d in (3, 6, 10)]


class LaneCamera:
    def __init__(self, k, baseline, width_m=3.5):
        self.width_m = width_m
        self.forward = np.arange(5.2, 18, 0.12)
        lateral = np.arange(-5, 5.01, 0.04)
        z = self.forward[:, None] - 1.6
        # Remap the road plane into a metric grid, relative to the ego origin.
        self.u = (k[0, 2] + k[0, 0]*(lateral + baseline/2)/z).astype("float32")
        self.v = np.broadcast_to(k[1, 2] + k[1, 1]*1.5/z, self.u.shape).astype("float32")
        slopes, offsets = np.meshgrid(np.arange(-0.19, 0.191, .01), np.arange(.65, 2.96, .05))
        self.slopes = slopes.ravel()
        self.offsets = offsets.ravel()
        self.columns = []
        for side in (-1, 1):
            y = side*self.offsets[:, None] + self.slopes[:, None]*self.forward
            self.columns.append(np.clip(np.rint((y+5)/.04).astype(int), 0, len(lateral)-1))

    def estimate(self, gray):
        view = cv2.remap(gray, self.u, self.v, cv2.INTER_LINEAR).astype("float32")
        ridge = np.zeros_like(view)
        ridge[:,5:-5] = view[:,5:-5] - np.maximum(view[:,:-10],view[:,10:])
        ridge = np.clip(ridge,0,60)
        fits = []
        for side, columns in zip((-1,1),self.columns):
            samples = ridge[np.arange(len(self.forward)),columns]
            scores = samples.mean(axis=1)
            best = scores.argmax()
            if scores[best] >= 9 and (samples[best] > 12).sum() >= 20:
                fits.append((scores[best],side,self.offsets[best]*side,self.slopes[best]))
        if len(fits)==2:
            _,_,lo,ls = fits[0]
            _,_,ro,rs = fits[1]
            widths = ro-lo+(rs-ls)*np.array([3,12])
            if np.all((widths>2.65)&(widths<4.5)):
                return Lane((lo+ro)/2,(ls+rs)/2,2)
        if fits:
            _,side,offset,slope = max(fits)
            return Lane(offset-side*self.width_m/2,slope,1)
        return None
