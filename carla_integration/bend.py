"""Scene setup and independent scoring for two marked Town10HD bends."""
import numpy as np


class BendReference:
    def __init__(self, road, turn):
        if not road.name.endswith("Town10HD_Opt"):
            raise ValueError("The bend starts are defined on Town10HD_Opt.")
        lane, start_s = (-2, 38.) if turn == "left" else (2, 116.)
        start = road.get_waypoint_xodr(1, lane, start_s).transform
        self.start = (start.location.x, start.location.y, start.rotation.yaw)
        self.lane_id = lane
        stations = np.arange(.5, 119., .25)
        if lane > 0:
            stations = stations[::-1]
        points = [road.get_waypoint_xodr(1, lane, float(s)).transform.location for s in stations]
        self.xy = np.array([(p.x, p.y) for p in points])
        self.segments = np.diff(self.xy, axis=0)
        self.squared_lengths = np.sum(self.segments**2, axis=1)
        self.arc = np.r_[0, np.cumsum(np.sqrt(self.squared_lengths))]

    def score(self, x, y):
        delta = np.array([x, y]) - self.xy[:-1]
        fraction = np.clip(np.sum(self.segments*delta, axis=1)/self.squared_lengths, 0, 1)
        residual = delta - fraction[:, None]*self.segments
        i = np.argmin(np.sum(residual**2, axis=1))
        normal_error = (self.segments[i, 1]*residual[i, 0] - self.segments[i, 0]*residual[i, 1])
        return dict(tracking_m=normal_error/np.sqrt(self.squared_lengths[i]),
                    station_m=self.arc[i]+fraction[i]*np.sqrt(self.squared_lengths[i]),
                    reference_x_m=x-residual[i, 0], reference_y_m=y-residual[i, 1],
                    within_reference=bool(0 < i < len(residual)-1))
