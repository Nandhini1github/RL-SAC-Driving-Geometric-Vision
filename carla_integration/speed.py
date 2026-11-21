"""Ground speed from timestamped GNSS fixes; no actor velocity or absolute pose input."""
from collections import deque
from math import cos, hypot, radians


class GNSSSpeed:
    def __init__(self):
        self.fixes = deque(maxlen=3)

    def update(self, fix):
        self.fixes.append((fix.timestamp, fix.latitude, fix.longitude))
        if len(self.fixes)<3:
            return None
        t,lat,lon = self.fixes[0]
        dt = fix.timestamp-t
        if not 0 < dt <= 0.35:
            return None
        east = radians(fix.longitude-lon)*cos(radians((lat+fix.latitude)/2))
        north = radians(fix.latitude-lat)
        return 6378137*hypot(east,north)/dt
