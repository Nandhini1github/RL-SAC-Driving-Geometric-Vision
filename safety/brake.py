def needs_brake(speed_mps, distance_m, valid_fraction, frame_age):
    if frame_age > 2 or valid_fraction < 0.08:
        return True
    stopping_distance = 3.0 + 0.6 * speed_mps + speed_mps**2 / 8.0
    return distance_m is not None and distance_m <= stopping_distance
