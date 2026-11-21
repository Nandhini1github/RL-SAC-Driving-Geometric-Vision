"""Regression check: identical camera measurements must produce identical learning signals."""
from unittest.mock import patch

import numpy as np
from driving_environment.route import RouteEnv
from driving_environment.camera_features import CameraFeatures


def check_hidden_state(a, b):
    assert np.array_equal(a.reset(seed=71)[0], b.reset(seed=71)[0])
    for i in range(12):
        act=np.float32([.05,.6])
        recorded=[]
        renderer=a.unwrapped.cameras.pair
        def capture(x,y,yaw):
            pair=renderer(x,y,yaw)
            recorded.extend(pair)
            return pair
        with patch.object(a.unwrapped.cameras,"pair",side_effect=capture):
            first=a.step(act)
        # If the reward, observation or a done flag used state directly this would change it.
        hidden = b.unwrapped
        hidden.x,hidden.y,hidden.yaw,hidden.speed,hidden.ox=250.,60.,1.9,9.,250.
        with patch.object(hidden.cameras,"pair",return_value=recorded):
            second=b.step(act)
        np.testing.assert_allclose(first[0],second[0])
        assert first[1:]==second[1:], f"Learning signals depend on hidden state at tick {i}"
    a.close()
    b.close()


def main():
    check_hidden_state(RouteEnv(), RouteEnv())
    print("Equal images and commands yielded equal learning signals under different hidden states.")

    env = RouteEnv()
    env.reset(seed=71, options={"scene": "obstacle"})
    left, right = env.cameras.pair(env.x, env.y, env.yaw)
    assert CameraFeatures().measure(left, right, 0)[2] is not None
    assert CameraFeatures().measure(left, left, 0)[2] is None

    blank = np.zeros_like(left)
    with patch.object(env.cameras, "pair", return_value=(blank, blank)):
        for tick in range(3):
            _, _, done, _, info = env.step(np.zeros(2))
            assert done == (tick == 2)
    assert info["departure"] and not info["success"] and env.progress == 0
    print("Passed: eliminating parallax removed range detection, and blank images terminated training after three frames.")


if __name__=="__main__":
    main()
