# Experimental methods and limitations

## From a stereo pair to a control observation

The live rig has two parallel 960 × 540 RGB cameras with 80° horizontal field of view, 0.32 m baseline, and matched timestamps. For the simulator pinhole model, fx = fy = W / [2 tan(FOV/2)], or about 572 px. The left camera sits 1.6 m forward, 0.16 m left and 1.5 m above the ego origin. CARLA supplies the mounting transforms and optics. OpenCV computes distortion/rectification maps from these calibrated parameters. A separate board-corner experiment tests recovery of an unknown relative pose with known intrinsics.

StereoSGBM searches a 64-disparity range using a 5 px block and a left-right consistency check. Its output is divided by 16 to obtain pixel disparity d. For d > 0, optical depth Z = fx B / d. Invalid pixels remain missing. Inverse projection gives X = (u − cx)Z/fx and Y = (v − cy)Z/fy. For this parallel left camera the ego forward/right/up coordinates are (Z + 1.6, X − 0.16, 1.5 − Y). The first-order sensitivity |∂Z/∂d| = Z²/(fx B) explains why far-range estimates are particularly sensitive to correspondence errors.

Obstacle localization filters points within a two-metre corridor, above the expected road surface. It scans range slices from near to far and requires an 18-pixel connected component. The median component coordinates estimate a visible surface. The two published pairs support a direct reproduction. The [sample coordinates](../results/carla_sensor/sample_points.json) give the estimated obstacle surface in both optical and ego frames. This is a corridor heuristic for this camera mount. Hills, roadside clutter and a changing camera pitch are important failure cases.

For evaluation the aligned left depth sensor is decoded from RGB bytes as 1000(R + 256G + 65536B)/(2²⁴ − 1). Dense error statistics pool pixels in each reference-depth band. Coverage is the number of finite predictions divided by the number of eligible reference pixels. Obstacle range uses a separate reference, the nearest forward coordinate of the parked vehicle’s bounding-box vertices in the left optical frame. Box extent and visible bodywork differ. The reported range error includes that discrepancy. Only the RGB estimator supplies obstacle observations to control. The controller uses camera-derived lane geometry and ground-plane visual odometry.

## Geometry checks at three levels

1. **Calibration.** Fit 36 synthetic stereo chessboard poses with 0.1 px corner noise and fixed camera intrinsics. Evaluate vertical alignment on 12 unseen poses. The baseline fit was 0.31985 m for a 0.32002 m rig. Rotation error was 0.0063°, and the held-out vertical residual was 0.096 px median and 0.278 px at the 95th percentile. These are projected corner measurements.
2. **Known planes.** At 4, 8 and 16 m the original 480 × 240 synthetic plane test gave MAEs of 0.056, 0.145 and 0.772 m. Ten images per range and a single warm-up per range were used. About 86.7% of pixels had valid disparity. This rig used fx = 360 px and B = 0.24 m.
3. **Urban scene.** The live experiment uses independently rendered left and right views. Pixel coverage and error by reference-depth band are recorded in each run. Empty disparity is kept invalid. Sparse range estimates describe the frontmost coherent surface in the driving corridor. [Fixed policy](../results/carla_sensor/metrics.json) / [Adaptive policy](../results/carla_sensor_adaptive/metrics.json)

## Tracking task and SAC objective

The Gymnasium simulator uses a forward-Euler kinematic bicycle with dt = 0.1 s and wheelbase L = 2.6 m. Position rates are v cos ψ and v sin ψ, yaw rate is v tan δ / L and speed rate is a. Normalized actions map to steering δ = 0.52uₛ radians and acceleration a = 3uₐ m/s², with speed clipped to [0,9] m/s. Each 40 m route is straight, sinusoidal, or a Gaussian lateral bypass of a stationary obstacle. Each episode begins at rest. Lateral position, heading, obstacle position, texture and bypass side vary with seed.

### What reaches the learner

The Gymnasium simulator renders a calibrated, rectified grayscale pair (192 x 112, focal length 115 px, baseline 0.32 m) from a textured road and painted boundaries. A textured vertical obstacle panel has parallax in the two images. Only the renderer and vehicle dynamics use the simulated pose. The perception frontend accepts two images and a timestamp. It extracts paired lane stripes on the ground plane, fits the near tangent, interpolates the painted center at the lookaheads and obtains a foreground range and bearing from SGBM disparity. The obstacle bypass is visible in the painted lane. It is a tracking task.

The actor receives visual speed, signed visual lane error, sine of visual heading error, left offsets at 3/6/10 m, stereo range, sine of bearing, a detection flag, and its two previous actuator commands. It is an MLP operating on camera-derived features. Both pure pursuit and the SAC policies receive the same perception. The CARLA frontend uses native RGB, a straight-lane fitter and larger images. Both domains use the same metric visual-odometry implementation.

Speed comes from successive left images. Camera height and intrinsics warp pavement into a vehicle-relative metric grid. Three image patches are registered against the previous grid. Consistent displacements divided by elapsed camera time give forward speed. If that estimate fails, tracked pavement corners and a robust planar transformation provide a fallback. Feature loss can leave speed unknown. In training the missing channel is filled with zero speed, invalid progress is suppressed, and three consecutive invalid lane/speed estimates terminate the episode. At deployment, missing lane or speed triggers full braking. This estimator assumes a roughly flat road, calibrated camera mount and textured, static pavement.

### Reward, stopping conditions and scoring

```text
r = visual progress - 0.65 e^2 - 0.5 sin(heading error)^2 - 0.055 (v - 5)^2
    - 0.1 |change in steering| - 0.03 acceleration_command^2
    - 25 I[visual hazard or lane loss] + 15 I[visual finish]
```

Here e is the estimated lane offset and v is optical speed. Visual progress is 0.1 times estimated speed on readable frames. Forty integrated metres imply visual finish. A close elevated stereo surface (under 2.4 m optical depth and within 1 m laterally) gives a visual hazard signal. A measured offset beyond 2.6 m or sustained perception loss ends the episode. A 160-step time limit bounds every rollout. These rules also supply validation metrics for both supervisors. No oracle resets a training episode because the true vehicle crossed a boundary or hit the obstacle.

The held-out evaluator independently measures true lane tracking RMSE, geometric contact within 1.35 m of the obstacle center, departures beyond 2.6 m and arrival within 1 m of the 40 m target. Success requires arrival without any recorded contact or departure. Ground truth scores the trajectory after actions have been selected. It never feeds gradients, curriculum decisions or a control observation. The same camera-based episode boundaries still apply during testing, so premature visual termination can prevent true completion. Top-down evaluation videos use the simulator coordinates only to draw the recorded trajectories.

SAC learns a squashed Gaussian actor and twin critics from replay. Its soft target is r + γ(1 − terminal)[min(Q′₁,Q′₂) − α log π(a′|s′)]. The actor minimizes E[α log π(a|s) − min(Q₁,Q₂)]. We used Stable-Baselines3 SAC, CPU, two 64-unit hidden layers, batch size 128, 32,768 replay capacity, default discount 0.99, and learned entropy temperature initialized at 0.1. Every four steps trigger two gradient updates after warm-up. The baseline uses geometric pure pursuit with 6 m lookahead and proportional speed control. [SAC implementation](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/sac.html)

## Controlled curriculum comparison

For each training seed (37, 83, 109), train a shared policy for 4,096 transitions and save weights, optimizers and replay. Each arm reloads the common fork, resets the training RNG identically and runs two further stages of 4,096 transitions. Fixed training samples scenarios uniformly. Adaptive training scores each validation stratum as the mean of 2 × collision + (1 − success) + episode RMSE / 3. The weakest stratum receives probability 0.6 and the other two each receive 0.2. The scenarios, reward, model, training budget and held-out cases are identical across arms. The test routes use seeds 8000–8019 per scene. Validation uses 700–703 per scene.

Completion and collision are per-episode fractions. Tracking is the mean of episode lateral RMSE. Latency is the average of per-episode median action-selection time with deterministic inference. Final scores use the independent held-out evaluator. Validation CSV collision and success fields describe the visual proxies used for curriculum selection. The same 60 cases recur for each training seed. The cross-seed sample size for training variation is three.

In the separate LLM arm, local Qwen2.5-7B reads stratified validation metrics and selects one scenario from a schema. Only the bounded mixture changes. The runner logs raw reasons and metrics and schedules the next fixed-length stage. The decisions log records whether Qwen answered or a fallback handled the stage. No part of the LLM drives the vehicle.

## Transfer and the deterministic brake

Fixed and adaptive SAC checkpoints (seed 37) each drive the straight Town10HD approach. Obstacle range and bearing come from RGB stereo. Above a stopping-distance threshold, SAC controls steering and longitudinal actuation. A range at or below 3 + 0.6v + v²/8 metres triggers full braking. Low frame coverage or stale sensor readings also trigger braking. For this stationary-target task, braking latches near standstill if the range is below 5 m. Steering remains the policy command.

The same deterministic safety rule is used for both policies. Steering and requested longitudinal action, applied throttle and brake, sensor availability, optical speed, and the independent evaluation references are logged on every frame. The environment uses a fixed 100 ms simulation step and can run slower than real time.

The left rectified image supplies the CARLA lane center through a calibrated ground-plane projection and stripe fitting. If only one reliable boundary is visible, the fit assumes a nominal 3.5 m lane. When both are available it fits their midpoint. The approach is scoped to a straight, level street. A single route per controller provides integration evidence. Two fixed spawn poses define the initial scene. The evaluator reads depth, actor box, actor velocity and map centerline after each command. Collision events are logged for scoring. [Frame telemetry](../results/carla_sensor/frames.csv)

## Model provenance

The [residual SAC follow-up](residual-experiment.md) tests a different training approach on 90 fresh Gymnasium routes. It compares a trained correction against the same tuned controller with the correction disabled and against direct-action SAC. The CARLA results above belong to the original checkpoints.

The [standalone TD3 study](td3-experiment.md) asks whether a network trained from demonstrations and camera rewards can outperform its geometric teacher. The deployed TD3 actor outputs both commands itself. The study includes an imitation-only control and a separate replication with frozen weights.

The Gymnasium renderer paints a reference lane into each scene. Both the Gymnasium and CARLA policies track lane geometry fitted from visible markings. OpenDriveVLA's public release uses gated weights and a UniAD/nuScenes input contract. Official inference and peak GPU memory remain unverified here. The local Qwen model is a training analyst. [OpenDriveVLA dependency and interface audit](opendrivevla.md)
