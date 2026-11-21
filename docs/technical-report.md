# Experimental methods and limitations

## From a stereo pair to a control observation

The live rig has two parallel 960 × 540 RGB cameras with 80° horizontal field of view, 0.32 m baseline, and matched timestamps. For the simulator pinhole model, fx = fy = W / [2 tan(FOV/2)], or about 572 px. The left camera sits 1.6 m forward, 0.16 m left and 1.5 m above the ego origin. CARLA supplies the mounting transforms and optics. OpenCV computes distortion/rectification maps from these calibrated parameters. A separate board-corner experiment tests recovery of an unknown relative pose with known intrinsics.

StereoSGBM searches a 64-disparity range using a 5 px block and a left-right consistency check. Its output is divided by 16 to obtain pixel disparity d. For d > 0, optical depth Z = fx B / d. Invalid pixels remain missing. Inverse projection gives X = (u − cx)Z/fx and Y = (v − cy)Z/fy. For this parallel left camera the ego forward/right/up coordinates are (Z + 1.6, X − 0.16, 1.5 − Y). The first-order sensitivity |∂Z/∂d| = Z²/(fx B) explains why far-range estimates are particularly sensitive to correspondence errors.

Obstacle localization filters points within a two-metre corridor, above the expected road surface. It scans range slices from near to far and requires an 18-pixel connected component. The median component coordinates estimate a visible surface. The two published pairs support a direct reproduction. At step 40 the recovered ego point was 8.17 m forward, 0.33 m left and 0.67 m high. This is a corridor heuristic for this camera mount. Hills, roadside clutter and a changing camera pitch are important failure cases.

For evaluation the aligned left depth sensor is decoded from RGB bytes as 1000(R + 256G + 65536B)/(2²⁴ − 1). Dense error statistics pool pixels in each reference-depth band. Coverage is the number of finite predictions divided by the number of eligible reference pixels. Obstacle range uses a separate reference, the nearest forward coordinate of the parked vehicle’s bounding-box vertices in the left optical frame. Box extent and visible bodywork differ. The reported range error includes that discrepancy. Only the RGB estimator supplies obstacle observations to control. The original map-assisted run used simulator ego state and map waypoints. The sensor-driven repeat uses camera lane geometry and GNSS speed.

## Geometry checks at three levels

1. **Calibration.** Fit 36 synthetic stereo chessboard poses with 0.1 px corner noise and fixed camera intrinsics. Evaluate vertical alignment on 12 unseen poses. The baseline fit was 0.31985 m for a 0.32002 m rig. Rotation error was 0.0063°, and the held-out vertical residual was 0.096 px median and 0.278 px at the 95th percentile. These are projected corner measurements.
2. **Known planes.** At 4, 8 and 16 m the original 480 × 240 synthetic plane test gave MAEs of 0.056, 0.145 and 0.772 m. Ten images per range and a single warm-up per range were used. About 86.7% of pixels had valid disparity. This rig used fx = 360 px and B = 0.24 m.
3. **Urban scene.** The live experiment used an independently rendered stereo pair. Pixel coverage averaged 58.4% across the whole left frame. The 3–8 m band had low mean absolute error alongside high RMSE, indicating large errors in some valid estimates. At 16–30 m the MAE reached 5.98 m. A coherent corridor estimate can be usable even when the dense map contains substantial errors. Report both metrics to make the distinction visible.

## Tracking task and SAC objective

The Gymnasium simulator uses a forward-Euler kinematic bicycle with dt = 0.1 s and wheelbase L = 2.6 m. Position rates are v cos ψ and v sin ψ, yaw rate is v tan δ / L and speed rate is a. Normalized actions map to steering δ = 0.52uₛ radians and acceleration a = 3uₐ m/s², with speed clipped to [0,9] m/s. Each 40 m route is straight, sinusoidal, or a Gaussian lateral bypass of a stationary obstacle. Initial speed, lateral position, heading, bypass side and obstacle position vary with seed.

The eleven normalized observations encode speed, signed tracking error, sine of heading error, three body-frame lookaheads, obstacle range, sine of bearing, visibility, and previous steering and longitudinal commands. Gym uses analytic obstacle geometry. CARLA fills these obstacle fields from stereo and substitutes native vehicle dynamics. CARLA’s normalized throttle/brake inputs have different dynamics from the Gym acceleration command. This contributes to the transfer challenge.

With e measured along the approximate path normal, the per-step reward is

```text
r = Δx − 0.65 e² − 0.5 sin²(heading error) − 0.055 (v − 5)²
    − 0.1 |Δuₛ| − 0.03 uₐ² − 25 I[collision or departure] + 15 I[finish]
```

Progress uses longitudinal x displacement on these forward routes. A collision occurs within 1.35 m of the obstacle center. A departure occurs beyond 2.6 m tracking error. Episodes time out after 160 steps. Steering smoothness is mean absolute change in normalized steering command per step.

SAC learns a squashed Gaussian actor and twin critics from replay. Its soft target is r + γ(1 − terminal)[min(Q′₁,Q′₂) − α log π(a′|s′)]. The actor minimizes E[α log π(a|s) − min(Q₁,Q₂)]. We used Stable-Baselines3 SAC, CPU, two 64-unit hidden layers, batch size 128, 32,768 replay capacity, default discount 0.99, and learned entropy temperature initialized at 0.1. Every four steps trigger two gradient updates after warm-up. The baseline uses geometric pure pursuit with 6 m lookahead and proportional speed control. [SAC implementation](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/sac.html)

## Controlled curriculum comparison

For each training seed (37, 83, 109), train a shared policy for 4,096 transitions and save weights, optimizers and replay. Each arm reloads the common fork, resets the training RNG identically and runs two further stages of 4,096 transitions. Fixed training samples scenarios uniformly. Adaptive training scores each validation stratum as the mean of 2 × collision + (1 − success) + episode RMSE / 3. The weakest stratum receives probability 0.6 and the other two each receive 0.2. The scenarios, reward, model, training budget and held-out cases are identical across arms. The test routes use seeds 8000–8019 per scene. Validation uses 700–703 per scene.

Completion and collision are per-episode fractions. Tracking is the mean of episode lateral RMSE. Latency is the average of per-episode median action-selection time with deterministic inference. One rule-guided training seed eliminated ten collisions despite worsening its tracking RMSE. Success and local tracking accuracy capture different aspects of driving. The same 60 cases recur for each training seed. The cross-seed sample size for training variation is three.

In the separate LLM arm, local Qwen2.5-7B reads stratified validation metrics and selects one scenario from a schema. Only the bounded mixture changes. The runner logs raw reasons and metrics and schedules the next fixed-length stage. Its second decision illustrates that schema validation constrains actions but cannot guarantee numerical reasoning. No part of the LLM drives the vehicle.

## Transfer and the deterministic brake

A fixed SAC checkpoint (seed 37) drives a single straight Town10HD approach. In the original 72-frame run, the ego route came from the map. Obstacle range and bearing come from RGB stereo. Above a stopping-distance threshold, SAC controls steering and longitudinal actuation. A range at or below 3 + 0.6v + v²/8 metres triggers full braking. Low frame coverage or stale sensor readings also trigger braking. For this stationary-target task, braking latches near standstill if the range is below 5 m. Steering remains the policy command.

The initial 130-frame pilot revealed oscillation around the release threshold and a dark camera exposure. Enabling RGB postprocessing and adding the terminal brake hold produced the reported 72-frame run. Both settings changed together, so that pilot cannot isolate their effects. The final stop was shield-assisted and the policy continued requesting positive acceleration. Dense stereo, rectification and obstacle extraction took a median 186 ms on CPU. At fixed 100 ms simulation steps this used synchronous simulation slower than real time. One tuned route provides integration evidence only. Held-out CARLA maps, weather and actor motion would be needed for broader transfer claims.

## Sensor-driven straight-road repeat

The same fixed checkpoint was replayed with a new observation front end. The left rectified RGB image is projected to a flat road grid using known camera intrinsics and mounting geometry. A bright-stripe detector scores straight candidate lane boundaries. When both boundaries agree on a plausible width, their midpoint gives the centerline. When only one reliable boundary is available, the fit assumes a 3.5 m lane width. The fitted line gives relative heading, cross-track error and body-frame lookahead offsets. Its 3 m lookahead extrapolates the fitted visible line to a near point. No global route is reconstructed.

Ground speed comes from the horizontal displacement between time-stamped GNSS fixes over two simulation ticks (0.2 s). The experiment added 0.00000006 degree standard deviation to each latitude and longitude channel with seed 37. Differencing introduces noise at standstill and lag during acceleration or braking. The controller applies full brake when lane or speed estimation is unavailable. It uses stereo-derived obstacle range and GNSS speed for the stopping-distance check. Neither GNSS absolute position nor actor pose is passed into the SAC policy. Map queries place the vehicles before driving and supply evaluation labels after each command.

Both lane boundaries supported the fit on 62 of 77 frames; the single-boundary estimate supplied the remaining 15. The vehicle stopped without a recorded collision. Map-scored lateral tracking RMSE increased from 0.162 m in the map-assisted run to 0.210 m here. The lane detector assumes a level camera and visible straight lane markings. A single straight approach does not establish performance at intersections, on hills or in adverse weather. Gymnasium training still used analytic features. [Telemetry](../results/carla_sensor/frames.csv) / [Summary](../results/carla_sensor/metrics.json)

## Model provenance

The executable reference paths in the Gym and CARLA runs are deterministic. OpenDriveVLA's public release uses gated weights and a UniAD/nuScenes input contract. Official inference and peak GPU memory remain unverified here. The local Qwen model is a training analyst. [OpenDriveVLA dependency and interface audit](opendrivevla.md)
