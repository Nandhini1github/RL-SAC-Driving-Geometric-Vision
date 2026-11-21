# RL-SAC-Driving-Geometric-Vision

**Can a stereo range estimate transfer into a learned driving controller, and does adaptive training improve that controller?** This project tests both questions with a Windows CARLA 1.0 stereo rig, a Gymnasium bicycle model, and Soft Actor-Critic (SAC). Three training seeds compare fixed and adaptive curricula. A separate experiment gives a local language model bounded authority over scenario sampling.

[![CARLA stereo braking replay](results/media/carla-poster.jpg)](https://github.com/Nandhini1github/RL-SAC-Driving-Geometric-Vision/raw/refs/heads/main/results/media/carla-stereo-stop.mp4)

**[Watch the 1080p CARLA stereo and braking video](https://github.com/Nandhini1github/RL-SAC-Driving-Geometric-Vision/raw/refs/heads/main/results/media/carla-stereo-stop.mp4)** · **[Watch the 900p controller comparison](https://github.com/Nandhini1github/RL-SAC-Driving-Geometric-Vision/raw/refs/heads/main/results/media/controllers.mp4)** · [Technical report](docs/technical-report.md)

The CARLA film displays both 960 × 540 sensor panels at native resolution. Its half-speed playback shows the speed response and exact brake-override frames. The second film replays three controllers on the same held-out Gymnasium route. All reference trajectories in these experiments are deterministic.

## Geometric vision on a live CARLA approach

A Tesla Model 3 approached a stationary vehicle on Town10HD. The controller received speed, map-relative tracking features and a corridor obstacle estimate from the synchronized RGB stereo pair. The aligned depth camera and parked vehicle bounding box supplied evaluation references. The final run reached **9.48 m/s**, recorded **zero collisions** and stopped **3.16 m** from the parked box, measured forward from the left camera. The deterministic brake overrode SAC on **33 of 72 frames**. On the final frame SAC still requested **+0.91** normalized longitudinal command.

| Reference depth | Valid-pixel coverage | Depth MAE | Depth RMSE |
|---|---:|---:|---:|
| 3–8 m | 38.4% | 0.52 m | 7.84 m |
| 8–16 m | 56.8% | 1.66 m | 8.04 m |
| 16–30 m | 60.6% | 5.98 m | 21.91 m |

![CARLA measured range, override timing and depth error](results/media/carla.png)

The stereo estimator returned a corridor surface in all 72 frames. Its forward range had **0.44 m MAE** and **0.63 m RMSE** against the parked box front. The dense depth map was less reliable, particularly at long range. Small disparity errors amplify with depth, and the gap between MAE and RMSE shows that some accepted correspondences were badly wrong. Median geometry processing latency was **186 ms** at 960 × 540. The run used synchronous stepping at 0.1 simulation seconds per frame. This single approach establishes a working integration with a brake override. [Frame-level measurements](results/carla/frames.csv) · [Metrics](results/carla/metrics.json) · [Replayable image pairs](results/carla)

Two independent checks isolate calibration and matching. Fitting a nonparallel synthetic rig from noisy board corners yielded **0.278 px** held-out 95th-percentile vertical rectification error. On ideal textured planes the depth MAE grew from **0.056 m at 4 m** to **0.772 m at 16 m**. [Calibration results](results/calibration.json) · [Synthetic depth plot](results/media/stereo.png)

## Reinforcement learning on held-out routes

### Why pure pursuit?

Each controller receives the same reference path, including a predefined bypass on obstacle routes. Pure pursuit gives the experiment a baseline that needs no training. It steers toward a waypoint 6 m ahead, while a separate proportional controller adjusts speed toward 5 m/s. Comparing SAC with this baseline tests what learning adds to the task of following a known path. Choosing a path around an obstacle is a separate planning problem.

Each SAC policy received **12,288 training transitions**. Within each seed, the two arms started from identical actor, critic, optimizer and replay states after 4,096 shared transitions. Each then trained for two equal stages. A uniform curriculum and a validation-driven curriculum faced the same 20 straight, 20 bend and 20 obstacle routes. Gymnasium obstacle observations use analytic scene geometry. No brake shield was applied during this comparison.

| Controller | Completed | Collision rate | Tracking RMSE | Mean return | Mean steering change |
|---|---:|---:|---:|---:|---:|
| Pure pursuit + proportional speed control | 60/60 | 0% | 0.212 m | 51.27 | 0.0121 |
| Fixed-distribution SAC | 170/180 | 5.6% | 0.270 m | 28.35 | 0.0398 |
| Rule-guided adaptive SAC | 180/180 | 0% | 0.263 m | 30.77 | 0.0381 |

![Tracking, completion and return by training seed](results/media/training-seeds.png)

**The ten fixed-policy collisions all occurred in obstacle scenarios for training seed 109.** Adaptive training emphasized obstacles in both stages for that seed and completed all 60 routes. The other two seeds already had perfect completion with fixed training. Pure pursuit completed all 60 routes with lower average tracking RMSE (0.212 m versus 0.263 m for adaptive SAC), smoother steering and higher return. Adaptive practice helped SAC in its failure-prone seed, while pure pursuit remained the more accurate tracker overall on these short, mapped routes. Policy latency averaged 0.31 ms for fixed SAC, 0.37 ms for adaptive SAC and 0.005 ms for pure pursuit on this machine. [Per-seed results](results/per_seed.csv) · [Aggregate results](results/aggregate.csv)

![Training returns, validation completion and held-out tracking for seed 37](results/media/control.png)

![Three controllers on the same held-out route](results/media/controllers.gif)

## What the training agent learned from the logs

The rule-guided supervisor audits validation failures after each stage. Its only decision is which of three scenarios receives 60% of the next stage, with the other two receiving 20% each. The **separate local Qwen2.5-7B arm** selected bends twice. It completed **60/60** seed-37 test routes with **0.215 m** tracking RMSE and **24.69** mean return. On that same seed fixed SAC achieved **0.205 m** and **27.30**.

The LLM correctly identified bends as the highest-error scenario after the common checkpoint. At the second decision it again called bends the highest-error case (0.246 m), although obstacles had reached 0.324 m. The saved decision record exposes the error and the exact training change it caused. This pilot provides no performance advantage for the language-model selector. [LLM decisions and inputs](results/control/llm_decisions.json) · [LLM held-out episodes](results/control/llm_heldout.csv)

## System architecture

### Where reinforcement learning happens

**Training runs in Gymnasium.** At each step, the SAC actor chooses continuous steering and acceleration commands. The bicycle environment advances the vehicle, scores progress, tracking, smoothness and terminal outcomes, and returns the next observation. Stable-Baselines3 stores these transitions in a replay buffer and samples them to update the actor and twin critics. The fixed arm samples all three scenarios uniformly. In the adaptive arms, a rule or local LLM uses validation results to choose the next stage's scenario mix. [Environment](driving_environment/route.py) / [SAC training loop](sac_training/experiment.py)

```mermaid
flowchart TD
    Actor[SAC actor] -->|Steering and acceleration| Env[Gymnasium bicycle environment]
    Env -->|Next observation| Actor
    Env -->|Transition including reward| Replay[Replay buffer]
    Replay -->|Sampled batches| Learn[SAC actor and twin-critic updates]
    Learn -->|Updated weights| Actor
    Learn --> Save[Policy checkpoint]
    Save --> Val[Evaluate on validation routes]
    Val --> Sup[Adaptive arms only: rule or local LLM]
    Sup -->|Scenario mix for next training stage| Env
```

### How CARLA uses the trained policy

**The live drive uses inference only.** CARLA loads the fixed-distribution checkpoint from training seed 37. At each tick the actor predicts steering and a signed longitudinal command. The brake supervisor can replace that longitudinal command with full braking while retaining the actor's steering. The vehicle's motion produces the next camera pair and ego state. [CARLA control loop](carla_integration/drive.py)

```mermaid
flowchart TD
    RGB[Synchronized stereo RGB] --> Depth[Rectification and SGBM depth]
    Depth --> Range[3D corridor obstacle estimate]
    Range --> Obs[Control observation]
    State[Ego state and map waypoints] --> Obs
    Obs --> Actor[Trained SAC actor]
    Actor -->|Requested controls| Brake[Deterministic brake supervisor]
    Range -->|Obstacle range| Brake
    Checks[Speed, stereo coverage and frame age] --> Brake
    Brake -->|Applied steering, throttle and brake| Ego[CARLA ego vehicle]
    Depth -.-> Eval[Geometry evaluation]
    Range -.-> Eval
    GT[Reference depth camera and actor box] -.-> Eval
```

The training environment generates analytic waypoints and obstacle features. The CARLA run uses map waypoints and stereo-derived obstacle features. The reference depth camera scores dense depth, and the actor box supplies the reference obstacle range.

## Reproduce on Windows

Tested with Python 3.12.12, Windows, CPU PyTorch 2.8.0 and an RTX 5060 Laptop GPU (8 GB) for CARLA rendering. From this repository in PowerShell:

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements-control.txt
$python = ".\.venv\Scripts\python.exe"
& $python -m evaluation.calibration
& $python -m evaluation.stereo_benchmark
& $python -m evaluation.replay_stereo
```

The last command runs obstacle localization on the **published stereo frames** and then scores depth against the saved references. It works without a running simulator.

```powershell
foreach ($trial in 37,83,109) { & $python -m sac_training.experiment --seed $trial }
& $python -m evaluation.figures
& $python -m evaluation.across_seeds
& $python -m evaluation.rl_video
```

Each training directory contains episode logs, validations, final tests, decisions and checkpoints. Training recreates the local common replay buffer. For the optional local LLM experiment, run a local Ollama server with `qwen2.5:7b` installed, then run `& $python -m agentic_supervisor.local_llm` after training seed 37. The decision log identifies the source if the model is unavailable and the deterministic selector takes over.

For live driving, set `$carlaDir` to an extracted **CARLA 1.0 Win64 shipping** directory. Use its matching cp312 Windows client wheel and a dedicated server. The capture temporarily enables synchronous stepping and restores world settings on exit.

```powershell
$wheel = Get-ChildItem "$carlaDir/PythonAPI/carla/dist/carla-1.0-cp312-cp312-win_amd64.whl"
uv pip install --python .venv/Scripts/python.exe $wheel.FullName
& "$carlaDir/CarlaUnreal.exe" -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2540
```

When Town10HD_Opt is ready, run these from the project environment in a second PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m carla_integration.drive --port 2540
.\.venv\Scripts\python.exe -m evaluation.carla_media
```

The live script uses the seed-37 fixed SAC checkpoint. GPU rendering, weather, ambient objects and CPU load can change pixel metrics and wall-clock timings. The included Dockerfile describes a portable CPU experiment environment; the native Windows workflow above was exercised. The [technical report](docs/technical-report.md) covers the geometry, SAC objective, reward, vehicle dynamics, control transfer and limitations.
