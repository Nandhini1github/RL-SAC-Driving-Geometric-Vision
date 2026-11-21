# RL-TD3-Driving-Geometric-Vision

Can a car learn steering and acceleration from camera measurements, then use the same policy in a different simulator? A frozen TD3 ensemble completed **598 of 600 held-out Gymnasium routes** and drove a straight approach to a parked car in **CARLA 1.0**. In CARLA, stereo vision measured the obstacle ahead, TD3 drove the approach, and a deterministic brake brought the vehicle to rest with **3.25 m of camera-to-box clearance**.

## Watch TD3 drive in CARLA

[![Animated replay of TD3 driving toward a parked car, showing the camera-fitted centerline, stereo depth, speed and brake override](results/carla_td3/media/carla-preview.gif)](https://nandhini1github.github.io/RL-TD3-Driving-Geometric-Vision/#carla)

[Play the 1080p video with controls](https://nandhini1github.github.io/RL-TD3-Driving-Geometric-Vision/#carla) · [Download the recording](results/carla_td3/media/carla-stereo-stop.mp4) · [Frame-by-frame telemetry](results/carla_td3/frames.csv)

The cyan line follows the center of the lane fitted from visible paint in the left image. On the right, warm colors indicate nearby surfaces, cool colors indicate greater depth, and black pixels have no valid stereo match. The orange range trace uses the parked car’s bounding box for evaluation. The control indicator shows when the brake supervisor overrides TD3. Playback is half speed. The clock shows simulation time.

## Why TD3?

Steering and acceleration are continuous. TD3 learns an actor that can issue both commands directly from lane position, camera-estimated speed, obstacle geometry and its previous commands. The actor first imitates driving demonstrations. During offline training, two critics estimate how much future camera-derived reward an action will earn. TD3 uses the lower estimate in its training target and updates the actor less often than the critics. These choices limit the effect of overoptimistic action values. Keeping an imitation term during training anchors the actor to demonstrated behavior.

We tested whether reward-based learning added anything beyond imitation. On the same 600 routes, an ensemble of three imitation-only policies finished **586**. Three policies that also received TD3 updates finished **598**, with zero recorded contacts for both ensembles. Both used the median of the three steering predictions and the median of the three acceleration predictions. [Training and checkpoint selection](docs/td3-experiment.md) · [Paired experiment](docs/td3-committee.md)

## What the stereo cameras measured

The CARLA rig used two synchronized 960 × 540 RGB cameras spaced 0.32 m apart. Rectification aligned their image rows. OpenCV StereoSGBM matched features between the views, producing a horizontal disparity `d`. Optical depth was `Z = fB/d`, with focal length `f` in pixels and camera separation `B` in metres. The controller used the estimated range and bearing of a connected group of points above the road in the driving lane. Successive pavement images supplied speed. [Geometry, calibration and coordinate frames](docs/technical-report.md)

```mermaid
flowchart LR
    A[Stereo cameras] --> B[Lane, visual speed and obstacle range]
    B --> C[Frozen TD3 ensemble]
    C --> D[Steering and acceleration request]
    B --> E[Brake supervisor]
    D --> E
    E --> F[CARLA vehicle]
    F --> A
```

The same three Gymnasium checkpoints ran unchanged on Town10HD_Opt. The simulator supplied the initial scene and rendered sensor images. During driving the actor and brake received camera estimates. Reference depth, vehicle pose and velocity, map centerline, parked box and collision events were used for scoring. [Run metadata and checkpoint hashes](results/carla_td3/run.json) · [Control loop](carla_integration/drive.py)

| Single CARLA approach | Measured result |
|---|---:|
| Control frames / frames with brake override | 101 / 25 |
| Recorded collisions | 0 |
| Final distance from left camera to parked box | 3.25 m |
| Stereo obstacle-range mean absolute error | 0.34 m |
| Lane-offset estimation RMSE | 0.058 m |
| Vehicle tracking RMSE against the lane center | 0.048 m |
| Camera speed RMSE against simulator velocity | 0.73 m/s |
| Median vision time / median TD3 ensemble inference time on CPU | 92.49 ms / 0.42 ms |

![TD3 approach with measured stereo range versus box range, visual speed during brake override, and depth error by distance](results/carla_td3/media/carla.png)

The 0.34 m range error describes one obstacle surface. Dense depth was less accurate at longer range. At 3–8 m, 36.6% of reference pixels had valid predictions, with 0.59 m mean absolute error. At 16–30 m, coverage was 64.5% and mean absolute error was 5.95 m. This single drive demonstrates closed-loop transfer on a straight road. Generalization to traffic, curved roads and weather requires additional CARLA trials. [Depth results by range](results/carla_td3/metrics.json) · [Replayable stereo examples](results/carla_td3/sample_points.json)

## Gymnasium route completion

Each controller drove the same 200 straight, 200 bend and 200 obstacle-bypass cases. The evaluator counted a finish when the vehicle reached 39 m on a 40 m route without a recorded contact or lane departure. Pure pursuit supplied the demonstrations, using a 4.5 m lookahead and 4.0 m/s speed target.

| Controller | Clean finishes | Contacts | Mean episode lateral RMSE |
|---|---:|---:|---:|
| TD3 ensemble | **598/600 (99.7%)** | 0 | 0.218 m |
| Imitation-only ensemble | 586/600 (97.7%) | 0 | 0.174 m |
| Tuned pure pursuit | 584/600 (97.3%) | 0 | 0.174 m |

TD3 gained 14 finishes over its teacher. The paired completion gain was 2.33 percentage points, with an adjusted interval of +0.83 to +4.00. On bends the ensemble finished all 200 routes and the teacher finished 189. The teacher followed the centerline more closely on average. In the earlier 300-route replication, the same teacher completed 288/300 (96.0%). [Per-route results and uncertainty](docs/td3-committee.md) · [Earlier replication](docs/td3-experiment.md) · [Gymnasium replay](https://nandhini1github.github.io/RL-TD3-Driving-Geometric-Vision/#td3)

![Unfinished routes by scene for the TD3 ensemble and its teacher](results/td3/committee/head-to-head.png)

## Reproduce on Windows

Use Python 3.12 from the repository root. The saved weights, stereo sample pairs and evaluation results are included. These commands replay the geometry and rebuild the head-to-head plot without launching CARLA.

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements-control.txt
$py = ".\.venv\Scripts\python.exe"
& $py -m evaluation.verify_camera_inputs
& $py -m evaluation.replay_stereo
& $py -m evaluation.td3_committee head-to-head
```

To repeat the live run, set `$carlaDir` to the installed CARLA 1.0 Win64 shipping folder. Install its matching client wheel and start a dedicated offscreen server with Town10HD_Opt loaded.

```powershell
$wheel = Get-Item "$carlaDir/PythonAPI/carla/dist/carla-1.0-cp312-cp312-win_amd64.whl"
uv pip install --python .venv/Scripts/python.exe $wheel.FullName
Start-Process "$carlaDir/CarlaUnreal.exe" -ArgumentList '-RenderOffScreen','-nosound','-quality-level=Low','-carla-rpc-port=2540' -WindowStyle Hidden
```

Once the server is ready, run the capture and render the video. The capture restores the server’s settings when it exits. To keep the published recording intact, write a repeat to separate directories.

```powershell
& $py -m carla_integration.drive --port 2540 --output .runtime/repeat --capture .runtime/repeat-frames
& $py -m evaluation.carla_media --run-dir .runtime/repeat --capture-dir .runtime/repeat-frames
```

[Train TD3 from the demonstration dataset](docs/td3-experiment.md) · [Reproduce all six controller evaluations](docs/td3-committee.md) · [Technical report](docs/technical-report.md)
