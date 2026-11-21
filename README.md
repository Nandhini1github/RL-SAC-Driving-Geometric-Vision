# Stereo perception and trajectory tracking in CARLA 1.0

This Windows project investigates whether a bounded training supervisor improves a SAC controller's ability to follow a reference path around obstacles. The first executable milestone measures stereo depth on known synthetic planes. SAC, the supervisor and live CARLA sensor capture are planned experiments.

## Run the first experiment

From this directory in PowerShell, using Python 3.12 and uv:

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
.\.venv\Scripts\python.exe -m evaluation.stereo_benchmark
```

The run writes [per-distance metrics](results/stereo_synthetic.json). It uses ten scored image pairs per distance plus one warm-up, 480 × 240 pixels, independent random textures, a fractional horizontal shift and noise in the right image. OpenCV runs on one CPU thread. Runtime covers matching and depth conversion. Coverage counts finite estimates across the whole left frame, including the search border. MAE and RMSE cover finite estimates. These inputs assume ideal rectification and contain no foreground occlusion or moving actors.

With rectified identical cameras, disparity is left minus right horizontal pixel position. Optical depth is Z = fB/d for focal length f in pixels and baseline B in metres. OpenCV returns fixed-point disparity with scale 16. Invalid estimates remain NaN. The left optical frame uses X = (u − cx)Z/fx to the right and Y = (v − cy)Z/fy down. Approximate depth uncertainty grows as Z² σd/(fB), which motivates reporting error by range.

## Windows environment

The inspected CARLA 1.0 Win64 archive includes the cp312 win_amd64 wheel. Install that wheel from the same shipping package as the server into the project environment for live experiments. The local client returned version 1.0. The server probe on localhost port 2000 timed out. Hardware reported an RTX 5060 Laptop GPU, 8151 MiB, driver 610.60.

## OpenDriveVLA feasibility

The 0.5B model is gated on Hugging Face. The listing shows a 1.47 GB weights file. End-to-end peak VRAM on this GPU remains unknown. The official one-GPU launcher uses torchrun, NCCL, BF16 and DeepSpeed. Its loader combines token IDs with UniAD data or feature files and instance indices. The data guide expects nuScenes trainval, CAN bus, maps, temporal infos and a cache. Predictions are JSON lines with decoded answers. The evaluator extracts six numeric coordinate pairs and can pad or truncate answers. Before control, check original point count, finite values, units, frame, timestep and freshness. [Checkpoint](https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B/tree/main) · [Inference code](https://github.com/DriveVLA/OpenDriveVLA/blob/main/drivevla/inference_drivevla.py) · [Data preparation](https://github.com/DriveVLA/OpenDriveVLA/blob/main/docs/2_DATA_PREP.md) · [Trajectory parser](https://github.com/DriveVLA/OpenDriveVLA/blob/main/drivevla/utils/trajectory_utils.py)

The installation guide uses Python 3.10, torch 2.1.2, a CUDA toolkit and GCC ≥5, patched MMCV 1.7.2 with ops and patched MMDetection3D 1.0.0rc6. It pins mmdet 2.26.0, mmsegmentation 0.29.1, mmengine 0.9.0, motmetrics 1.4.0, casadi 3.6.0, scipy 1.10.1 and scikit-image 0.19.3. The manifest additionally specifies torchvision 0.16.2, DeepSpeed 0.14.2 and Transformers commit 1c39974a4c4036fd641bc1191cc32799f85715a4 among its extras. CUDA_HOME is required to compile operators. The sample toolkit path is only an example. [Install guide](https://github.com/DriveVLA/OpenDriveVLA/blob/main/docs/1_INSTALL.md) · [Full dependency manifest](https://github.com/DriveVLA/OpenDriveVLA/blob/main/pyproject.toml)

PyTorch announced Blackwell support with 2.7 and CUDA 12.8. Moving the legacy stack into a Linux container would address shell and NCCL assumptions while leaving custom kernel compatibility to verify. Treat local VLA execution as an experiment. With authorized weights, prepared inputs and compatible Linux compute, run the published single-GPU command first, retain raw predictions and measure peak allocated/reserved memory. Until then label the reference source deterministic. [PyTorch release](https://pytorch.org/blog/pytorch-2-7/) · [Official evaluation example](https://github.com/DriveVLA/OpenDriveVLA/blob/main/docs/3_EVAL.md)

## Seven-day experiment

| Day | Checkpoint | Fallback or acceptance criterion |
|---|---|---|
| 1 | Verify hardware, client, checkpoint access and stereo benchmark. | Keep the VLA frozen. Proceed with deterministic reference paths while access or compatibility is unresolved. |
| 2 | Capture synchronized CARLA RGB stereo and aligned evaluation depth. Derive calibration from FOV and measured sensor transforms, rectify, check vertical residuals and estimate obstacle range from RGB disparity. | Replay recorded pairs offline. Report valid coverage, depth MAE/RMSE by range, obstacle range error and latency. Verify reference depth convention against a known plane. |
| 3 | Build a seeded Gymnasium bicycle model and a pure-pursuit plus speed-PID baseline. Validate reset, action limits and termination on straight, bend and obstacle routes. | Use deterministic waypoints with source, timestamp and ego-frame metres. Label any analytic obstacle observation as a simulator baseline. |
| 4 | Train state-vector SAC. Save configuration, seed, checkpoint, replay buffer and learning curves. | Evaluate the baseline and deterministic SAC actions on identical routes. Check inference latency, lateral RMSE, collision rate, completion and action changes. |
| 5 | Fork fixed and adaptive training from the same checkpoint and buffer. At fixed intervals select among predefined scenario mixtures using validation failures. | Log evidence, bounded proposal, applied setting and outcome. A rule-based selector is the fallback for LLM proposals. Keep reward and safety settings fixed. |
| 6 | Feed stereo range, bearing and validity plus path errors and speed to the controller. Apply deterministic braking for stale perception or insufficient stopping distance. Log requested and applied commands. | Record an offline perception replay and separate Gym closed-loop rollouts if live integration fails. Keep nuScenes VLA replay labelled open-loop. |
| 7 | Evaluate on held-out route seeds, export metrics and curves, record video and write the report. | Match environment steps, gradient updates, evaluation cases and shield settings across training arms. Report per-seed outcomes and intervention counts. |

Core deliverables are measured stereo, a trained controller, a classical baseline, a bounded supervisor and reproducible evaluation. Live VLA-to-CARLA inference, MPC, ROS and VLA fine-tuning are outside the core week. The architecture uses parallel reference and perception inputs to SAC. The supervisor reads validation metrics between training stages and never writes control commands.

For the bicycle model, use ẋ = v cos ψ, ẏ = v sin ψ, ψ̇ = v tan δ/L and v̇ = a. Scale normalized actions into steering angle δ and signed acceleration a. Use route arc-length progress as the positive dense reward, subtract weighted squared lateral/heading errors, acceleration magnitude and steering changes, and add terminal collision, departure and success terms. Fix coefficients before comparing curricula. SAC optimizes expected discounted reward plus policy entropy, using replayed transitions and the smaller of two target critic estimates. Tune on validation seeds and reserve final test routes for reporting.
