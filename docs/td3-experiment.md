# Training the TD3 driving controller

The offline dataset contains 18,814 transitions from 192 Gymnasium episodes. Each transition includes camera measurements, an executed action, the next observation, a camera-derived reward, an episode-end flag and a demonstration action. The original demonstration setting uses a 4.5 m lookahead and a 4.0 m/s speed target. The published longitudinal weights come from this first phase of TD3 training.

The steering phase holds those longitudinal weights fixed. It transports past lane-center points observed 6 m ahead into the current vehicle frame using visual odometry and the previous steering command. Three extra features estimate lane position at 0, 3.5 and 4 m. On the chosen configuration the training steering label targets the 3.5 m point. The additional cost per step is the square of the estimated lateral offset in metres. The actor takes 1,600 imitation minibatch updates before 3,200 critic updates with a steering update every second critic update. The actor minimizes imitation error minus 0.12 times the normalized predicted value. Both the reward and the demonstration target come from camera-derived measurements.

Three settings varied the lane-error cost, use of past markings and lookahead. Camera-only validation used 96 routes (seeds 57000–57031 in three scenes). Among candidates with the fewest visual hazards and within one finish of the best, the setting with the lowest camera-derived lane RMSE was selected. Its three steering checkpoints were frozen before testing. TD3 takes the median steering and median acceleration across seeds 37, 83 and 109. [Validation and frozen steering hashes](../results/tracking/selection.json) · [Training source](../td3_training/train_tracking.py) · [Controller implementation](../td3_training/tracking.py)

## How the project uses random seeds

A seed sets the starting point of a pseudorandom number generator. The project fixes the training seeds at 37, 83, and 109 so each run can be repeated with the same settings. All three policies learn from the same saved driving dataset. [Seed list](../td3_training/routes.py#L7)

| Stage | What the random numbers do |
|---|---|
| Collect driving examples | Gymnasium route seeds 2000–2063 vary the starting position, heading and road appearance within each scene. A separate generator seeded with 5221 varies controller settings and adds steering and acceleration noise on selected drives. [Data collection](../td3_training/td3_camera.py#L25-L47) |
| Train each policy | PyTorch uses the training seed to initialize network weights. Each update samples 256 recorded driving steps. During critic training, TD3 also adds a small, bounded random change to the predicted next action when estimating future reward. [Initial training](../td3_training/td3_camera.py#L82-L135) · [Steering training](../td3_training/train_tracking.py#L35-L75) |
| Compare controllers | The evaluator uses route seeds 61000–61199 for each of three scenes. This gives TD3 and pure pursuit the same 600 cases, including the same starting offset and obstacle placement for each paired run. [Route generation](../driving_environment/route.py#L21-L31) · [Benchmark](../evaluation/tracking.py#L19-L33) |

During inference the frozen policies compute their commands directly. For example, if their steering requests are 0.10, 0.12 and 0.40, the ensemble requests 0.12. It takes the median separately for acceleration. The brake supervisor then checks whether to override the longitudinal command. [Controller](../td3_training/tracking.py#L41-L62) · [Brake check](../carla_integration/drive.py#L128-L149)

### How the teacher chooses steering

For a lookahead point a distance `L` ahead with lateral offset `y`, the implemented pure pursuit rule requests approximately `atan(2 × wheelbase × y / L²)` radians of steering. A centered point gives zero steering. A point to the left gives left steering. The camera already supplies both quantities, so each teacher action has a visible geometric explanation. A steering PID could instead track an error over time, Stanley could combine heading and cross-track error, and MPC could optimize a sequence of controls over a prediction horizon. Pure pursuit supplied the steering labels, while a separate speed rule supplied the longitudinal labels. [Teacher implementation](../td3_training/routes.py#L17-L26)

## Online TD3 from scratch

Online TD3 collected its own experience in the Gymnasium bicycle simulator using the 14 camera-based inputs, including lane history. It started from random weights, without demonstration actions or offline checkpoints. The environment rewarded visual progress and penalized camera-estimated tracking error, speed error and abrupt steering. The first 2,000 steps filled the replay buffer before learning began. During training, Gaussian exploration noise varied the controls. Each seed received 48,000 interactions (144,000 total). The actor and critics used two 128-unit hidden layers and batches of 128 replay samples. [Trainer](../td3_training/online.py) · [Environment reward](../driving_environment/route.py)

Every 16,000 steps the policy drove 24 separate validation routes, scored from camera measurements. Checkpoints were ranked first by visual finishes, then by fewer visual hazards, then by lower visual tracking error. Validation selected 48,000 steps for seed 37, 32,000 for seed 83 and 16,000 for seed 109. Test actions used the frozen checkpoints with exploration turned off. Only the held-out evaluator read simulator geometry, after each action. [Validation logs and checkpoints](../results/online_td3)

| Clean finishes out of 200 per scene | Offline TD3 | Pure pursuit | Online TD3 |
|---|---:|---:|---:|
| Straight | 198 | 197 | 173 |
| Bend | 200 | 198 | 195 |
| Obstacle bypass | 200 | 199 | 132 |

All 100 unsuccessful online runs ended short of the 39 m finish threshold, with zero recorded contacts and zero departures beyond 2.6 m. The published route CSV includes the finish, departure and contact flags for each case. Online scores describe Gymnasium driving. [Online test results](../results/online_td3/online_td3.csv) · [Offline TD3 results](../results/tracking/td3.csv) · [Pure pursuit results](../results/tracking/pure_pursuit.csv)

From the repository root, use the control environment described in the README. To replay the published online checkpoints:

```powershell
& $py -m td3_training.online evaluate --output .runtime/online-replay
```

To train fresh policies and evaluate the validation-selected weights:

```powershell
& $py -m td3_training.online train --output .runtime/online-repeat
& $py -m td3_training.online evaluate --models .runtime/online-repeat --output .runtime/online-repeat
```

## Reproduce the offline steering weights

Run from the repository root with the published dataset and longitudinal weights present. This writes the retrained steering heads to a separate directory.

```powershell
.\.venv\Scripts\python.exe -m td3_training.train_tracking --output .runtime/retrained
```

The initial data collection and longitudinal training are implemented in [td3_camera.py](../td3_training/td3_camera.py). The evaluation uses independent simulator references to score only after each decision. [Gymnasium benchmark](td3-committee.md) · [CARLA run metadata](../results/carla_td3/run.json)
