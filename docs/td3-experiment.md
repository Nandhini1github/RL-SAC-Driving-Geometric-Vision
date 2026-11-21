# Can a learned policy beat its geometric teacher?

## Protocol fixed before final testing

The task is the existing 40 m camera-based Gymnasium route. The primary outcome is completion without a reference contact or departure. Compare the published tuned pure-pursuit controller (4.5 m lookahead, 4 m/s target) with two standalone neural policies: imitation only and imitation followed by offline TD3 with a demonstration penalty. All receive the same 11 camera-derived features and previous commands as in the original study. The networks output steering and acceleration directly at inference.

**Decision rule.** To claim an RL completion advantage, TD3 must finish a greater fraction than both the geometric teacher and imitation alone for every training seed. Its pooled contact rate must be no higher than either comparator. The lower bound of the paired 95% bootstrap interval for each completion difference must exceed zero. Report all training seeds, contacts and tracking error even if the criterion is missed. Arrival time on mutually successful routes is secondary. Different policies can finish different subsets.

## Data and training

* Collect 64 training episodes per scene on seeds 2000–2063. One quarter use the tuned teacher's actions. The others vary speed, lookahead and action noise to provide recovery and failure examples. Save the executed action, the teacher's action on the same observation, a camera-derived reward, next observation and terminal flag. Training never reads map pose, reference contact or ground-truth speed.
* Initialize each actor with 2,400 minibatches of behavior cloning. Both branches start with identical weights. Then take 3,200 further actor updates per branch with the same minibatches, teacher labels and learning rate. Only the RL branch trains twin critics on the rewards and adds a value term to the actor loss. This is an offline TD3 variant using a geometric teacher for the imitation target. The original TD3+BC algorithm regularizes toward dataset actions.
* Use seeds 37, 83 and 109, and share the training dataset. Choose from the warm-start actor and four subsequent checkpoints per branch using 30 camera-scored validation cases on seeds 6700–6709. Rank by visual completions, then fewer visual hazards, then lower visual lane RMSE. Keep the earlier checkpoint in a tie. Log whether the warm-start policy won.
* Freeze all selected weights before evaluating seeds 22000–22059 in each of three scenes (180 new routes). Compare on matched scene and seed and include all runs. Resample route seeds in blocks that retain all three scenes, and resample the three training seeds. The geometric teacher is scored once on the 180 routes.

The controllers and safety criteria in this experiment are unchanged from the camera route task apart from the learned action policies. The CARLA films illustrate the earlier SAC policies. [TD3 paper](https://proceedings.mlr.press/v80/fujimoto18a.html) · [TD3+BC paper](https://arxiv.org/abs/2106.06860)

## Reproduce

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m sac_training.td3_camera collect
foreach ($seed in 37,83,109) { & $py -m sac_training.td3_camera train --seed $seed }
& $py -m sac_training.td3_camera test --arm pursuit
foreach ($seed in 37,83,109) {
    foreach ($arm in 'td3','clone') { & $py -m sac_training.td3_camera test --arm $arm --seed $seed }
}
& $py -m evaluation.td3_results
```
