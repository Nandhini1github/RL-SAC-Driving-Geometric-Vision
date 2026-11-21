# Can a learned policy beat its geometric teacher?

## Protocol fixed before final testing

The task is the existing 40 m camera-based Gymnasium route. The primary outcome is completion without a reference contact or departure. Compare the published tuned pure-pursuit controller (4.5 m lookahead, 4 m/s target) with two standalone neural policies: imitation only and imitation followed by offline TD3 with a demonstration penalty. These three comparators receive the original 11 inputs, including camera-derived features and the previous commands. The networks output steering and acceleration directly at inference.

**Decision rule.** To claim an RL completion advantage, TD3 must finish a greater fraction than both the geometric teacher and imitation alone for every training seed. Its pooled contact rate must be no higher than either comparator. The lower bound of the paired 95% bootstrap interval for each completion difference must exceed zero. Report all training seeds, contacts and tracking error even if the criterion is missed. Arrival time on mutually successful routes is secondary. Different policies can finish different subsets.

## Data and training

* Collect 64 training episodes per scene on seeds 2000–2063. One quarter use the tuned teacher's actions. The others vary speed, lookahead and action noise to provide recovery and failure examples. Save the executed action, the teacher's action on the same observation, a camera-derived reward, next observation and terminal flag. Training never reads map pose, reference contact or ground-truth speed.
* Initialize each actor with 2,400 minibatches of behavior cloning. Both branches start with identical weights. Then take 3,200 further actor updates per branch with the same minibatches, teacher labels and learning rate. Only the RL branch trains twin critics on the rewards and adds a value term to the actor loss. This is an offline TD3 variant using a geometric teacher for the imitation target. The original TD3+BC algorithm regularizes toward dataset actions.
* Use seeds 37, 83 and 109, and share the training dataset. Choose from the warm-start actor and four subsequent checkpoints per branch using 30 camera-scored validation cases on seeds 6700–6709. Rank by visual completions, then fewer visual hazards, then lower visual lane RMSE. Keep the earlier checkpoint in a tie. Log whether the warm-start policy won.
* Freeze all selected weights before evaluating seeds 22000–22059 in each of three scenes (180 new routes). Compare on matched scene and seed and include all runs. Resample route seeds in blocks that retain all three scenes, and resample the three training seeds. The geometric teacher is scored once on the 180 routes.

Two secondary comparisons use the same routes: the previous study's frozen residual SAC and the final TD3 checkpoint after all 3,200 RL actor updates. The latter shows what a policy that definitely received RL updates did when validation favored a warm start. Residual SAC retains its two additional camera-validity flags. Both comparisons were specified before opening the held-out set. All arms use the same environment and final scoring criteria. The CARLA films illustrate the earlier SAC policies. [TD3 paper](https://proceedings.mlr.press/v80/fujimoto18a.html) · [TD3+BC paper](https://arxiv.org/abs/2106.06860)

## What changes during the RL updates?

### Independent replication protocol

The secondary final-TD3 comparison in the first 180 routes motivated one independent replication. Before testing, freeze the three final TD3 actors and the three selected continuing-imitation actors, without further updates or selection. Evaluate seeds 31000–31099 in all three scenes (300 distinct routes). The primary comparators are the published tuned teacher and continuing imitation. Use the same per-seed, contact and paired-interval criteria as before, now applied to the final RL actors. Report paired travel time and tracking error. Also score a conservative pure-pursuit reference with the same 4.5 m lookahead and a 3.75 m/s target to check whether a small speed reduction can achieve a similar completion rate. This additional reference is a sensitivity check. Publish this one replication irrespective of outcome.

The actor starts by minimizing squared error against the teacher's steering and acceleration. During the RL phase, two critics learn discounted future camera reward from the actions actually driven in the simulator. Their training target uses the smaller target-network value, target-action smoothing, a 0.99 discount and zero bootstrap after the end of an episode. Every second critic update, the actor balances imitation error against the first critic's estimate of its proposed action's value. The imitation-only control receives the same minibatch of observations and labels on each such step. The actor uses only its learned weights and training-set normalization at deployment.

The reward credits visual progress and adds 25 for a visual finish. It charges 0.12 per step, 0.10 times absolute measured lane offset, 0.08 times absolute steering change, and 1.5 for a frame with missing lane or speed. A visual hazard or lane failure incurs 25, and a timeout without another terminal event incurs 12. Independent collision and finish references never enter the training targets.

## Reproduce

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m sac_training.td3_camera collect
foreach ($seed in 37,83,109) { & $py -m sac_training.td3_camera train --seed $seed }
& $py -m sac_training.td3_camera test --arm pursuit
foreach ($seed in 37,83,109) {
    foreach ($arm in 'td3','clone','residual','final') { & $py -m sac_training.td3_camera test --arm $arm --seed $seed }
}
& $py -m evaluation.td3_results
```
