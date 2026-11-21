# Can learned corrections improve camera based route completion?

## Evaluation protocol

The primary comparison is bounded residual SAC against its own geometric controller with the learned correction set to zero. The previous pure-pursuit controller is a second reference. All three use the original camera pipeline, road generator, bicycle dynamics, episode endings and independent success definition. The two availability flags expose whether the speed and lane estimates were valid. Simulator state is used to generate images and score final runs.

Before opening the new test set, the criterion for an improvement in completion was set to **at least five percentage points higher mean completion than tuned pure pursuit**, with a positive difference for each of three training seeds and no increase in the pooled observed contact rate. The unit of comparison is a matched route. The 90 distinct routes are repeated across training seeds. Average lateral tracking RMSE covers the portion driven, so it is a secondary measure.

## Training and selection

1. Tune a geometric controller on 36 camera-scored validation cases (seeds 6000 through 6011 in each scene). Test lookaheads of 4.5, 6 and 7.5 metres crossed with three speed schedules: 4 m/s, 5 m/s, and 5 m/s with a reduction for a camera-detected bend. Select by visual finishes, then fewer visual hazard endings, then lower mean episode visual lane RMSE.
2. Train SAC from scratch on top of the selected controller. The actor can change normalized steering by at most 0.20 and normalized longitudinal command by at most 0.45. The camera-derived reward credits progress and a visual finish and charges for elapsed time, lateral error, steering changes, correction magnitude, unreadable measurements and unsuccessful endings. Camera failures still end episodes under the original rules.
3. Use uniform scenario sampling and independent training seeds 37, 83 and 109, each with 32,768 steps. Evaluate checkpoints every 8,192 steps. Select the checkpoint for each seed using the same camera-only validation ranking. Keep the earliest checkpoint in a tie. This comparison tests the trained residual on top of the tuned baseline. It cannot identify which individual change to the training recipe caused an outcome.
4. After selection, evaluate all controllers on seeds 12000 through 12029 in each of the three scenes. Every actor receives only camera measurements and its previous commands. The held-out evaluator independently records true completion, contact and lateral error. Report all three seeds and all three scene types regardless of the outcome. This test is in Gymnasium.

For context, also score the three previously published fixed-training SAC checkpoints on the new routes. Those checkpoints used 12,288 transitions and their original reward. Any difference from residual SAC combines several training changes. On paired successful routes, compare time to first reach the finish threshold to expose a possible speed versus reliability trade-off.

## Reproduce on Windows

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m sac_training.residual tune
foreach ($seed in 37,83,109) { & $py -m sac_training.residual train --seed $seed }
& $py -m sac_training.residual test
& $py -m evaluation.residual_results
```
