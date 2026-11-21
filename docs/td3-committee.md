# Can three camera policies recover more routes together?

In the earlier 300-route replication, one of the three frozen TD3 actors accounted for 14 of the 20 unsuccessful trials, including all eight that ended on missing camera input. This motivated one change to inference. Each trained actor sees the same camera features. The controller applies the median of their three steering requests and the median of their three acceleration requests. A single extreme request therefore cannot determine either command. The network weights stay frozen. There is no geometric controller in this inference path.

## Test design and reporting scope

* Test once on route seeds 45000 through 45199 in each of the original three scenes. These 600 cases have not been used for model selection. Every controller receives the same camera pipeline, action limits, 40 m route task and episode-ending rules. Only the evaluator reads reference pose and contacts.
* The published comparisons use the tuned teacher (4 m/s) and an equally sized median of imitation-only actors. The reporting scope was narrowed after the protocol was registered. Also score all three individual final TD3 actors to isolate the effect of combining their actions. Choose no model, speed, aggregation rule or route after opening this set.
* The primary metric is clean route completion. Report the raw counts separately from the evidence for superiority. To establish a consistent advantage, require a positive lower confidence bound against each primary comparator and no additional observed contacts. Use 20,000 paired bootstrap samples grouped by route seed, with all three scenes kept together, and the originally specified percentile limits (0.833rd and 99.167th percentiles). Report time to the finish on jointly successful routes and lateral tracking RMSE as secondary outcomes.
* This is a comparison of these specific already-trained controllers in Gymnasium. The three actors constitute one deployed committee. They do not yield three independent replications of committee training.

## What happened on the 600 routes?

| Result on the same 600 routes | TD3 ensemble | Tuned pure pursuit |
|---|---:|---:|
| Clean finishes | 598 (99.7%) | 584 (97.3%) |
| Recorded contacts | 0 | 0 |
| Straight finishes | 199/200 | 197/200 |
| Bend finishes | 200/200 | 189/200 |
| Obstacle bypass finishes | 199/200 | 198/200 |
| Episodes ending after consecutive missing camera inputs | 0 | 13 |
| Mean episode lateral tracking RMSE | 0.218 m | 0.174 m |

Both finished 582 of the same routes. TD3 finished 16 routes the teacher missed, and the teacher finished two routes TD3 missed. The paired difference was +2.33 percentage points, with an adjusted interval from +0.83 to +4.00 points. On the jointly completed routes TD3 reached the finish 0.066 seconds later on average. This comparison supports an improvement in route completion in this simulator. [TD3 episode scores](../results/td3/committee/td3_median.csv) · [Teacher episode scores](../results/td3/committee/pursuit.csv) · [Paired statistics](../results/td3/committee/head-to-head.json)

**What did reward-based learning add beyond imitation?** Combining three imitation-only networks in exactly the same way yielded 586/600 finishes. The TD3 ensemble gained 12 finishes, a paired difference of +2.00 percentage points with an adjusted interval of +0.50 to +3.83. Both recorded zero contacts. Used individually, the frozen TD3 actors for seeds 37, 83 and 109 completed 588, 593 and 592 routes respectively. This measures one ensemble on new routes within the same simulator. [Scores for all six controllers](../results/td3/committee/summary.json)

## Reproduce the head-to-head

The README compares the learned controller with the tuned controller that supplied its training demonstrations. The commands below require only these two arms. The report retains the interval adjustment from the original protocol, so its counts, timing and interval match the corresponding full-benchmark comparison.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m evaluation.td3_committee td3_median
& $py -m evaluation.td3_committee pursuit
& $py -m evaluation.td3_committee head-to-head
```

## Reproduce the full experiment

```powershell
$py = ".\.venv\Scripts\python.exe"
foreach ($arm in 'td3_median','imitation_median','pursuit','td3_37','td3_83','td3_109') {
    & $py -m evaluation.td3_committee $arm
}
& $py -m evaluation.td3_committee summary
```
