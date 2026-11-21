# Can three camera policies recover more routes together?

In the earlier 300-route replication, one of the three frozen TD3 actors accounted for 14 of the 20 unsuccessful trials, including all eight that ended on missing camera input. This motivated one change to inference. Each trained actor sees the same camera features. The controller applies the median of their three steering requests and the median of their three acceleration requests. A single extreme request therefore cannot determine either command. The network weights stay frozen. There is no geometric controller in this inference path.

## Protocol fixed before testing

* Test once on route seeds 45000 through 45199 in each of the original three scenes. These 600 cases have not been used for model selection. Every controller receives the same camera pipeline, action limits, 40 m route task and episode-ending rules. Only the evaluator reads reference pose and contacts.
* Primary comparisons are the published tuned pure pursuit (4 m/s), the slower pure pursuit (3.75 m/s), and an equally sized median of imitation-only actors. Also score all three individual final TD3 actors to isolate the effect of combining their actions. Choose no model, speed, aggregation rule or route after opening this set.
* The primary metric is clean route completion. Report the raw counts separately from the evidence for superiority. To establish a consistent advantage, require a positive lower confidence bound against each primary comparator and no additional observed contacts. Use 20,000 paired bootstrap samples grouped by route seed, with all three scenes kept together, and Bonferroni-adjusted percentile intervals for the three comparisons (nominal simultaneous coverage 95%). Report time to the finish on jointly successful routes and lateral tracking RMSE as secondary outcomes.
* This is a comparison of these specific already-trained controllers in Gymnasium. The three actors constitute one deployed committee. They do not yield three independent replications of committee training.

## Reproduce

```powershell
$py = ".\.venv\Scripts\python.exe"
foreach ($arm in 'td3_median','imitation_median','pursuit','slower_pursuit','td3_37','td3_83','td3_109') {
    & $py -m evaluation.td3_committee $arm
}
& $py -m evaluation.td3_committee summary
```
