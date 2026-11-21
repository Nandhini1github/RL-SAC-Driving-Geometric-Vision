# TD3 and Pure pursuit on matched routes

The same 200 seeds (61000 through 61199) generated one straight, one bend and one obstacle-bypass route each. Each controller starts a fresh episode for every case. The lane-history feature transports previously observed lane-center points into the current vehicle frame using camera-estimated speed, the previous steering command and the bicycle model. Both controllers use the same feature extractor.

A clean finish requires reaching 39 m of the 40 m route, with no recorded contact or lane departure beyond 2.6 m. Each episode contributes one lateral tracking RMSE, calculated over its observed control steps, including unfinished episodes. The independent evaluator uses simulator geometry only after the action has been applied. In the 594 jointly finished cases, the mean RMSE was 14.64 cm for TD3 and 14.80 cm for Pure pursuit.

| Controller | Clean finishes | Lateral tracking RMSE | Contacts |
|---|---:|---:|---:|
| TD3 | 598/600 (99.7%) | 14.61 cm | 0 |
| Pure pursuit | 594/600 (99.0%) | 14.78 cm | 0 |

The paired completion difference is +0.67 percentage points, with an adjusted interval of 0 to +1.50. The paired reduction in RMSE is 0.17 cm, with an adjusted interval of 0.13 to 0.21 cm. The intervals use 20,000 bootstrap samples of route seeds, retaining all three scenes per seed, with Bonferroni adjustment for the two metrics. They quantify route variation for these frozen controllers. [TD3 episodes](../results/tracking/td3.csv) · [Pure pursuit episodes](../results/tracking/pure_pursuit.csv) · [Paired statistics](../results/tracking/summary.json)

## Reproduce

Run from the repository root. A separate output folder keeps your run alongside the published data.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m evaluation.tracking td3 --output .runtime/paired
& $py -m evaluation.tracking pure_pursuit --output .runtime/paired
& $py -m evaluation.tracking summary --output .runtime/paired
```
