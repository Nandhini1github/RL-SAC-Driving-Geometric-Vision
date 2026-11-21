"""Report the predeclared residual-controller comparison on matched routes."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sac_training.residual import OUT, SCENES, SEEDS


def main():
    data = pd.read_csv(OUT/"heldout.csv")
    direct = OUT/"heldout_direct.csv"
    if direct.exists():
        data = pd.concat([data, pd.read_csv(direct)], ignore_index=True)
    baseline = data[data.controller == "tuned_pursuit"].set_index(["scene", "seed"]).sort_index()
    per_seed = []
    differences = []
    for seed in SEEDS:
        policy = data[(data.controller == "residual_sac") & (data.training_seed == seed)]
        policy = policy.set_index(["scene", "seed"]).sort_index()
        assert len(policy) == 90 and baseline.index.equals(policy.index)
        diff = policy.success - baseline.success
        differences.append(diff.unstack("scene").to_numpy())
        joint = (policy.success == 1) & (baseline.success == 1)
        per_seed.append(dict(training_seed=seed, completions=int(policy.success.sum()),
                             gained=int((diff == 1).sum()), lost=int((diff == -1).sum()),
                             joint_finishes=int(joint.sum()), contacts=int(policy.collision.sum()),
                             paired_arrival_difference_s=float((policy.first_arrival_s[joint] -
                                                               baseline.first_arrival_s[joint]).mean())))
        matched_direct = data[(data.controller == "matched_direct_sac") & (data.training_seed == seed)]
        if not matched_direct.empty:
            matched_direct = matched_direct.set_index(["scene", "seed"]).sort_index()
            assert policy.index.equals(matched_direct.index)
            vs_direct = policy.success - matched_direct.success
            per_seed[-1].update(direct_completions=int(matched_direct.success.sum()),
                                gained_vs_direct=int((vs_direct == 1).sum()),
                                lost_vs_direct=int((vs_direct == -1).sum()))
    # Resample policy seeds and route seeds. Each route seed retains all three scenes.
    diff = np.stack(differences)
    rng = np.random.default_rng(7701)
    trials = 10000
    policy_ids = rng.integers(0, len(SEEDS), size=(trials, len(SEEDS), 1))
    route_ids = rng.integers(0, diff.shape[1], size=(trials, 1, diff.shape[1]))
    bootstrap = diff[policy_ids, route_ids, :].mean(axis=(1, 2, 3))
    names = {"original_pursuit": ("Original pursuit", "#a5afb6"), "tuned_pursuit": ("Tuned pursuit", "#51646c"),
             "standalone_sac": ("Direct SAC (12k)", "#dfaa7a"), "matched_direct_sac": ("Direct SAC (33k)", "#a46330"),
             "residual_sac": ("Residual SAC (33k)", "#197a73")}
    arms = [arm for arm in names if arm in set(data.controller)]
    totals = {}
    for arm in arms:
        d = data[data.controller == arm]
        totals[arm] = dict(completions=int(d.success.sum()), episodes=len(d),
                           contacts=int(d.collision.sum()), mean_episode_rmse_m=float(d.rmse_m.mean()),
                           camera_loss_endings=int(d.ended_on_missing_input.sum()),
                           missing_speed_frame_fraction=float(d.missing_speed_frames.sum()/d.steps.sum()),
                           by_scene={s:int(d.loc[d.scene == s, "success"].sum()) for s in SCENES})
    meets = (diff.mean() >= .05 and all(r["gained"] > r["lost"] for r in per_seed) and
             sum(r["contacts"] for r in per_seed) <= len(SEEDS) * baseline.collision.sum())
    report = dict(totals=totals, residual_by_seed=per_seed,
                  paired_completion_gain_pp=float(100*diff.mean()),
                  paired_bootstrap_95_pp=(100*np.quantile(bootstrap,[.025,.975])).tolist(),
                  meets_predeclared_criterion=bool(meets))
    output = json.dumps(report, indent=2)
    (OUT/"summary.json").write_text(output+"\n")
    print(output)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    x = np.arange(len(SCENES))
    width = .84 / len(arms)
    for j, arm in enumerate(arms):
        label, color = names[arm]
        fraction = data[data.controller == arm].groupby("scene").success.mean().reindex(SCENES)
        left.bar(x+(j-(len(arms)-1)/2)*width, 100*fraction, width*.94, color=color, label=label)
    left.set(xticks=x, xticklabels=["Straight", "Bend", "Obstacle"], ylim=(0, 105),
             ylabel="Routes completed (%)", title="Fresh held-out routes")
    left.legend(loc="lower left", fontsize=8)
    y = np.arange(len(SEEDS))
    right.barh(y+.16, [r["gained"] for r in per_seed], .3, color=names["residual_sac"][1], label="Only residual SAC finished")
    right.barh(y-.16, [r["lost"] for r in per_seed], .3, color=names["tuned_pursuit"][1], label="Only tuned pursuit finished")
    right.set(yticks=y, yticklabels=[str(s) for s in SEEDS], ylabel="SAC training seed",
              xlabel="Number of routes (out of 90)", title="What changed on matched routes?")
    right.legend(fontsize=8)
    fig.savefig(OUT/"completion.png", dpi=155)
    plt.close(fig)

    fig, (returns, validation) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for seed in SEEDS:
        log = pd.read_csv(OUT/str(seed)/"episodes.monitor.csv", comment="#")
        returns.plot(log.l.cumsum(), log.r.rolling(20, min_periods=1).mean(), label=str(seed))
        stages = json.loads((OUT/str(seed)/"validation.json").read_text())
        validation.plot([s["steps"] for s in stages],
                        [s["validation_rank"][0]/36*100 for s in stages], "o-", label=str(seed))
    returns.set(xlabel="Training steps", ylabel="Return (20-episode average)")
    validation.set(xlabel="Training steps", ylabel="Camera-scored validation finishes (%)", ylim=(0, 103))
    for ax in (returns, validation):
        ax.legend(title="Training seed", fontsize=8)
    fig.savefig(OUT/"training.png", dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()
