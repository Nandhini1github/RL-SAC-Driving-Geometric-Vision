"""Paired route comparisons for the camera-only TD3 experiment."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sac_training.td3_camera import OUT, SCENES, SEEDS, STAGES


def read_scores(path, count=180):
    scores = pd.read_csv(path).set_index(["seed", "scene"]).sort_index()
    if len(scores) != count or scores.index.has_duplicates:
        raise ValueError(f"Expected {count} unique held-out routes in {path}")
    return scores


def paired(rl, other, rng):
    changes, time_changes = [], []
    for r, b in zip(rl, other):
        assert r.index.equals(b.index)
        changes.append((r.success-b.success).unstack("scene").loc[:, list(SCENES)].to_numpy())
        both = r.success.eq(1) & b.success.eq(1)
        time_changes.extend((r.first_arrival_s[both]-b.first_arrival_s[both]).tolist())
    d = np.stack(changes)
    training_ids = rng.integers(0, len(SEEDS), size=(10000, len(SEEDS), 1))
    route_ids = rng.integers(0, d.shape[1], size=(10000, 1, d.shape[1]))
    boot = d[training_ids, route_ids, :].mean(axis=(1, 2, 3))*100
    return dict(gain_pp=float(d.mean()*100), ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
                per_seed_net_routes=d.sum(axis=(1, 2)).astype(int).tolist(),
                jointly_finished=len(time_changes), paired_arrival_difference_s=float(np.mean(time_changes)))


def describe(scores):
    summary = {}
    for arm, tables in scores.items():
        data = pd.concat(tables)
        summary[arm] = dict(episodes=len(data), successes=int(data.success.sum()),
                            contacts=int(data.collision.sum()), mean_episode_rmse_m=float(data.rmse_m.mean()),
                            completions_per_run=[int(table.success.sum()) for table in tables],
                            scene_successes={scene:int(data.xs(scene,level="scene").success.sum()) for scene in SCENES})
    return summary


def meets(comparisons, summary, arm):
    return all(min(c["per_seed_net_routes"]) > 0 and c["ci95_pp"][0] > 0 and
               summary[arm]["contacts"]/summary[arm]["episodes"] <= summary[name]["contacts"]/summary[name]["episodes"]
               for name,c in comparisons.items())


def confirmation():
    directory = OUT/"confirmation"
    scores = {arm:[read_scores(directory/f"{arm}.csv",300)] for arm in ("pursuit","conservative")}
    for arm in ("final","clone"):
        scores[arm] = [read_scores(directory/str(seed)/f"{arm}.csv",300) for seed in SEEDS]
    rng = np.random.default_rng(804)
    summary = describe(scores)
    comparisons = {name:paired(scores["final"], scores[name] if name == "clone" else scores[name]*len(SEEDS), rng)
                   for name in ("pursuit", "clone", "conservative")}
    primary = {name:comparisons[name] for name in ("pursuit", "clone")}
    report = dict(arms=summary, paired_final_minus_comparator=comparisons,
                  meets_criterion=bool(meets(primary,summary,"final")))
    output = json.dumps(report, indent=2)
    (directory/"summary.json").write_text(output+"\n")
    print(output)
    fig,ax = plt.subplots(figsize=(8.4,3.7), layout="constrained")
    for i,(arm,title) in enumerate((("pursuit","Tuned pure pursuit"),("clone","Imitation only"),
                                   ("conservative","Slower pure pursuit"))):
        c = comparisons[arm]
        lo, hi = c["ci95_pp"]
        ax.hlines(i,lo,hi,color="#2d7184",linewidth=3)
        ax.plot(c["gain_pp"],i,"o",color="#193d48")
    ax.axvline(0,color="#858585",ls="--")
    ax.set(yticks=[0,1,2], yticklabels=["Tuned pure pursuit","Imitation only","Slower pure pursuit"],
           xlabel="TD3 minus comparator completion (percentage points)",ylim=(2.6,-.6),
           title="Frozen policies on 300 fresh routes • paired 95% intervals")
    fig.savefig(directory/"paired.png",dpi=160)
    plt.close(fig)


def main():
    teacher = read_scores(OUT/"pursuit.csv")
    scores = {arm:[read_scores(OUT/str(seed)/f"{arm}.csv") for seed in SEEDS]
              for arm in ("td3","clone","residual","final")}
    scores["pursuit"] = [teacher]
    summary = describe(scores)
    rng = np.random.default_rng(804)
    comparisons = {"pursuit":paired(scores["td3"], [teacher]*len(SEEDS), rng),
                   "clone":paired(scores["td3"], scores["clone"], rng)}
    final_comparisons = {"pursuit":paired(scores["final"], [teacher]*len(SEEDS), rng),
                         "clone":paired(scores["final"], scores["clone"], rng)}
    selected = {}
    for seed in SEEDS:
        history = json.loads((OUT/str(seed)/"validation.json").read_text())
        assert len(history) == 2*(STAGES+1)
        selected[seed] = {arm:max((s for s in history if s["arm"] == arm),
                                  key=lambda s:s["validation_rank"])["post_warmup_actor_updates"]
                          for arm in ("td3","clone")}
    passed = meets(comparisons, summary, "td3")
    report = dict(arms=summary, paired_td3_minus_comparator=comparisons,
                  paired_final_minus_comparator=final_comparisons,
                  selected_actor_updates_after_warmup=selected, meets_criterion=bool(passed))
    output=json.dumps(report, indent=2)
    (OUT/"summary.json").write_text(output+"\n")
    print(output)
    arms = ["pursuit", "residual", "clone", "td3", "final"]
    selection_label = ("Imitation warm start (selected)" if all(s["td3"] == 0 for s in selected.values())
                       else "Validation-selected TD3 branch")
    names = ["Tuned pure pursuit", "Residual SAC", "Continuing imitation only", selection_label, "TD3 final RL checkpoint"]
    colors = ["#586872", "#69958c", "#b48047", "#246c92", "#53458a"]
    x = np.arange(len(SCENES))
    fig, ax = plt.subplots(figsize=(8.2,4.4), layout="constrained")
    for j, (arm, name, color) in enumerate(zip(arms,names,colors)):
        rates = [summary[arm]["scene_successes"][s]/(summary[arm]["episodes"]/3)*100 for s in SCENES]
        ax.bar(x+(j-2)*.17, rates, .16, label=name, color=color)
    ax.set(xticks=x, xticklabels=["Straight","Bend","Obstacle"], ylim=(0,105),
           ylabel="Routes completed (%)", title="Camera-only controllers on 180 new routes")
    ax.legend(loc="lower center", ncol=2)
    fig.savefig(OUT/"completion.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--confirm",action="store_true")
    if parser.parse_args().confirm:
        confirmation()
    else:
        main()
