"""Build static plots and a side-by-side closed-loop Gym replay from saved results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/control")
MEDIA = Path("results/media")
COLORS = ["#24748b","#8c499c","#c26c27"]


def figures():
    MEDIA.mkdir(exist_ok=True, parents=True)
    test = pd.read_csv(OUT/"heldout.csv")
    keys = ["success","collision","departure","rmse_m","return_sum","steering_delta","latency_ms"]
    test.groupby("controller")[keys].mean().to_csv(OUT/"summary.csv")
    valid = pd.read_csv(OUT/"validation.csv")
    fig, axes = plt.subplots(1,3, figsize=(12,3.5),layout="constrained")
    for j, name in enumerate(("fixed","adaptive")):
        base = pd.read_csv(OUT/"shared.monitor.csv",comment="#")
        tail = pd.read_csv(OUT/f"{name}.monitor.csv",comment="#")
        t = np.r_[base.l.cumsum(),4096+tail.l.cumsum()]
        ret = pd.concat([base.r,tail.r],ignore_index=True).rolling(12,min_periods=1).mean()
        axes[0].plot(t,ret,label=name,color=COLORS[j])
        vals=valid[valid.arm==name].groupby("steps")
        axes[1].plot(vals.success.mean().index, vals.success.mean(), "o-", color=COLORS[j],label=name)
    axes[0].set(xlabel="Training transitions",ylabel="Return (12-episode mean)")
    axes[1].set(xlabel="Training transitions",ylabel="Validation completion fraction",ylim=(-0.05,1.05))
    test.groupby(["scene","controller"]).rmse_m.mean().unstack().plot.bar(ax=axes[2],color=COLORS,rot=0)
    axes[2].set(xlabel="Held-out scene",ylabel="Mean episode tracking RMSE (m)")
    for ax in axes: ax.legend(fontsize=8)
    fig.savefig(MEDIA/"control.png",dpi=160)
    plt.close(fig)
    rows=json.loads(Path("results/stereo_synthetic.json").read_text())["rows"]
    fig,ax=plt.subplots(figsize=(6,3.4),layout="constrained")
    for key,label in (("mae_m","MAE"),("rmse_m","RMSE")):
        ax.plot([r["depth_m"] for r in rows],[r[key] for r in rows],"o-",label=label)
    ax.set(xlabel="True optical depth (m)",ylabel="Valid-pixel error (m)",title="Rectified synthetic planes")
    ax.legend()
    fig.savefig(MEDIA/"stereo.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    figures()
