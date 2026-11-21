"""Summarize paired trials without treating repeated routes as independent seeds."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT=Path("results")
SEEDS=(37,83,109)


def main():
    tables=[]
    for seed in SEEDS:
        folder=OUT/("control" if seed==37 else f"control_seed{seed}")
        tables.append(pd.read_csv(folder/"heldout.csv").assign(training_seed=seed))
    data=pd.concat(tables)
    metrics=["success","collision","departure","return_sum","rmse_m","steering_delta","latency_ms"]
    summary=data.groupby(["training_seed","controller"])[metrics].mean()
    summary.to_csv(OUT/"per_seed.csv")
    summary.groupby("controller").mean().to_csv(OUT/"aggregate.csv")
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    for ax,key,title in zip(axes,("success","rmse_m","return_sum"),("Completion fraction","Tracking RMSE (m)","Episode return")):
        for arm,color in (("fixed","#24748b"),("adaptive","#ab561b"),("pursuit","#524076")):
            s=summary.xs(arm,level="controller")[key]
            ax.plot(range(len(s)),s,"o-",label=arm,color=color)
        ax.set(xticks=range(len(SEEDS)),xticklabels=SEEDS,xlabel="Training seed",ylabel=title)
        ax.legend(fontsize=8)
    fig.savefig(OUT/"media/training-seeds.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
