"""Plot the synthetic stereo depth benchmark."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEDIA = Path("results/media")


def figures():
    MEDIA.mkdir(exist_ok=True, parents=True)
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
