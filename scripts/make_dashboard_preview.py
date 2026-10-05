"""Draw the intended Tableau dashboard from data/tableau/*.csv (a visual target to build toward; not made in Tableau)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COL = {"At rest": "#b9c2ca", "Reaching": "#3d78b8", "Grasping": "#e08a2c", "Holding": "#3f9a5e", "Releasing": "#d1544f", "Returning": "#8f66a8"}


def draw(out: Path):
    ph = pd.read_csv(ROOT / "data" / "tableau" / "tableau_phases.csv")
    sg = pd.read_csv(ROOT / "data" / "tableau" / "tableau_signals.csv")
    labels = sorted(ph["take_label"].unique())
    n = len(labels)
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(1 + n, 1, height_ratios=[2.6] + [0.7] * n, hspace=0.25)
    a1 = fig.add_subplot(gs[0])
    det = ph[ph["source"] == "Detected"]
    for i, t in enumerate(labels):
        for r in det[det["take_label"] == t].itertuples():
            a1.barh(i, r.duration_s, left=r.start_s, color=COL[r.phase_name], height=0.7, edgecolor="white", linewidth=0.5)
    a1.set_yticks(range(n)); a1.set_yticklabels(labels); a1.invert_yaxis(); a1.set_xlim(0, 35)
    a1.set_title("What the hand was doing (detected automatically from video)", loc="left", fontsize=11)
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()], labels=list(COL), ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.3), frameon=False)
    plt.setp(a1.get_xticklabels(), visible=False)
    axes = [fig.add_subplot(gs[1 + i], sharex=a1) for i in range(n)]
    ymax = float(sg["speed"].max())
    for i, (ax, t) in enumerate(zip(axes, labels)):
        g = sg[sg["take_label"] == t]
        ax.plot(g["t_s"], g["speed"], color="#33414d", lw=0.9)        # NaN speeds leave gaps where the hand was out of view
        ax.set_ylim(0, ymax); ax.set_yticks([]); ax.set_ylabel(t, rotation=0, ha="right", va="center", fontsize=10)
        if i == 0:
            ax.set_title("How fast the wrist was moving (higher = faster; each row is one recording)", loc="left", fontsize=11)
        if i < n - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel("Time (seconds)")
    for a in [a1, *axes]:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("How a hand reaches, grabs and sets down an object", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.savefig(out, dpi=90, bbox_inches="tight")


if __name__ == "__main__":
    out = ROOT / "tableau" / "target_dashboard.png"
    draw(out)
    print("wrote", out)
