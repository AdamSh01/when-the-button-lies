"""Generate the headline figure from the pilot run data.

Two panels, one per model family, each plotting per-prediction confidence split by
whether the prediction was correct. The point of the figure is the *non*-separation:
incorrect predictions are not lower-confidence than correct ones — neither for a
trained web world model nor for a conflict-prompted frontier LLM.

Data is the hand-graded pilot exactly as run (n=6 for WM-C0, n=15 for the LLM).
Confidence scales differ by construction (log-prob vs self-reported), so the two
families are shown in separate panels and never compared on a shared axis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# (block, confidence, is_correct, kind)
# WM-C0 = stock WebWorld-8B (Qwen3-8B based), confidence = geometric-mean token prob
c0 = [("05", 0.83, False, "conflict"), ("08", 0.82, False, "conflict"),
      ("10", 0.80, False, "conflict"), ("11", 0.90, False, "conflict"),
      ("12", 0.85, True,  "honest"),   ("15", 0.88, False, "honest")]

# Conflict-prompt frontier LLM, confidence = model's self-reported number, 15 blocks
llm = [("01", 0.35, True,  "conflict"), ("02", 0.35, True,  "conflict"),
       ("03", 0.35, False, "conflict"), ("04", 0.35, False, "conflict"),
       ("05", 0.35, True,  "conflict"), ("06", 0.35, True,  "conflict"),
       ("07", 0.72, False, "conflict"), ("08", 0.35, False, "conflict"),
       ("09", 0.45, False, "conflict"), ("10", 0.35, False, "conflict"),
       ("11", 0.55, False, "conflict"), ("12", 0.75, True,  "honest"),
       ("13", 0.75, True,  "honest"),   ("14", 0.40, True,  "conflict"),
       ("15", 0.35, True,  "honest")]

COL = {"conflict": "#d1495b", "honest": "#2a9d8f"}


def panel(ax, data, title, ylabel):
    for label, xc in [("Correct", 0), ("Incorrect", 1)]:
        want = (label == "Correct")
        pts = sorted([d for d in data if d[2] == want], key=lambda d: d[1])
        n = len(pts)
        offs = np.linspace(-0.26, 0.26, n) if n > 1 else [0.0]
        for (name, conf, corr, kind), dx in zip(pts, offs):
            x = xc + dx
            ax.scatter(x, conf, c=COL[kind], s=150, edgecolor="white",
                       linewidth=1.4, zorder=3)
            ax.annotate(name, (x, conf), fontsize=6.5, ha="center", va="center",
                        color="white", zorder=4, fontweight="bold")
        vals = [d[1] for d in pts]
        if vals:
            m = float(np.mean(vals))
            ax.plot([xc - 0.34, xc + 0.34], [m, m], color="#1d3557", lw=2.4, zorder=2)
            ax.annotate(f"mean {m:.2f}", (xc - 0.36, m), ha="right", va="center",
                        fontsize=8, color="#1d3557", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Correct", "Incorrect"], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
panel(axes[0], c0, "WebWorld-8B  (trained web world model)", "log-prob confidence")
panel(axes[1], llm, "Conflict-prompt frontier LLM", "self-reported confidence")

h_conf = mlines.Line2D([], [], marker="o", color="w", markerfacecolor=COL["conflict"],
                       markersize=11, label="label\u2013behavior conflict")
h_hon = mlines.Line2D([], [], marker="o", color="w", markerfacecolor=COL["honest"],
                      markersize=11, label="honest label")
h_mean = mlines.Line2D([], [], color="#1d3557", lw=2.4, label="group mean")
fig.legend(handles=[h_conf, h_hon, h_mean], loc="upper center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=9)

fig.suptitle("Confidence does not separate correct from incorrect predictions",
             y=1.10, fontsize=13.5, fontweight="bold")
fig.text(0.5, -0.04,
         "Pilot, hand-graded (n=6 WM-C0, n=15 LLM). If the models were calibrated, the "
         "Incorrect column would sit well below Correct.\nIt does not \u2014 and every "
         "label\u2013behavior conflict the trained model saw, it got wrong, at 0.80\u20130.90 confidence.",
         ha="center", fontsize=8.5, color="#555")
plt.tight_layout()
plt.savefig("results/calibration.png", dpi=170, bbox_inches="tight", facecolor="white")
print("saved results/calibration.png")
