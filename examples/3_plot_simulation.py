"""Draws the figures from the summaries of examples/2_simulation.py. Needs matplotlib.

- simulation.png shows the bias and false rejection as functions of the off-target effect of one
  guide, and the power as a function of the on-target perturbation effect. It compares Pooling
  and PETRO with parametric inference.
- simulation_permutation.png shows the false rejection and power of the pointwise test of PETRO,
  with parametric and with permutation inference. The script draws this figure only after the
  optional permutation round.

Run from the repository folder: python examples/3_plot_simulation.py
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})
# Colour and marker of each method.
STYLE = {"Pooling": ("#B51F1F", "o"), "PETRO pointwise": ("#1F3F8F", "s"), "PETRO worst case": ("#5FA6E3", "D"),
         "PETRO pointwise (permutation)": ("#8FA5DA", "s")}
LABELS = {"Pooling": "Pooling", "PETRO pointwise": "PETRO (pointwise)", "PETRO worst case": "PETRO (worst-case)"}
X_LABELS = {"delta": "Off-target effect of one guide, δ",
            "gamma": "On-target perturbation effect, Γ (all guides valid)"}
Y_LABELS = {"bias": "Bias", "rejection": "False rejection rate", "power": "Power"}


def figure(summary, rows, labels, path):
    """Draw one row of panels for each (x, y) pair and one column for each model. The legend has one
    entry for each method in labels."""
    off_target = summary[summary.pattern == "one"].sort_values("delta")
    valid = summary[summary.pattern == "none"].sort_values("gamma")
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.0, 2.1 * len(rows) + 0.4), sharey="row",
                             layout="constrained")
    for row, (x, y) in enumerate(rows):
        data = off_target if x == "delta" else valid
        for column, model in enumerate(("ols", "nb")):
            ax = axes[row, column]
            if y == "bias":
                ax.axhline(0, color="0.75", lw=0.6)
            elif y == "rejection":
                ax.axhline(0.05, color="0.4", lw=0.6, ls=(0, (3, 3)))
                ax.set_ylim(bottom=-0.03)  # keeps rates of 0 off the axis line
            else:
                ax.set_ylim(0, 1)
            values = sorted(data[x].unique())
            width = 0.8 / len(labels)
            for k, (method, label) in enumerate(labels.items()):
                if y == "bias" and method == "PETRO worst case":
                    continue  # both PETRO tests share the estimate
                line = data[(data.model == model) & (data.method == method)]
                color, marker = STYLE[method]
                if y == "power":  # grouped bars, one group per simulated Gamma
                    shift = (k - (len(labels) - 1) / 2) * width
                    ax.bar([i + shift for i in range(len(line))], line[y], width, color=color, label=label)
                else:
                    ax.plot(line[x], line[y], color=color, lw=1.2, ls="--" if "permutation" in method else "-",
                            marker=marker, ms=3.5, clip_on=False, zorder=3, label=label)
            if y == "power":
                ax.set_xticks(range(len(values)), [f"{v:.2f}" for v in values])
            else:
                ax.set_xticks(values)  # the simulated values
            ax.set_xlabel(X_LABELS[x])
            if row == 0:
                ax.set_title("OLS" if model == "ols" else "Negative binomial")
            ax.text(-0.16 if column == 0 else -0.06, 1.04, "abcdefgh"[2 * row + column],
                    transform=ax.transAxes, fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(Y_LABELS[y])
    handles, names = axes[[y for _, y in rows].index("rejection"), 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncols=len(handles), frameon=False)
    fig.savefig(path)
    plt.close(fig)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--folder", default="examples/output", help="folder of the summaries and figures")
parser.add_argument("--fig-dir", help="figure folder; defaults to --folder")
args = parser.parse_args()
start = time.perf_counter()
fig_dir = args.fig_dir or args.folder
os.makedirs(fig_dir, exist_ok=True)
summary = pd.read_csv(os.path.join(args.folder, "summary_parametric.csv"))
figures = [os.path.join(fig_dir, "simulation.png")]
figure(summary, [("delta", "bias"), ("delta", "rejection"), ("gamma", "power")], LABELS, figures[0])
permutation = os.path.join(args.folder, "summary_permutation.csv")
if os.path.exists(permutation):  # written by the optional permutation round
    figures.append(os.path.join(fig_dir, "simulation_permutation.png"))
    inference = {"PETRO pointwise": "Parametric inference", "PETRO pointwise (permutation)": "Permutation inference"}
    figure(pd.concat([summary, pd.read_csv(permutation)]), [("delta", "rejection"), ("gamma", "power")],
           inference, figures[1])
print(f"Wrote {' and '.join(figures)} ({time.perf_counter() - start:.1f} s)")
