"""ZNF618 guides at SLC25A1 and TUBB4B in the RPE-1 CRISPRa screen of Southard et al.

The bundled file contains the guide effects and their covariances. Pooling and PETRO need only
these data. Each guide effect is an OLS contrast between the guide and the control cells, from a
fit of the whole screen. Expression is centred and scaled on the control cells of each GEM group.
Each covariance is the OLS covariance of the contrasts.

Pooling (published filter) uses only the bundled guides that Southard et al. kept in their final
library (Supplementary Table S4, final_tf_library). The library also kept a sixth ZNF618 guide.
This guide is not in the bundled file because its on-target activation in RPE-1 cells is not
significant. The seed matches are labels in the figure only. They do not change an estimate.

Run from the repository folder: python examples/1_real_data.py
Needs matplotlib. Writes examples/output/znf618.png. The expected figure is
examples/expected/znf618.png.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats

import petro

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", default="examples/data/znf618.npz", help="bundled guide contrasts and covariances")
parser.add_argument("--fig-dir", default="examples/output", help="folder for the PNG")
parser.add_argument("--draws", type=int, default=19999, help="null draws per pointwise test")
args = parser.parse_args()
start = time.perf_counter()
with np.load(args.data, allow_pickle=False) as data:
    bundle = {key: data[key] for key in data.files}
target, outcomes, labels = bundle["target"].item(), bundle["outcomes"], bundle["guide_labels"]
gamma, covariance, control = bundle["gamma"], bundle["covariance"], bundle["control"]
seed_bp, keep = bundle["seed_bp"], bundle["final_tf_library"]
m = len(labels)


def pooling(y, V):
    """Return the inverse-covariance-weighted (ICW) mean of the guide effects and its two-sided
    Wald P value."""
    weights = np.linalg.solve(V, np.ones(len(y)))
    estimate = weights @ y / weights.sum()
    return estimate, 2 * stats.norm.sf(abs(estimate) * np.sqrt(weights.sum()))


print(f"RPE-1 CRISPRa: {target}, {m} guides; the published filter keeps {', '.join(labels[keep])}")
print(f"Contrasts against control (control s.d.); PETRO pointwise P with {args.draws:,} draws\n")
print(f"{'Outcome':9} {'Method':28} {'Guides':>6} {'Estimate':>10} {'P':>11}")
results = []
for y, V, outcome in zip(gamma, covariance, outcomes):
    result = petro.fit(y, V)
    rows = [("Pooling", m, *pooling(y, V)),
            ("Pooling (published filter)", int(keep.sum()), *pooling(y[keep], V[np.ix_(keep, keep)])),
            ("PETRO", m, result.estimate, petro.pointwise_test(result, V, draws=args.draws, seed=123))]
    results.append(rows)
    for name, n, estimate, p in rows:
        print(f"{outcome:9} {name:28} {n:6d} {estimate:10.6f} {p:11.4g}")


def p_label(p):
    if p >= 0.01:
        return f"$P$ = {p:.2f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"$P$ = {mantissa} × 10$^{{{int(exponent)}}}$"


# Draw each guide with its 95% CI, then the three estimates. For display only, add back the control
# coefficient of the outcome. The dashed line is then the control level.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.cal": "sans", "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "axes.linewidth": 0.4, "xtick.major.width": 0.3, "ytick.major.width": 0.3, "xtick.major.size": 1.5,
    "ytick.major.size": 1.5, "axes.spines.top": False, "axes.spines.right": False,
})
CARRIER, GUIDE, MUTED = "#8F2937", "#A9B2BB", "#737980"       # seed-match guide, other guides, notes
COLORS = ["#B51F1F", "#D9892F", "#5F7FCA"]                    # Pooling, published filter, PETRO
names = ["Pooling", "Pooling\n(published filter)", "PETRO"]
positions = [m + 1.2, m + 3.5, m + 5.8]
rows_y = [-8.8 / 39, -12.1 / 39]                              # two rows under the 39-mm axes
fig, axes = plt.subplots(1, len(outcomes), figsize=(180 / 25.4, 61.8 / 25.4), squeeze=False)
fig.subplots_adjust(left=0.105, right=0.99, top=1 - 6.2 / 61.8, bottom=16.6 / 61.8, wspace=0.1)
z = stats.norm.ppf(0.975)
for i, ax in enumerate(axes[0]):
    below = ax.get_xaxis_transform()        # x in data units, y as a fraction of the axes
    ax.axhline(control[i], color="#676B70", lw=0.4, ls=(0, (3, 2)))
    ax.axvline(m, color="#9DA1A6", lw=0.3)
    for j in range(m):
        carrier = seed_bp[i, j] >= 10
        y = gamma[i, j] + control[i]
        ax.errorbar(j, y, yerr=z * np.sqrt(covariance[i, j, j]), fmt="o", color=CARRIER if carrier else GUIDE,
                    ms=3.5 if carrier else 2.5, mew=0, elinewidth=0.5)
        if carrier:
            ax.annotate(f"{int(seed_bp[i, j])} bp seed match", (j, y), xytext=(5, 0), textcoords="offset points",
                        color=CARRIER, va="center")
        ax.text(j, rows_y[0], bundle["seed_labels"][i, j], transform=below, ha="center",
                color=CARRIER if carrier else MUTED)
        if keep[j]:
            ax.plot(j, rows_y[1] + 0.025, "o", ms=3, color=COLORS[1], mew=0, transform=below, clip_on=False)
        else:
            ax.text(j, rows_y[1], "–", transform=below, ha="center", color=MUTED)
    for x, color, (_, _, estimate, p) in zip(positions, COLORS, results[i]):
        ax.plot(x, estimate + control[i], "o", ms=4, color=color, mew=0)
        ax.text(x, rows_y[0], p_label(p), transform=below, ha="center", fontsize=5)
    ax.set_xticks([*range(m), *positions], [*labels, *names])
    for j in np.flatnonzero(seed_bp[i] >= 10):
        ax.get_xticklabels()[j].set_color(CARRIER)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.6, positions[-1] + 0.8)
    ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))  # each outcome has its own scale
    ax.set_title(outcomes[i], fontstyle="italic", pad=6)
    ax.text(-0.12 if i == 0 else -0.06, 1.05, "abcdefgh"[i], transform=ax.transAxes, fontsize=8,
            fontweight="bold")
first = axes[0, 0]
first.set_ylabel("Effect on outcome gene expression\n(control s.d.)")
first.annotate("control", (-0.5, control[0]), xytext=(0, 1.2), textcoords="offset points", color=MUTED,
               fontsize=5, va="bottom")
for y, text in zip(rows_y, ["Seed (bp)", "Published filter"]):
    first.text(-0.02, y, text, transform=first.transAxes, ha="right")
os.makedirs(args.fig_dir, exist_ok=True)
path = os.path.join(args.fig_dir, f"{target.lower()}.png")
fig.savefig(path, dpi=300)
plt.close(fig)
print(f"\nWrote {path} ({time.perf_counter() - start:.1f} s)")
