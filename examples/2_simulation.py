"""Simulation of bias, false rejection and power for Pooling and PETRO, with OLS and NB guide fits.

The default round uses parametric inference. It tests each target with Pooling and with the
pointwise and worst-case tests of PETRO. Pooling fits one indicator for all guides of the target.
The optional round (--inference permutation) uses the same simulated data. It calculates
permutation P values for the pointwise test of PETRO.

Each simulated target has five guides with 150 cells each, and 900 control cells. The effect of
guide g is Gamma + delta_g.
- Off-target settings: Gamma = 0, and one guide has an off-target effect d (deltas 0, 0, 0, 0, d).
  The value of d is -0.8, -0.4, 0, 0.4 or 0.8.
- Power settings: Gamma = 0.05, 0.1, 0.15 or 0.2. No guide has an off-target effect.
OLS outcomes are Gaussian with unit variance. NB outcomes have a control mean of 0.494 and a size
of 3.32. NB effects are natural-log ratios.

This simulation is a simplified version of the simulation in the accompanying paper.
Run from the repository folder:
    python examples/2_simulation.py                            # parametric
    python examples/2_simulation.py --inference permutation    # optional
The defaults (50 replicates, 199 draws) give a quick run. The expected figures in examples/expected
use --replicates 2000 --draws 999. For more precise rates, increase --replicates. For finer P
values, increase --draws. To use more cores, increase --jobs.
"""
import os

os.environ.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS")})

import argparse
from multiprocessing import Pool
import time

import numpy as np
import pandas as pd

import petro
from petro.adapters import fit_guides, fit_pooled

N_CONTROL, N_GUIDE, N_GUIDES = 900, 150, 5
MU0, THETA = 0.4937622398138046, 3.3192731722847437
LABELS = np.repeat(np.arange(N_GUIDES + 1), [N_CONTROL] + [N_GUIDE] * N_GUIDES)
PATTERNS = {"none": [0, 0, 0, 0, 0], "one": [0, 0, 0, 0, 1]}


def settings():
    rows = [dict(model=model, pattern="one", gamma=0.0, delta=delta)
            for model in ("ols", "nb") for delta in (-0.8, -0.4, 0.0, 0.4, 0.8)]
    rows += [dict(model=model, pattern="none", gamma=gamma, delta=0.0)
             for model in ("ols", "nb") for gamma in (0.05, 0.1, 0.15, 0.2)]
    return [dict(row, setting=i) for i, row in enumerate(rows)]


def replicate(task):
    setting, index, inference, draws, seed = task
    rng = np.random.default_rng([seed, setting["setting"], index])
    mean = np.r_[0, setting["gamma"] + setting["delta"] * np.array(PATTERNS[setting["pattern"]])][LABELS]
    if setting["model"] == "ols":
        y = mean + rng.standard_normal(len(LABELS))
    else:
        y = rng.negative_binomial(THETA, THETA / (THETA + MU0 * np.exp(mean)))
    effects = fit_guides(y, LABELS, setting["model"], theta=THETA)
    result = petro.fit(effects.gamma, effects.working)
    parametric_seed, permutation_seed = rng.integers(2**63, size=2)
    common = dict(setting, replicate=index)
    petro_row = dict(common, estimate=result.estimate, status=result.status, low_support=result.low_support)
    if inference == "permutation":
        pvalue = petro.permutation_test(y, LABELS, setting["model"], theta=THETA, draws=draws,
                                        seed=int(permutation_seed))
        return [dict(petro_row, method="PETRO pointwise (permutation)", pvalue=pvalue)]
    pooled, _, pooled_p = fit_pooled(y, LABELS, setting["model"], theta=THETA)
    pointwise = petro.pointwise_test(result, effects.sampling, draws=draws, seed=int(parametric_seed))
    worst_case = (petro.worst_case_test_nb(y, LABELS, theta=THETA) if setting["model"] == "nb"
                  else petro.worst_case_test(result, effects.sampling))
    return [dict(common, method="Pooling", estimate=pooled, pvalue=pooled_p),
            dict(petro_row, method="PETRO pointwise", pvalue=pointwise),
            dict(petro_row, method="PETRO worst case", pvalue=worst_case)]


def summarize(table, level=0.05):
    """Summarize each setting and method. Bias: the mean bias and its Monte Carlo SE. Rejection:
    the rejection rate, which is the false rejection rate when Gamma = 0. Power: the rate of
    rejections with an estimate of the correct sign, when Gamma > 0."""
    table = table.assign(bias=table.estimate - table.gamma, reject=table.pvalue <= level)
    table["power"] = table.reject & (np.sign(table.estimate) == np.sign(table.gamma))
    keys = ["model", "pattern", "gamma", "delta", "method"]
    summary = table.groupby(keys, sort=False).agg(
        replicates=("bias", "size"), bias=("bias", "mean"), bias_se=("bias", "sem"),
        rejection=("reject", "mean"), power=("power", "mean")).reset_index()
    summary.loc[summary.gamma == 0, "power"] = np.nan
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--inference", choices=("parametric", "permutation"), default="parametric")
    parser.add_argument("--draws", type=int, default=199, help="null draws per pointwise test")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="examples/output", help="output folder")
    args = parser.parse_args()
    grid = settings()
    tasks = [(s, i, args.inference, args.draws, args.seed) for s in grid for i in range(args.replicates)]
    print(f"Simulation, {args.inference} inference: {len(tasks):,} targets ({len(grid)} settings x "
          f"{args.replicates} replicates), {args.draws} draws per pointwise test, {args.jobs} job(s)")
    start, rows = time.perf_counter(), []
    with Pool(args.jobs) as pool:
        for done, result in enumerate(pool.imap(replicate, tasks, chunksize=10), 1):
            rows += result
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                elapsed = time.perf_counter() - start
                print(f"  {done / len(tasks):4.0%}  {done:6,} targets  {elapsed:5.0f} s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    table = pd.DataFrame(rows)
    replicates = os.path.join(args.out, f"replicates_{args.inference}.csv")
    summary = os.path.join(args.out, f"summary_{args.inference}.csv")
    table.to_csv(replicates, index=False)
    summarize(table).to_csv(summary, index=False)
    print(f"Wrote {replicates} and {summary}; done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
