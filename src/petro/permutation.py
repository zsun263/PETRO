"""Permutation tests of Gamma = 0 and of alpha = 0, for OLS and NB guide fits on cells.

The null draws for a support S (a strict majority of guides) shuffle the labels of the guides
in S together with the controls. When blocks are given, the shuffles stay within blocks. The
labels of the other guides stay fixed. The responses, covariates, offsets and NB sizes also stay
fixed. Every draw refits the guide-level model:

- Pointwise test: every draw refits PETRO. The test spreads the draws over the supports as in
  pointwise_test.
- Worst-case test: each smallest support S gets its own draws. OLS uses the ICW statistic of
  the guides in S. NB refits one shared coefficient for S and keeps separate coefficients for
  the other guides. It divides this coefficient by its HC1 SE from the pooled model. The P value
  is the largest of the P values of the supports.
"""
import itertools
from functools import partial

import numpy as np
import pandas as pd

from .adapters.nb_grouped import fit_nb_grouped, grouped_influences
from .adapters.regression import (_design, _fit_nb_support, _hc1, _indicators, _nb_fit, _ols_fit,
                                  _ols_joint, check_labels, fit_guides)
from .alpha import _eligible_indices, fit_alpha
from .estimator import C, Q_MAX, fit
from .inference import _check_draws, _icw_statistic, _pvalue, _supports, statistic


def _check(labels, blocks, test, draws):
    if test not in ("pointwise", "worst_case"):
        raise ValueError("test must be 'pointwise' or 'worst_case'")
    _check_draws(draws)
    blocks = np.zeros(len(labels)) if blocks is None else np.asarray(blocks)
    if blocks.shape != labels.shape or pd.isna(blocks).any():
        raise ValueError("blocks must give every cell a (non-missing) block")
    return blocks


def _grouped(offset, covariates):
    """Whether NB draws can be refitted from group sums: no covariates and a constant offset."""
    return covariates is None and (offset is None or np.ptp(offset) == 0)


def _guide_fits(y, model, theta, offset, covariates):
    """The guide fit of a labeling (fit_guides) and its closed form (None unless _grouped)."""
    exact = partial(fit_guides, y, model=model, theta=theta, offset=offset, covariates=covariates)
    return exact, partial(fit_nb_grouped, y, theta=theta) if model == "nb" and _grouped(offset, covariates) else None


def _refitter(score, exact, fast, observed):
    """A function that gives the statistic of a labeling.

    score(fit) gives (statistic, whether PETRO fell back to the median). The function uses the
    closed-form fit when there is one. It uses the exact fit instead when the closed-form fit
    fails or PETRO falls back to the median. It also uses the exact fit when the statistic nearly
    ties the observed one. The two fits agree up to rounding, and rounding could decide such draws.
    """
    def statistic_of(labels):
        if fast is not None:
            try:
                t, median = score(fast(labels))
                if not median and np.isfinite(t) and abs(t - observed) > 1e-8 * max(1.0, abs(observed)):
                    return t
            except (ValueError, ArithmeticError):  # np.linalg.LinAlgError is a ValueError
                pass
        return score(exact(labels))[0]
    return statistic_of


def _null_statistics(statistic_of, labels, blocks, guides, mask, draws, seed):
    """Statistics of the null draws for the support with the given mask.

    Each draw shuffles the labels of the controls and of the given guides within blocks. The
    draws use the random stream of this mask.
    """
    rng = np.random.default_rng([seed, 1, int(mask)])
    moving = np.isin(labels, np.r_[0, guides])
    groups = [np.flatnonzero(moving & (blocks == block)) for block in np.unique(blocks)]
    statistics = []
    for draw in range(draws):
        permuted = labels.copy()
        for group in groups:
            permuted[group] = rng.permutation(labels[group])
        try:
            statistics.append(statistic_of(permuted))
        except (ValueError, ArithmeticError, RuntimeError) as error:
            raise ValueError(f"the refit of support {mask}, draw {draw} failed: {error}") from error
    return statistics


def _pointwise(statistic_of, observed_fit, observed, columns, labels, blocks, law, draws, seed):
    """P value of the pointwise test.

    columns[j] is the guide-fit column of guide j in the observed fit.
    """
    if not np.isfinite(observed):
        raise ValueError("the observed statistic is not finite")
    supports = _supports(observed_fit, law, draws, seed)
    if not supports:
        return 1.0
    statistics = []
    for mask, rows, n in supports:
        statistics += _null_statistics(statistic_of, labels, blocks, columns[rows] + 1, mask, n, seed)
    return _pvalue(statistics, observed)


def _worst_case(exact, labels, blocks, columns, draws, seed):
    """P value of the worst-case test over the guides at the given columns of the guide fit."""
    m = len(columns)
    if m < 3:
        raise ValueError("support tests need at least three guides")
    observed_effects = exact(labels)
    pvalues = []
    for chosen in map(list, itertools.combinations(range(m), m // 2 + 1)):
        support = columns[chosen]

        def score(effects):
            return _icw_statistic(effects.gamma, effects.sampling, support), False

        observed = score(observed_effects)[0]
        if not np.isfinite(observed):
            raise ValueError("a support statistic is not finite")
        mask = sum(1 << j for j in chosen)
        statistics = _null_statistics(_refitter(score, exact, None, observed), labels, blocks, support + 1,
                                      mask, draws, seed)
        pvalues.append(_pvalue(statistics, observed))
    return max(pvalues)


def _worst_case_nb(y, labels, blocks, columns, draws, seed, theta, offset, covariates):
    """Pooled-NB worst-case test: every draw refits the support and its pooled design."""
    m = len(columns)
    if m < 3:
        raise ValueError("support tests need at least three guides")
    pvalues = []

    def score(effects):
        return effects[0] / effects[1], False

    for chosen in map(list, itertools.combinations(range(m), m // 2 + 1)):
        support = columns[chosen] + 1
        exact = partial(_fit_nb_support, y, support=support, theta=theta, offset=offset, covariates=covariates)
        fast = partial(exact, fast=True) if _grouped(offset, covariates) else None
        observed = score(exact(labels))[0]
        if not np.isfinite(observed):
            raise ValueError("a support statistic is not finite")
        mask = sum(1 << j for j in chosen)
        statistics = _null_statistics(_refitter(score, exact, fast, observed), labels, blocks, support,
                                      mask, draws, seed)
        pvalues.append(_pvalue(statistics, observed))
    return max(pvalues)


def permutation_test(y, labels, model="ols", *, theta=None, offset=None, covariates=None, blocks=None,
                     test="pointwise", draws=999, seed=0, law=2, c=C, cap=Q_MAX):
    """Two-sided permutation P value of Gamma = 0.

    y, labels (0 for controls, 1..G for guides), model, theta, offset and covariates are as for
    fit_guides. blocks gives one value per cell. The shuffles stay within blocks. test is
    "pointwise" or "worst_case". For "pointwise", draws and law are as in pointwise_test. For
    "worst_case", draws is the number of draws for each smallest support. A support is a strict
    majority of guides.
    """
    y, labels = np.asarray(y, dtype=float), check_labels(labels)
    blocks = _check(labels, blocks, test, draws)
    exact, fast = _guide_fits(y, model, theta, offset, covariates)
    columns = np.arange(labels.max())
    if test == "worst_case":
        if model == "nb":
            return _worst_case_nb(y, labels, blocks, columns, draws, seed, theta, offset, covariates)
        return _worst_case(exact, labels, blocks, columns, draws, seed)
    effects = exact(labels)
    observed_fit = fit(effects.gamma, effects.working, c=c, cap=cap)
    observed = statistic(observed_fit, effects.sampling)
    # When _grouped, the working covariance is V = a diag(1/n_g) + b 11', where n_g are the guide
    # sizes. The baseline weights sum to one, so q'Vq = a q' diag(1/n_g) q + b. Thus the baseline
    # weights depend only on the guide sizes, which the shuffles keep.
    q = observed_fit.q if _grouped(offset, covariates) else None

    def score(effects):
        result = fit(effects.gamma, effects.working, c=c, cap=cap, q=q)
        return statistic(result, effects.sampling), result.status == "median"

    return _pointwise(_refitter(score, exact, fast, observed), observed_fit, observed, columns, labels, blocks,
                      law, draws, seed)


def _alpha_fits(y, target, labels, model, theta, theta_target, offset, target_offset, covariates, eligible, c, cap):
    """The AlphaFit of an outcome labeling and its closed form (None unless NB and _grouped).

    It fits the target gene once, at the observed labels.
    """
    design, p = _design(covariates, _indicators(labels))
    n, k = design.shape

    def outcome_design(permuted):
        return _design(covariates, _indicators(permuted))[0]

    if model == "ols":
        target_fit = _ols_fit(target, design, p)

        def exact(permuted):
            outcome_fit = _ols_fit(y, outcome_design(permuted), p)
            joint = _ols_joint(outcome_fit, target_fit, n - k)
            return fit_alpha(outcome_fit[0], target_fit[0], joint, eligible=eligible, c=c, cap=cap)
        return exact, None
    if model != "nb" or theta is None or theta_target is None:
        raise ValueError("model must be 'ols', or 'nb' with theta and theta_target")
    beta, target_influences = _nb_fit(target, labels, design, slice(p, None), theta_target, target_offset)

    def paired(gamma, outcome_influences):
        joint = _hc1(np.column_stack([outcome_influences, target_influences]), k)
        return fit_alpha(gamma, beta, joint, eligible=eligible, c=c, cap=cap)

    def exact(permuted):
        return paired(*_nb_fit(y, permuted, outcome_design(permuted), slice(p, None), theta, offset))

    def fast(permuted):
        return paired(*grouped_influences(y, permuted))
    return exact, fast if _grouped(offset, covariates) else None


def permutation_test_alpha(y, target, labels, model="ols", *, eligible, theta=None, theta_target=None,
                           offset=None, target_offset=None, covariates=None, blocks=None, test="pointwise",
                           draws=999, seed=0, law=2, c=C, cap=Q_MAX):
    """Two-sided permutation P value of alpha = 0.

    y (the outcome), target, labels and covariates are as for fit_ols_pair and fit_nb_pair. For
    model "nb", so are theta, theta_target and the offsets. eligible is as for fit_alpha. blocks,
    test, draws and law are as for permutation_test.

    Under alpha = 0, the eligible guides of a support have no effect on the outcome. Thus the
    draws shuffle only the labels of the outcome. The fit of the target gene stays at the
    observed labels, because a shuffle of its labels would erase beta. Pointwise draws refit
    the outcome, rebuild the covariance of (gamma, beta) and refit alpha.

    At alpha = 0, the worst-case statistic uses the outcome alone. Thus the worst-case test here
    is the worst-case test of permutation_test for the outcome, over the eligible guides. It does
    not use target, theta_target, target_offset, c or cap.
    """
    y, target, labels = np.asarray(y, dtype=float), np.asarray(target, dtype=float), check_labels(labels)
    blocks = _check(labels, blocks, test, draws)
    if test == "worst_case":
        columns = _eligible_indices(eligible, labels.max())
        if model == "nb":
            return _worst_case_nb(y, labels, blocks, columns, draws, seed, theta, offset, covariates)
        return _worst_case(_guide_fits(y, model, theta, offset, covariates)[0], labels, blocks, columns, draws, seed)
    exact, fast = _alpha_fits(y, target, labels, model, theta, theta_target, offset, target_offset, covariates,
                              eligible, c, cap)
    observed_fit = exact(labels)
    columns = observed_fit.eligible_indices
    observed = statistic(observed_fit)

    def score(result):
        return statistic(result), result.status == "median"

    return _pointwise(_refitter(score, exact, fast, observed), observed_fit, observed, columns, labels, blocks,
                      law, draws, seed)
