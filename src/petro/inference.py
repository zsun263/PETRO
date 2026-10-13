"""PETRO tests of Gamma = null, or of alpha = null for an AlphaFit, with parametric P values.

The two tests are the pointwise test and the worst-case test. The pointwise test mixes null draws
over supports (see pointwise_test). A support is a strict majority of guides that could be the
valid guides. The module permutation.py gives permutation P values for both tests.
"""
import itertools

import numpy as np
from scipy import special, stats

from .adapters.regression import _fit_nb_support, check_labels
from .alpha import AlphaFit, _alpha_statistics, _eligible_indices
from .estimator import _bisquare, _petro, check_covariance, njit


@njit(cache=True)
def _statistics(draws, q, k, sampling, null):
    """Studentized PETRO statistic of every row of draws."""
    out = np.empty(len(draws))
    for b in range(len(draws)):
        estimate, weights, _ = _petro(draws[b], q, k)
        out[b] = (estimate - null) / np.sqrt(weights @ sampling @ weights)
    return out


def statistic(result, sampling=None, *, null=0.0):
    """(estimate - null) / sqrt(w' S w), with the final weights w held fixed.

    S is the sampling covariance. An AlphaFit carries its own S.
    """
    if isinstance(result, AlphaFit):
        if sampling is not None:
            raise ValueError("an AlphaFit carries its own sampling covariance")
        sampling = result.sampling
    else:
        sampling = check_covariance(sampling, len(result.gamma), "sampling")
    return (result.estimate - null) / np.sqrt(result.weights @ sampling @ result.weights)


def _icw_statistic(values, covariance, support):
    """ICW statistic of the guides in support: 1'P v / sqrt(1'P 1).

    P is the inverse of the covariance of these guides.
    """
    precision = np.linalg.solve(covariance[np.ix_(support, support)], np.ones(len(support)))
    return precision @ values[support] / np.sqrt(precision.sum())


def _check_draws(draws):
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")


def _supports(result, law, draws, seed):
    """(mask, members, draws) for each support that gets draws, with weights as in pointwise_test.

    The result is empty when every weight is zero.
    """
    _check_draws(draws)
    m = len(result.gamma)
    if m < 3:
        raise ValueError("support tests need at least three guides")
    if law not in (1, 2):
        raise ValueError("law must be 1 or 2")
    masks = np.array([mask for mask in range(1 << m) if 2 * mask.bit_count() > m])
    inside = (masks[:, None] >> np.arange(m)) & 1 == 1
    a = _bisquare((result.gamma - result.estimate) / result.scales)
    with np.errstate(divide="ignore"):
        log_r = law * np.log(np.where(inside, a, 1 - a)).sum(axis=1)
    if np.isneginf(log_r).all():
        return []
    chosen = np.random.default_rng([seed, 0]).choice(len(masks), size=draws, p=special.softmax(log_r))
    counts = np.bincount(chosen, minlength=len(masks))
    return [(mask, rows, n) for mask, rows, n in zip(masks, inside, counts) if n]


def _pvalue(statistics, observed):
    """Two-sided Monte Carlo P value, as in scipy.stats.monte_carlo_test.

    It is twice the smaller inclusive tail proportion, (1 + count) / (B + 1), capped at one.
    """
    statistics = np.asarray(statistics)
    if not np.isfinite(statistics).all():
        raise ValueError("a null draw gave a nonfinite statistic")
    tolerance = 100 * np.finfo(float).eps * abs(observed)  # both tails include ties up to rounding
    lower = np.count_nonzero(statistics <= observed + tolerance)
    upper = np.count_nonzero(statistics >= observed - tolerance)
    return min(1.0, 2 * (1 + min(lower, upper)) / (len(statistics) + 1))


def pointwise_test(result, sampling=None, *, draws=999, seed=0, null=0.0, law=2):
    """Two-sided parametric pointwise P value of Gamma = null, or of alpha = null for an AlphaFit.

    sampling is the sampling covariance. An AlphaFit carries its own.

    Each support S (a strict majority of guides) gets the weight r(S)^law, with
    r(S) = prod_{g in S} a_g prod_{g not in S} (1 - a_g). Here a_g is the Tukey factor of guide g
    at the estimate. Each draw picks a support with a probability proportional to its weight.
    When every weight is zero, P = 1.

    The draws for support S hold gamma_S = null. For alpha, they hold gamma_S - null * beta_S = 0.
    The draws keep the part of the data that is independent of these contrasts at its observed
    value. Every draw refits PETRO. For alpha, each draw first rebuilds its ratios, median ratio,
    working covariance and baseline weights.
    """
    alpha = isinstance(result, AlphaFit)
    observed = statistic(result, sampling, null=null)
    if not np.isfinite(observed):
        raise ValueError("the observed statistic is not finite")
    supports = _supports(result, law, draws, seed)
    if not supports:
        return 1.0
    m = len(result.gamma)
    effects = np.r_[result.outcome, result.beta] if alpha else result.gamma
    covariance = np.ascontiguousarray(result.joint if alpha else sampling, dtype=float)
    values, vectors = np.linalg.eigh(covariance)
    root = vectors * np.sqrt(np.clip(values, 0, None))
    contrast = np.hstack((np.eye(m), -null * np.eye(m))) if alpha else np.eye(m)
    b = 0.0 if alpha else null
    statistics = []
    for mask, rows, n in supports:
        restriction = contrast[rows]  # A z = b under the null
        # Let z = effects, V = covariance and the gain K = V A' (A V A')^-1. Keep (I - K A) z, the
        # part of z that is independent of A z, at its observed value. Draw A z from N(b, A V A').
        # Here root @ root.T = V.
        gain = covariance @ restriction.T @ np.linalg.inv(restriction @ covariance @ restriction.T)
        mean = effects - gain @ (restriction @ effects - b)
        normals = np.random.default_rng([seed, 1, int(mask)]).standard_normal((n, len(effects)))
        generated = mean + normals @ (gain @ restriction @ root).T
        if alpha:
            statistics.append(_alpha_statistics(generated, covariance, result.c, result.cap, float(null)))
        else:
            statistics.append(_statistics(generated, result.q, result.scales, covariance, float(null)))
    return _pvalue(np.concatenate(statistics), observed)


def worst_case_test(result, sampling=None, *, null=0.0):
    """Guide-summary worst-case test: the largest two-sided Wald P value over the smallest supports.

    A support is a strict majority of guides. Each support combines the null contrasts of its
    guides by inverse-covariance weighting (ICW). The contrasts are gamma_g - null. For an
    AlphaFit, they are gamma_g - null * beta_g, with the joint covariance of (gamma, beta).
    worst_case_test_nb refits a pooled NB model instead.
    """
    m = len(result.gamma)
    if m < 3:
        raise ValueError("support tests need at least three guides")
    if isinstance(result, AlphaFit):
        if sampling is not None:
            raise ValueError("an AlphaFit carries its own sampling covariance")
        contrast = np.hstack((np.eye(m), -null * np.eye(m)))
        values = contrast @ np.r_[result.outcome, result.beta]
        covariance = contrast @ result.joint @ contrast.T
    else:
        values, covariance = result.gamma - null, check_covariance(sampling, m, "sampling")
    supports = itertools.combinations(range(m), m // 2 + 1)
    z = [_icw_statistic(values, covariance, list(support)) for support in supports]
    if not np.isfinite(z).all():
        raise ValueError("a support statistic is not finite")
    return np.max(2 * stats.norm.sf(np.abs(z)))


def worst_case_test_nb(y, labels, *, theta, offset=None, covariates=None, eligible=None, null=0.0):
    """Largest pooled-NB Wald P value over the smallest supports of the eligible guides.

    A support here is a strict majority of the eligible guides. The inputs are as for fit_nb.
    eligible is an optional Boolean guide mask, fixed in advance. The guides of each support share
    one coefficient, and all other guides keep separate coefficients. The SE is the HC1 sandwich
    of the pooled model. The function tests Gamma = null. With null=0 and an eligibility mask, it
    also tests alpha=0 from the outcome alone. It does not test a nonzero alpha null.
    """
    y, labels = np.asarray(y, dtype=float), check_labels(labels)
    columns = np.arange(labels.max()) if eligible is None else _eligible_indices(eligible, labels.max())
    m = len(columns)
    if m < 3:
        raise ValueError("support tests need at least three guides")
    if not np.isscalar(null) or not np.isfinite(null):
        raise ValueError("null must be a finite number")
    z = []
    for chosen in itertools.combinations(range(m), m // 2 + 1):
        estimate, se = _fit_nb_support(y, labels, columns[list(chosen)] + 1, theta=theta,
                                       offset=offset, covariates=covariates)
        z.append((estimate - null) / se)
    if not np.isfinite(z).all():
        raise ValueError("a support statistic is not finite")
    return np.max(2 * stats.norm.sf(np.abs(z)))
