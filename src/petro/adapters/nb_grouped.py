"""Closed-form NB guide fits for a design with only an intercept and guide indicators.

Without covariates and with a constant offset or no offset, these fits give the same results as
fit_nb up to rounding. They use only group sums. The permutation tests use them to refit draws
quickly.
"""
import numpy as np

from .regression import GuideEffects, _check_counts, _indicators


def fit_nb_grouped(y, labels, theta):
    """NB effects (natural-log ratios) and covariances from group sums.

    The labels are as for fit_nb. The NB maximum-likelihood group means are the sample means.
    The sampling covariance is the HC1 sandwich. The working covariance is the inverse expected
    information with every guide at the median effect, where the control mean u solves a
    quadratic.
    """
    counts = np.bincount(labels).astype(float)
    totals = np.bincount(labels, weights=y)
    _check_counts(totals)
    means = totals / counts
    gamma = np.log(means[1:]) - np.log(means[0])
    # Sandwich variance of log(mean) in group g: sum over its cells of ((y - mean) / total)^2.
    variance = np.bincount(labels, weights=((y - means[labels]) / totals[labels]) ** 2)
    hc1 = len(y) / (len(y) - len(counts))
    sampling = hc1 * (np.diag(variance[1:]) + variance[0])
    # Common effect: the controls have mean u and every guide has mean u * ratio. With dispersion d,
    # the NB score equation in u, (T0 - n0 u) / (1 + d u) + (T1 - n1 ratio u) / (1 + d ratio u) = 0,
    # is the quadratic a u^2 + b u - (T0 + T1) = 0. Take the positive root in a cancellation-free
    # form.
    dispersion, ratio = 1 / theta, np.exp(np.median(gamma))
    n0, n1, t0, t1 = counts[0], counts[1:].sum(), totals[0], totals[1:].sum()
    a = dispersion * ratio * (n0 + n1)
    b = n0 + n1 * ratio - dispersion * (ratio * t0 + t1)
    root = np.sqrt(b * b + 4 * a * (t0 + t1))
    u = 2 * (t0 + t1) / (b + root) if b >= 0 else (root - b) / (2 * a)
    working = np.diag((1 / (u * ratio) + dispersion) / counts[1:]) + (1 / u + dispersion) / counts[0]
    return GuideEffects(gamma, working, sampling)


def grouped_influences(y, labels):
    """NB effects and each cell's influence on them, from group sums.

    For this design, the influences equal those of fit_nb: the score times the inverse expected
    information. A cell of guide g has influence (y - mean_g) / total_g on the effect of guide g.
    A control cell has influence -(y - mean_0) / total_0 on every effect. Neither the effects nor
    the influences depend on theta.
    """
    counts = np.bincount(labels).astype(float)
    totals = np.bincount(labels, weights=y)
    _check_counts(totals)
    means = totals / counts
    scaled = (y - means[labels]) / totals[labels]
    return np.log(means[1:]) - np.log(means[0]), (_indicators(labels) - (labels == 0)[:, None]) * scaled[:, None]
