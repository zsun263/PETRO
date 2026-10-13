"""PETRO for the regulatory effect alpha: gamma_g = alpha * beta_g + delta_g.

gamma_g is the effect of guide g on the outcome. beta_g is its effect on the target gene's own
expression. PETRO aggregates the ratios gamma_g / beta_g with the same estimator as for the
on-target perturbation effect Gamma.
"""
from dataclasses import dataclass

import numpy as np

from .estimator import C, Q_MAX, Fit, _capped_icw_weights, _petro, _scales, check_covariance, fit, njit


@dataclass
class AlphaFit(Fit):
    """A Fit on the ratios gamma_g / beta_g of the eligible guides.

    Its gamma field holds the ratios. An AlphaFit also holds the inputs that the tests refit.
    """
    outcome: np.ndarray           # effects of the eligible guides on the outcome
    beta: np.ndarray
    joint: np.ndarray             # covariance of (gamma, beta) of the eligible guides
    sampling: np.ndarray          # delta-method covariance of the ratios
    cap: float
    eligible_indices: np.ndarray  # positions of the eligible guides in the inputs


@njit(cache=True)
def _ratio_inputs(outcome, beta, joint):
    """Ratios and their working covariance at the median ratio rho.

    The working covariance is D^-1 (V_gg + rho^2 V_bb - rho (V_gb + V_bg)) D^-1, with V = joint
    and D = diag(beta).
    """
    if np.any(beta == 0):
        raise ValueError("on-target effects must be nonzero")
    m = len(beta)
    ratios = outcome / beta
    rho = np.median(ratios)
    cross = joint[:m, m:]
    structural = joint[:m, :m] + rho**2 * joint[m:, m:] - rho * (cross + cross.T)
    inverse = 1 / beta
    return ratios, inverse[:, None] * structural * inverse


def _eligible_indices(eligible, m):
    """Positions of the eligible guides, from a Boolean mask with one entry per guide."""
    eligible = np.asarray(eligible)
    if eligible.shape != (m,) or not np.isin(eligible, [0, 1]).all() or not eligible.any():
        raise ValueError("eligible must be a Boolean mask with one entry per guide, selecting at least one")
    return np.flatnonzero(eligible)


def fit_alpha(gamma, beta, joint, *, eligible, c=C, cap=Q_MAX):
    """PETRO estimate of alpha from guide effects on the outcome (gamma) and on the target (beta).

    joint is the covariance of (gamma_1..gamma_m, beta_1..beta_m), including the cross-covariances.
    eligible is a Boolean mask of the guides to aggregate, fixed in advance. For example, it can
    select the guides with a clear on-target effect. Each eligible guide must have a nonzero beta.
    """
    gamma, beta = np.asarray(gamma, dtype=float), np.asarray(beta, dtype=float)
    m = len(gamma)
    if gamma.ndim != 1 or beta.shape != (m,):
        raise ValueError("gamma and beta need one entry per guide")
    if not (np.isfinite(gamma).all() and np.isfinite(beta).all()):
        raise ValueError("gamma and beta must be finite")
    selected = _eligible_indices(eligible, m)
    joint = check_covariance(joint, 2 * m, "joint")
    rows = np.r_[selected, m + selected]
    gamma, beta, joint = gamma[selected], beta[selected], (joint + joint.T)[np.ix_(rows, rows)] / 2
    ratios, working = _ratio_inputs(gamma, beta, joint)
    result = fit(ratios, working, c=c, cap=cap)
    jacobian = np.hstack((np.diag(1 / beta), np.diag(-gamma / beta**2)))
    return AlphaFit(**vars(result), outcome=gamma, beta=beta, joint=joint,
                    sampling=jacobian @ joint @ jacobian.T, cap=cap, eligible_indices=selected)


@njit(cache=True)
def _alpha_statistics(draws, joint, c, cap, null):
    """Studentized alpha statistic of every row of draws (gamma, beta).

    For each draw, the function rebuilds the ratios, the median ratio, the working covariance and
    the baseline weights. It computes w' S w as (J' w)' V (J' w), with J the Jacobian of the ratios.
    """
    m = len(joint) // 2
    out = np.empty(len(draws))
    for b in range(len(draws)):
        outcome, beta = draws[b, :m], draws[b, m:]
        ratios, working = _ratio_inputs(outcome, beta, joint)
        if np.linalg.eigvalsh(working).min() <= 0:
            raise ValueError("an alpha draw has a working covariance that is not positive definite")
        estimate, weights, _ = _petro(ratios, _capped_icw_weights(working, cap), _scales(working, c))
        gradient = np.concatenate((weights / beta, -weights * outcome / beta**2))
        out[b] = (estimate - null) / np.sqrt(gradient @ joint @ gradient)
    return out
