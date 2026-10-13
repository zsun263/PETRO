"""OLS and negative-binomial (NB) guide fits on cells, paired fits for alpha, and Pooling.

Cells have integer labels: 0 for controls and 1..G for the guides of one target.
Each guide fit returns GuideEffects(gamma, working, sampling). gamma holds the guide indicator
coefficients, with the controls as the reference. working and sampling are the working covariance
and the sampling covariance of gamma (see petro).
Paired fits return JointEffects(gamma, beta, joint) for fit_alpha.
"""
from typing import NamedTuple

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
import statsmodels.api as sm


class GuideEffects(NamedTuple):
    gamma: np.ndarray
    working: np.ndarray
    sampling: np.ndarray


class JointEffects(NamedTuple):
    """Guide effects on the outcome (gamma) and on the target gene (beta), and their covariance.

    joint is the covariance of (gamma_1..gamma_m, beta_1..beta_m).
    """
    gamma: np.ndarray
    beta: np.ndarray
    joint: np.ndarray


def check_labels(labels):
    """Check the labels and return them as an integer array.

    The labels are 0 for controls and 1..G for the guides. Each label must be present.
    """
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0
            or np.any(np.bincount(labels) == 0)):
        raise ValueError("labels must be integers: 0 for controls and 1..G for the guides, each present")
    return labels


def _indicators(labels):
    return (labels[:, None] == np.arange(1, labels.max() + 1)).astype(float)


def _design(covariates, columns):
    """Return the design and the number of nuisance columns.

    The design has an intercept, the covariates and the given columns. Without covariates, the
    design has full rank by construction, because every group has cells.
    """
    ones = np.ones((len(columns), 1))
    nuisance = ones if covariates is None else np.column_stack([ones, covariates])
    design = np.column_stack([nuisance, columns])
    if len(design) <= design.shape[1]:
        raise ValueError("the fit needs more cells than design columns")
    if covariates is not None and np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("the design (intercept, covariates and guide indicators) is not of full rank")
    return design, nuisance.shape[1]


def _inverse_information(design, mu, dispersion):
    """Inverse expected information of a log-link NB fit with variance mu + dispersion * mu^2."""
    return np.linalg.inv(design.T @ (design * (mu / (1 + dispersion * mu))[:, None]))


def _check_counts(totals):
    """A group without counts has no finite NB log ratio. The fit would drift toward -infinity."""
    if np.any(totals == 0):
        raise ValueError("a group (controls or a guide) has no counts; its NB effect has no finite estimate")


def _glm(y, design, theta, offset):
    # Judge convergence on the coefficients, because the deviance can oscillate at rounding level.
    result = sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=1 / theta),
                    offset=offset).fit(tol=1e-12, tol_criterion="params")
    if not result.converged:
        raise RuntimeError("the NB fit did not converge")
    return result


def _nb_fit(y, groups, design, columns, theta, offset):
    """Fixed-theta NB2 fit of one response: the coefficients in columns and each cell's influence.

    The influence of a cell on these coefficients is its score times the inverse expected
    information. The fit checks the NB inputs: nonnegative integer counts, some counts in every
    group, and a positive size theta.
    """
    y = np.asarray(y, dtype=float)
    if not (np.isfinite(y).all() and np.all(y >= 0) and np.all(y == np.floor(y))):
        raise ValueError("NB responses must be nonnegative integer counts")
    if not (np.isscalar(theta) and np.isfinite(theta) and theta > 0):
        raise ValueError("theta must be a positive number")
    _check_counts(np.bincount(groups, weights=y))
    fitted = _glm(y, design, theta, offset)
    mu = fitted.fittedvalues
    bread = design.T @ (design * (mu / (1 + mu / theta))[:, None])
    scores = design * ((y - mu) / (1 + mu / theta))[:, None]
    factor = cho_factor((bread + bread.T) / 2, lower=True)
    return fitted.params[columns], scores @ cho_solve(factor, np.eye(design.shape[1])[:, columns])


def _hc1(influences, k):
    """HC1 sandwich covariance from the cells' influences: n / (n - k) I'I, for k design columns."""
    n = len(influences)
    return n / (n - k) * (influences.T @ influences)


def fit_ols(y, labels, *, covariates=None):
    """Joint OLS on guide indicators. Both covariances are the usual OLS covariance."""
    y, labels = np.asarray(y, dtype=float), check_labels(labels)
    design, p = _design(covariates, _indicators(labels))
    result = sm.OLS(y, design).fit()
    covariance = result.cov_params()[p:, p:]
    return GuideEffects(result.params[p:], covariance, covariance)


def fit_nb(y, labels, *, theta, offset=None, covariates=None):
    """Joint negative-binomial (NB2, log link) fit with a fixed size theta.

    Estimate theta beforehand, for example from the controls. The effects are natural-log ratios.
    The sampling covariance is the HC1 sandwich with the expected information. The working
    covariance is the inverse expected information with every guide at the median guide effect
    and the nuisance refitted.
    """
    y, labels = np.asarray(y, dtype=float), check_labels(labels)
    design, p = _design(covariates, _indicators(labels))
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    gamma, influences = _nb_fit(y, labels, design, slice(p, None), theta, offset)
    common = _glm(y, design[:, :p], theta, offset + (labels > 0) * np.median(gamma))
    working = _inverse_information(design, common.fittedvalues, 1 / theta)[p:, p:]
    return GuideEffects(gamma, working, _hc1(influences, design.shape[1]))


def fit_guides(y, labels, model="ols", *, theta=None, offset=None, covariates=None):
    """Run fit_ols (model "ols") or fit_nb (model "nb", which needs theta).

    offset applies to NB only.
    """
    if model == "ols":
        return fit_ols(y, labels, covariates=covariates)
    if model == "nb" and theta is not None:
        return fit_nb(y, labels, theta=theta, offset=offset, covariates=covariates)
    raise ValueError("model must be 'ols', or 'nb' with theta")


def fit_pooled(y, labels, model="ols", *, theta=None, offset=None, covariates=None):
    """Pooling: all guides share one indicator. Returns (estimate, standard error, P value).

    OLS uses the t test. NB uses an HC1 Wald test.
    """
    if model not in ("ols", "nb") or (model == "nb" and theta is None):
        raise ValueError("model must be 'ols', or 'nb' with theta")
    y, labels = np.asarray(y, dtype=float), check_labels(labels)
    design, _ = _design(covariates, labels > 0)
    if model == "ols":
        result = sm.OLS(y, design).fit()
        return result.params[-1], result.bse[-1], result.pvalues[-1]
    (estimate,), influence = _nb_fit(y, labels > 0, design, [-1], theta, offset)
    se = np.sqrt(_hc1(influence, design.shape[1])[0, 0])
    return estimate, se, 2 * stats.norm.sf(abs(estimate / se))


def _fit_nb_support(y, labels, support, *, theta, offset=None, covariates=None, fast=False):
    """NB coefficient and HC1 SE, with the guide labels in support pooled and the others separate.

    fast uses group sums after the observed exact fit has checked the inputs. fast applies only
    without covariates and with a constant offset. HC1 uses the number of columns of the pooled
    design.
    """
    outside = np.setdiff1d(np.arange(1, labels.max() + 1), support)
    mapping = np.zeros(labels.max() + 1, dtype=int)
    mapping[outside] = np.arange(1, len(outside) + 1)
    mapping[support] = len(outside) + 1
    groups = mapping[labels]
    if fast:
        if covariates is not None or (offset is not None and np.ptp(offset) != 0):
            raise ValueError("grouped NB needs no covariates and a constant offset")
        counts, totals = np.bincount(groups), np.bincount(groups, weights=y)
        _check_counts(totals)
        if len(y) <= len(counts):
            raise ValueError("the fit needs more cells than design columns")
        means = totals / counts
        variance = np.bincount(groups, weights=((y - means[groups]) / totals[groups]) ** 2)
        se = np.sqrt(len(y) / (len(y) - len(counts)) * (variance[0] + variance[-1]))
        return np.log(means[-1]) - np.log(means[0]), se
    design, _ = _design(covariates, _indicators(groups))
    (estimate,), influence = _nb_fit(y, groups, design, [-1], theta, offset)
    return estimate, np.sqrt(_hc1(influence, design.shape[1])[0, 0])


def _ols_fit(y, design, p):
    """OLS of one response: the guide coefficients, the residuals and G.

    The guide coefficients are those of columns p onward. G holds the guide rows of (X'X)^-1 X', so
    the guide coefficients are G y.
    """
    factor = cho_factor(design.T @ design, lower=True)
    coefficients = cho_solve(factor, design.T @ y)
    return coefficients[p:], y - design @ coefficients, cho_solve(factor, design.T)[p:]


def _ols_joint(outcome, target, df):
    """Covariance of (gamma, beta) from two _ols_fit results whose designs may differ.

    The designs may differ because alpha's permutation draws shuffle only the outcome's labels.
    The covariance has blocks s_ab G_a G_b', where s is the 2 x 2 residual covariance. With one
    design, it equals the covariance from fit_ols_pair.
    """
    (_, residuals_y, g_y), (_, residuals_t, g_t) = outcome, target
    residuals = np.column_stack([residuals_y, residuals_t])
    s = residuals.T @ residuals / df
    cross = s[0, 1] * (g_y @ g_t.T)
    return np.block([[s[0, 0] * (g_y @ g_y.T), cross], [cross.T, s[1, 1] * (g_t @ g_t.T)]])


def fit_ols_pair(y, target, labels, *, covariates=None):
    """Joint OLS of the outcome y and the target gene's expression on guide indicators.

    The joint covariance is the Kronecker product of the 2 x 2 residual covariance and the guide
    block of (X'X)^-1.
    """
    responses = np.column_stack([np.asarray(y, dtype=float), np.asarray(target, dtype=float)])
    labels = check_labels(labels)
    design, p = _design(covariates, _indicators(labels))
    factor = cho_factor(design.T @ design, lower=True)
    coefficients = cho_solve(factor, design.T @ responses)
    residuals = responses - design @ coefficients
    residual_covariance = residuals.T @ residuals / (len(design) - design.shape[1])
    geometry = cho_solve(factor, np.eye(design.shape[1])[:, p:])[p:]
    return JointEffects(coefficients[p:, 0], coefficients[p:, 1], np.kron(residual_covariance, geometry))


def fit_nb_pair(y, target, labels, *, theta, theta_target, offset=None, target_offset=None, covariates=None):
    """Joint NB fits (fixed sizes theta and theta_target) of the outcome and target-gene counts.

    The effects are natural-log ratios. The joint covariance is the HC1 sandwich of the stacked
    score influences of the two fits. Thus it keeps the outcome-target covariance within cells.
    """
    labels = check_labels(labels)
    design, p = _design(covariates, _indicators(labels))
    gamma, outcome_influences = _nb_fit(y, labels, design, slice(p, None), theta, offset)
    beta, target_influences = _nb_fit(target, labels, design, slice(p, None), theta_target, target_offset)
    return JointEffects(gamma, beta, _hc1(np.column_stack([outcome_influences, target_influences]), design.shape[1]))
