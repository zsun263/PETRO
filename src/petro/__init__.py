"""PETRO: robust aggregation of guide-level effects into one target-level effect.

Model: the effect of guide g is gamma_g = Gamma + delta_g, where Gamma is the on-target
perturbation effect and delta_g is the off-target perturbation effect of guide g. For a strict
majority of guides, delta_g = 0. The inputs are the estimated guide effects and two covariances
from a guide-level fit (see petro.adapters):

- The working covariance sets the baseline weights and the Tukey scales.
- The sampling covariance studentizes the estimate and generates the null draws.

fit_alpha estimates the regulatory effect alpha instead: gamma_g = alpha * beta_g + delta_g.
It uses paired guide effects on the outcome and on the target gene (see petro.alpha).

The pointwise test and the worst-case test give parametric P values (pointwise_test,
worst_case_test, worst_case_test_nb). From OLS and NB fits on cells, they also give permutation
P values: permutation_test for Gamma and permutation_test_alpha for alpha.
"""
from .alpha import AlphaFit, fit_alpha
from .estimator import C, Q_MAX, Fit, capped_icw_weights, fit
from .inference import pointwise_test, statistic, worst_case_test, worst_case_test_nb
from .permutation import permutation_test, permutation_test_alpha
