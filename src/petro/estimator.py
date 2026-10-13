"""PETRO point estimate: capped inverse-covariance (ICW) baseline weights and the Tukey root rule.

If Numba is installed, it compiles the root finding and caches the compiled code on disk, so only
the first run compiles. Without Numba, or with NUMBA_DISABLE_JIT=1, the same code runs as plain
Python.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import pascal

try:
    from numba import njit
except ImportError:
    def njit(**options):
        return lambda function: function

C = 4.685    # Tukey bisquare cutoff, in units of each guide's standard error
Q_MAX = 0.4  # cap on any guide's baseline weight
_PASCAL = pascal(6, kind="lower", exact=False)  # binomial coefficients C(n, j)


@dataclass
class Fit:
    estimate: float
    status: str           # "petro" or "median" (fewer than three guides, no eligible root or a tie)
    weights: np.ndarray   # final weights, with estimate = weights @ gamma
    gamma: np.ndarray
    working: np.ndarray
    q: np.ndarray         # baseline weights
    c: float

    @property
    def scales(self):
        """Tukey window half-widths, c times each guide's working standard error."""
        return _scales(self.working, self.c)

    @property
    def effective_guides(self):
        """Effective number of guides, 1 / sum(w^2) of the final weights.

        It is NaN for the median fallback.
        """
        return 1 / np.sum(self.weights**2) if self.status == "petro" else np.nan

    @property
    def low_support(self):
        """Diagnostic flag for fewer than m/2 effective guides.

        It is None (not computed) for the median fallback, which the status already marks.
        """
        return bool(self.effective_guides < len(self.gamma) / 2) if self.status == "petro" else None


def capped_icw_weights(covariance, *, cap=Q_MAX):
    """Capped inverse-covariance (ICW) baseline weights.

    The weights are argmin q' V q subject to sum(q) = 1 and 0 <= q <= cap. Without a binding
    bound, they are the ICW weights V^-1 1 / 1' V^-1 1. Otherwise, the function exactly solves
    each pattern of guides at 0, at the cap or free, one at a time. The problem is strictly
    convex, so a pattern that meets the optimality conditions is the optimum. If no pattern meets
    them, the feasible solution with the smallest variance is the optimum.
    """
    return _capped_icw_weights(np.ascontiguousarray(covariance, dtype=float), float(cap))


@njit(cache=True)
def _capped_icw_weights(covariance, cap):
    m = len(covariance)
    if m * cap < 1:
        raise ValueError("cap is infeasible for the number of guides")
    q = np.linalg.solve(covariance, np.ones(m))
    q /= q.sum()
    if q.min() >= 0 and q.max() <= cap:
        return q
    v = covariance / np.abs(covariance).max()
    best, smallest = np.full(m, np.nan), np.inf
    powers = 3 ** np.arange(m - 1, -1, -1)
    for index in range(3**m):
        pattern = index // powers % 3  # 0 = free, 1 = at 0, 2 = at the cap
        if np.count_nonzero(pattern == 2) * cap > 1:
            continue  # the guides at the cap hold more than the total weight
        free = pattern == 0
        q = np.where(pattern == 2, cap, 0.0)
        level = np.nan  # Lagrange multiplier of sum(q) = 1, equal to (V q)_g on the free guides
        if free.any():
            a = np.linalg.solve(v[free][:, free], np.ones(free.sum()))
            b = np.linalg.solve(v[free][:, free], v[free][:, ~free] @ q[~free])
            level = (1 - q[~free].sum() + b.sum()) / a.sum()
            q[free] = level * a - b
        if q.min() >= -1e-12 and q.max() <= cap + 1e-12 and abs(q.sum() - 1) < 1e-12:
            gradient = v @ q  # optimal if no weight should rise from 0 or fall from the cap
            optimal = np.all(gradient[pattern == 1] >= level) and np.all(gradient[pattern == 2] <= level)
            if free.any() and optimal:
                return np.clip(q, 0, cap)
            variance = q @ v @ q
            if variance < smallest:
                best, smallest = q, variance
    return np.clip(best, 0, cap)


@njit(cache=True)
def _scales(working, c):
    return c * np.sqrt(np.diag(working))


@njit(cache=True)
def _bisquare(u):
    return np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)


@njit(cache=True)
def _score(t, y, k, q):
    """U(t) = sum_g q_g a_g (y_g - t), with the Tukey factors a_g = bisquare((y_g - t) / k_g)."""
    return np.sum(q * _bisquare((y - t) / k) * (y - t))


@njit(cache=True)
def _curvature(t, y, k, q):
    """Curvature of the Tukey loss, -dU/dt. It is positive at a strict local minimum."""
    u = (y - t) / k
    return np.sum(np.where(np.abs(u) < 1, q * (1 - u**2) * (1 - 5 * u**2), 0.0))


@njit(cache=True)
def _roots(y, k, q):
    """Real roots of U.

    Between consecutive window edges y_g -/+ k_g, U is a polynomial: the sum over the guides
    inside their windows of q_g (r - 2 r^3 / k_g^2 + r^5 / k_g^4), with r = y_g - t. The function
    solves each piece in local coordinates, t = mid + half * x with x in [-1, 1]. In these
    coordinates, every term is bounded, so guides with very different standard errors stay well
    conditioned.
    """
    edges = np.unique(np.concatenate((y - k, y + k)))
    found = []
    for i in range(len(edges) - 1):
        lower, upper = edges[i], edges[i + 1]
        mid, half = (lower + upper) / 2, (upper - lower) / 2
        inside = np.abs(y - mid) < k
        if 2 * np.count_nonzero(inside) <= len(y):
            continue  # no root here can keep more than half of the guides inside their windows
        coefficients = np.zeros(6)  # ascending powers of x
        for g in range(len(y)):
            if inside[g]:
                for n, a in ((1, 1.0), (3, -2 / k[g] ** 2), (5, 1 / k[g] ** 4)):
                    for j in range(n + 1):
                        coefficients[j] += q[g] * a * _PASCAL[n, j] * (y[g] - mid) ** (n - j) * (-half) ** j
        largest = np.max(np.abs(coefficients))
        if largest == 0:
            continue
        coefficients /= largest
        degree = 5
        while degree > 0 and abs(coefficients[degree]) < 1e-13:
            degree -= 1
        if degree == 0:
            continue
        for root in np.roots(coefficients[:degree + 1][::-1].astype(np.complex128)):
            if abs(root.imag) < 1e-9 * (1 + abs(root.real)) and abs(root.real) <= 1 + 1e-10:
                t = mid + half * min(max(root.real, -1.0), 1.0)
                for _ in range(2):  # Newton polish, kept inside the interval
                    curvature = _curvature(t, y, k, q)
                    if abs(curvature) > 1e-12:
                        t = min(max(t + _score(t, y, k, q) / curvature, lower), upper)
                found.append(t)
    roots = np.sort(np.array(found))
    keep = np.ones(len(roots), dtype=np.bool_)
    keep[1:] = np.diff(roots) > 1e-8 * np.min(k)  # a root on a shared edge is found twice
    return roots[keep]


@njit(cache=True)
def _petro(gamma, q, k):
    """Estimate, final weights and whether the median fallback was used.

    The eligible roots of U are the strict local minima of the Tukey loss that keep more than half
    of the guides inside their windows. The estimate is the eligible root nearest the median of
    the guide effects. With no eligible root, a tie or fewer than three guides, the estimate is the
    median.
    """
    m = len(gamma)
    center = np.median(gamma)
    if m >= 3:
        # Center at the median and rescale, so root finding works on numbers of order one.
        scale = max(np.max(np.abs(gamma - center)), np.max(k))
        y, k = (gamma - center) / scale, k / scale
        best, second, root = np.inf, np.inf, 0.0
        for t in _roots(y, k, q):
            if _curvature(t, y, k, q) > 1e-10 and 2 * np.count_nonzero(np.abs(y - t) < k) > m:
                if abs(t) < best:
                    best, second, root = abs(t), best, t
                elif abs(t) < second:
                    second = abs(t)
        if second - best > 1e-8 * np.min(k):
            weights = q * _bisquare((y - root) / k)
            return center + scale * root, weights / np.sum(weights), False
    middle = np.argsort(gamma, kind="mergesort")[(m - 1) // 2:m // 2 + 1]
    weights = np.zeros(m)
    weights[middle] = 1 / len(middle)
    return center, weights, True


def check_covariance(matrix, size, name, definite=False):
    """A finite, symmetric size x size covariance matrix.

    With definite=True, it is positive definite. Otherwise, it is positive semidefinite up to
    rounding.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {size} x {size} matrix")
    scale = np.abs(matrix).max()
    smallest = np.linalg.eigvalsh(matrix).min()
    if np.abs(matrix - matrix.T).max() > 1e-10 * scale or smallest <= (0 if definite else -1e-10 * scale):
        kind = "definite" if definite else "semidefinite"
        raise ValueError(f"{name} must be a symmetric positive-{kind} covariance")
    return matrix


def fit(gamma, working, *, c=C, cap=Q_MAX, q=None):
    """PETRO estimate from guide effects, their working covariance and optional baseline weights q.

    The result is a Fit with the estimate, the final weights and the status "petro" or "median".
    """
    gamma = np.asarray(gamma, dtype=float)
    m = len(gamma)
    if gamma.ndim != 1 or not np.isfinite(gamma).all():
        raise ValueError("gamma must be a finite vector")
    working = check_covariance(working, m, "working", definite=True)
    if not (c > 0 and 0 < cap <= 1):
        raise ValueError("c must be positive and cap in (0, 1]")
    if q is not None:
        q = np.asarray(q, dtype=float)
        valid = q.shape == (m,) and np.isfinite(q).all() and q.min() >= 0 and q.max() <= cap + 1e-12
        if not valid or abs(q.sum() - 1) > 1e-9:
            raise ValueError("q must be m nonnegative weights at most cap that sum to one")
    elif m >= 3:
        q = capped_icw_weights(working, cap=cap)
    else:
        q = np.full(m, np.nan)  # fewer than three guides: the median and no baseline weights
    estimate, weights, median = _petro(gamma, q, _scales(working, c))
    return Fit(estimate, "median" if median else "petro", weights, gamma, working, q, c)
