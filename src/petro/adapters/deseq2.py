"""DESeq2 guide fit: runs PyDESeq2 and returns the guide effects of every gene."""
import numpy as np
import pandas as pd

from .regression import GuideEffects, _inverse_information, check_labels


def fit_deseq2(counts, labels, *, n_cpus=1):
    """Fit DESeq2 with one indicator factor per guide and return {gene: GuideEffects}.

    counts: a samples x genes DataFrame of raw counts. labels: 0 for control samples and 1..G for
    the guides. The GuideEffects are in log2 fold changes. Sampling covariance: the inverse
    expected information at DESeq2's fitted means. Working covariance: the same with every guide
    at the gene's median effect and the intercept refitted.

    The result omits genes without a finite fit (all-zero genes, also after outlier replacement).
    It also omits genes whose intercept refit does not converge. It keeps genes with no counts in
    a group (the controls or a guide), as PyDESeq2 does. Their log fold changes are finite only
    because of DESeq2's floor on the fitted means (min_mu). This floor sets the magnitude of these
    log fold changes.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
    except ImportError as error:
        raise ImportError("fit_deseq2 needs PyDESeq2 0.4.12: pip install '.[deseq2]'") from error

    labels = check_labels(labels)
    guides = [f"g{j}" for j in range(1, labels.max() + 1)]
    metadata = pd.DataFrame({g: np.where(labels == j, "1", "0") for j, g in enumerate(guides, 1)},
                            index=counts.index)
    dds = DeseqDataSet(counts=counts, metadata=metadata, design_factors=guides, refit_cooks=True,
                       inference=DefaultInference(n_cpus=n_cpus), quiet=True)
    dds.deseq2()
    names = list(dds.obsm["design_matrix"].columns)
    if names[0] != "intercept":
        raise ValueError("PyDESeq2's design matrix should start with the intercept")
    columns = [names.index(f"{g}_1_vs_0") for g in guides]
    design = np.asarray(dds.obsm["design_matrix"], dtype=float)
    size_factors = np.asarray(dds.obsm["size_factors"], dtype=float)
    lfc = np.asarray(dds.varm["LFC"], dtype=float)  # natural log
    dispersions = np.asarray(dds.varm["dispersions"], dtype=float)
    refitted = dds.varm["refitted"]
    replaced_to_zero = set(getattr(dds, "new_all_zeroes_genes", []))
    effects = {}
    for i, gene in enumerate(dds.var_names):
        if gene in replaced_to_zero or not (np.isfinite(lfc[i]).all() and np.isfinite(dispersions[i])):
            continue
        # PyDESeq2 refits a gene with Cook's outliers on counts with the outliers replaced.
        y = dds.counts_to_refit[:, gene].X if refitted[i] else dds.X[:, i]
        y = np.asarray(y, dtype=float).ravel()
        result = _gene_effects(y, lfc[i], dispersions[i], design, size_factors, dds.min_mu, columns)
        if result is not None:
            effects[gene] = result
    return effects


def _gene_effects(y, coefficients, dispersion, design, size_factors, min_mu, columns):
    """One gene's GuideEffects (log2 scale), or None if the intercept refit does not converge.

    The GuideEffects come from the gene's counts y and DESeq2's natural-log coefficients.
    """
    mu = np.maximum(size_factors * np.exp(design @ coefficients), min_mu)
    # Refit the intercept by Newton steps, with every guide at the median effect. Floor the means at
    # min_mu, as DESeq2 does. When every mean is at that floor, the score no longer depends on the
    # intercept. The intercept then stops at its bound of -30. Evaluate the working covariance at
    # the floored means.
    offset = np.log(size_factors) + design[:, columns].sum(axis=1) * np.median(coefficients[columns])

    def common_means(intercept):
        return np.maximum(np.exp(np.clip(offset + intercept, -30, 30)), min_mu)

    intercept = coefficients[0]
    for _ in range(500):
        common = common_means(intercept)
        step = (np.sum((y - common) / (1 + dispersion * common))
                / np.sum(common * (1 + dispersion * y) / (1 + dispersion * common) ** 2))
        intercept, previous = np.clip(intercept + np.clip(step, -2, 2), -30, 30), intercept
        if abs(intercept - previous) <= 1e-8 * (1 + abs(intercept)):
            break
    else:
        return None
    block = np.ix_(columns, columns)
    sampling = _inverse_information(design, mu, dispersion)[block] / np.log(2) ** 2
    working = _inverse_information(design, common_means(intercept), dispersion)[block] / np.log(2) ** 2
    return GuideEffects(coefficients[columns] / np.log(2), working, sampling)
