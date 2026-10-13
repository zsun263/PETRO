# edgeR guide-level fit that gives PETRO its inputs, gene by gene.
#
# fit_edger(y, design, guides):
#   y:      a DGEList with normalization factors.
#   design: the nuisance columns (intercept, batch, ...) and one indicator column per guide,
#           with the control samples as the reference.
#   guides: the names of the guide columns.
# For every gene, fit_edger returns the guide effects (log2 fold changes) and their two
# covariances (see petro):
#   sampling: the inverse expected information at the fit, times the quasi-likelihood scale.
#   working:  the same with every guide at the gene's median effect and the nuisance refitted.
# The result omits genes with no counts in the controls or in a guide's samples. It also omits
# genes with a non-finite effect. fit_edger uses the observation weights in y$weights.
# Write the result with jsonlite::write_json(result, path, digits = I(17)).
# Read it in Python with petro.adapters.read_edger(path).
suppressPackageStartupMessages(library(edgeR))

# Covariance of the guide coefficients X, with the nuisance coefficients Z profiled out.
guide_covariance <- function(X, Z, h) {
  xz <- crossprod(X, Z * h)
  v <- solve(crossprod(X, X * h) - xz %*% solve(crossprod(Z, Z * h), t(xz)))
  (v + t(v)) / 2
}

fit_edger <- function(y, design, guides) {
  X <- design[, guides, drop = FALSE]
  Z <- design[, setdiff(colnames(design), guides), drop = FALSE]
  y <- estimateDisp(y, design)
  fit <- glmQLFit(y, design, robust = TRUE, legacy = TRUE, prior.count = 0.125)
  gamma <- fit$unshrunk.coefficients[, guides, drop = FALSE]
  center <- apply(gamma, 1, median)
  y$offset <- as.matrix(fit$offset) + outer(ifelse(is.finite(center), center, 0), rowSums(X))
  common <- glmFit(y, Z, dispersion = fit$dispersion, prior.count = 0)
  dispersion <- rep_len(fit$dispersion, nrow(y))
  weights <- if (is.null(fit$weights)) matrix(1, nrow(y), ncol(y)) else fit$weights
  totals <- y$counts %*% cbind(rowSums(X) == 0, X)  # control and guide count totals per gene
  effects <- lapply(seq_len(nrow(y)), function(i) {
    if (any(totals[i, ] == 0) || any(!is.finite(gamma[i, ]))) return(NULL)  # no counts in a group
    h <- function(mu) weights[i, ] * mu / (1 + dispersion[i] * mu)
    scale <- fit$s2.post[i] / log(2)^2
    list(gamma = unname(gamma[i, ]) / log(2),
         working = unname(guide_covariance(X, Z, h(common$fitted.values[i, ]))) * scale,
         sampling = unname(guide_covariance(X, Z, h(fit$fitted.values[i, ]))) * scale)
  })
  names(effects) <- rownames(y)
  Filter(Negate(is.null), effects)
}
