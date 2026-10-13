"""Guide-level fits that give PETRO its inputs, GuideEffects(gamma, working, sampling).

- regression.py: OLS and NB fits on cells, paired fits for alpha, and Pooling.
- deseq2.py: DESeq2 (PyDESeq2) fit on pseudobulk samples.
- edger.R: edgeR quasi-likelihood fit on pseudobulk samples. edger.py reads the result.
"""
from .deseq2 import fit_deseq2
from .edger import read_edger
from .regression import (GuideEffects, JointEffects, fit_guides, fit_nb, fit_nb_pair, fit_ols, fit_ols_pair,
                         fit_pooled)
