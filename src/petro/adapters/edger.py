"""Reads the guide effects that edger.R writes."""
import json

import numpy as np

from .regression import GuideEffects


def read_edger(path):
    """Read {gene: GuideEffects} from the JSON file that edger.R writes."""
    with open(path) as handle:
        genes = json.load(handle)
    return {gene: GuideEffects(*(np.array(fields[name], dtype=float) for name in GuideEffects._fields))
            for gene, fields in genes.items()}
