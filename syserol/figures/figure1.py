"""
This creates Figure 1.
"""

import numpy as np
from .common import subplotLabel, getSetup
from ..impute import evaluate_missing


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((3, 3), (1, 1))

    comps = np.arange(1, 12)

    Q2X = evaluate_missing(comps)
    ax[0].scatter(comps, Q2X)
    ax[0].set_ylabel("Q2X of Imputation")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
