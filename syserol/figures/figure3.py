"""
This creates Figure 3.
"""

import numpy as np
from .common import subplotLabel, getSetup
from ..impute import evaluate_missing, increase_missing


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))

    comps = np.arange(1, 2)

    Q2Xchord, _, _ = evaluate_missing(comps, 15, chords=True)
    ax[0].scatter(comps, Q2Xchord)
    ax[0].set_ylabel("Q2X of Imputation")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)

    CMTFR2X, PCAR2X, _ = evaluate_missing(comps, 15, chords=False, PCAcompare=True)
    ax[1].plot(comps, CMTFR2X, ".", label="CMTF")
    ax[1].plot(comps, PCAR2X, ".", label="PCA")
    ax[1].set_ylabel("Q2X of Imputation")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_xticks([x for x in comps])
    ax[1].set_xticklabels([x for x in comps])
    ax[1].set_ylim(0, 1)
    ax[1].legend()

    CMTFR2X, PCAR2X, missing = increase_missing(comps, PCAcompare=False)
    ax[2].plot(missing, CMTFR2X, ".", label="CMTF")
    ax[2].plot(missing, PCAR2X, ".", label="PCA")
    ax[2].set_ylabel("Q2X of Imputation")
    ax[2].set_xlabel("Fraction Missing")
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
