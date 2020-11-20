"""
This creates Figure S3.
"""
import numpy as np
from ..impute import evaluate_missing
from .common import getSetup, subplotLabel

def makeFigure():
    """ Plot Reconstruction Error for Varying Components"""
    ax, f = getSetup((7, 4), (1, 1))

    R2X = evaluate_missing()
    ax[0].scatter(np.arange(1, R2X.size + 1), R2X)
    ax[0].set_ylabel("Q2X of Imputation")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_ylim(0, 1)

    subplotLabel(ax)

    return f