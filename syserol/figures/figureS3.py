"""
This creates Figure S3.
"""
import numpy as np
from ..impute import evaluate_missing
from .common import getSetup, subplotLabel

def makeFigure():
    """ Plot Reconstruction Error for Varying Components"""
    ax, f = getSetup((7, 4), (1, 1))

    Sumsqs = evaluate_missing()
    comps = np.arange(1,17)
    
    ax[0].scatter(comps, Sumsqs)
    ax[0].set_ylabel("Sum Squared Error")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.5, np.amax() + 0.5)

    subplotLabel(ax)

    return f