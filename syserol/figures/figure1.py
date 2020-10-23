"""
This creates Figure 1.
"""

import numpy as np
from .common import subplotLabel, getSetup
from ..regression import function_prediction
from ..tensor import perform_CMTF
from ..dataImport import functions


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 4), (1, 2))

    comps = np.arange(1, 9)
    tensorArr = np.zeros(comps.shape)
    comp_accuracies = np.zeros(comps.shape)

    for i in comps:
        tFac, _, R2X = perform_CMTF(r=i)
        tensorArr[i - 1] = R2X
        accur = [function_prediction(tFac, function=f, evaluation="Alter")[2] for f in functions]
        comp_accuracies[i - 1] = np.mean(accur)

    ax[0].scatter(comps, tensorArr)
    ax[0].set_ylabel("R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].legend()
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.5, np.amax(comps) + 0.5)

    ax[1].scatter(comps, comp_accuracies)
    ax[1].set_ylabel("Correlation Coefficient")
    ax[1].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
