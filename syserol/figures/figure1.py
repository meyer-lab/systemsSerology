"""
This creates Figure 1.
"""

import numpy as np
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((3, 3), (1, 1))

    comps = np.arange(1, 10)
    tensorArr = np.empty(comps.size)
    matrixArr = np.empty(comps.size)
    for i, compp in enumerate(comps):
        _, _, tensorR2X, matrixR2X = perform_CMTF(r=compp)
        tensorArr[i-1] = tensorR2X
        matrixArr[i-1] = matrixR2X

    ax[0].plot(comps, tensorArr, label="Tensor R2X")
    ax[0].plot(comps, matrixArr, label="Matrix R2X")
    ax[0].set_ylabel("R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].legend()
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(1, np.amax(comps))

    # Add subplot labels
    subplotLabel(ax)

    return f
