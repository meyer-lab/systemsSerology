"""
This creates Figure 1.
"""

import numpy as np
from .common import subplotLabel, getSetup
from ..dataImport import createCube
from ..tensor import perform_CMTF


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))

    cube, glyCube = createCube()
    comps = np.arange(1, 13)
    tensorArr = np.zeros(12)
    matrixArr = np.zeros(12)
    for i in comps:
        _, _, tensorR2X, matrixR2X = perform_CMTF(cube, glyCube, i)
        tensorArr[i-1] = tensorR2X
        matrixArr[i-1] = matrixR2X

    ax[0].plot(comps, tensorArr, label="Tensor R2X")
    ax[0].plot(comps, matrixArr, label="Matrix R2X")
    ax[0].set_ylabel("R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].legend()
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0, np.amax(comps))

    # Add subplot labels
    subplotLabel(ax)

    return f
