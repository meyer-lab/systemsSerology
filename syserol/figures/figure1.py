"""
This creates Figure 1.
"""

import numpy as np
from .common import subplotLabel, getSetup
from ..dataImport import createCube
from ..tensor import perform_decomposition, find_R2X


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))

    cube, _ = createCube()
    comps = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    arr = [find_R2X(cube, perform_decomposition(cube, i)) for i in comps]

    ax[0].plot(comps, arr)
    ax[0].set_ylabel("R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
