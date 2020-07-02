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
    comps = np.arange(1, 8)

    arr = []
    for i in comps:
        factors = perform_decomposition(cube, i)
        arr.append(find_R2X(cube, factors))

    ax[0].plot(comps, arr)
    ax[0].set_ylabel("R2X")
    ax[0].set_xlabel("Number of Components")

    # Add subplot labels
    subplotLabel(ax)

    return f
