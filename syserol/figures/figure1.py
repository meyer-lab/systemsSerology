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

    comps = np.arange(1, 11)
    tensorArr = np.empty(comps.size)
    for i, compp in enumerate(comps):
        _, _, tensorR2X = perform_CMTF(r=compp)
        tensorArr[i] = tensorR2X

    ax[0].scatter(comps, tensorArr)
    ax[0].set_ylabel("R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].legend()
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.5, np.amax(comps) + 0.5)

    # Add subplot labels
    subplotLabel(ax)

    return f
