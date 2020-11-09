"""
This creates Figure S3.
"""
import numpy as np
from ..dataImport import createCube
from ..tensor import perform_CMTF
from .common import getSetup, subplotLabel

def makeFigure():
    """ Plot Reconstruction Error for Varying Components"""
    Cube, glyCube = createCube()
    ax, f = getSetup((7, 4), (1, 1))

    r2xvals = np.zeros(16)
    comps = np.arange(1,17)
    for i in comps:
        _, _, R2X = perform_CMTF(Cube, glyCube, i)
        r2xvals[i - 1] = R2X
    
    ax[0].scatter(comps, r2xvals)
    ax[0].set_ylabel("R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.5, np.amax() + 0.5)

    subplotLabel(ax)

    return f