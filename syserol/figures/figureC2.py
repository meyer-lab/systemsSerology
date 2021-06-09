""" This makes Figure 1. Plot of R2X values"""
import numpy as np
from ..COVID import Tensor4D
from ..tensor import perform_CMTF
from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((5, 4), (1, 1))
    comps = np.arange(1, 11)

    tensor, _ = Tensor4D()
    R2X = [perform_CMTF(tensor, r=cc).R2X for cc in comps]

    ax[0].scatter(comps, R2X, color="b")
    ax[0].set_ylabel("R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.0, np.amax(comps) + 0.5)

    subplotLabel(ax)

    return f
