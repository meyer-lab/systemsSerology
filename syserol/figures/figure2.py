"""
This creates Figure 2.
"""

import numpy as np
from statsmodels.multivariate.pca import PCA
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF
from ..dataImport import functions, createCube
from matplotlib.ticker import ScalarFormatter


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))

    comps = np.arange(1, 12)
    CMTFR2X = np.zeros(comps.shape)
    PCAR2X = np.zeros(comps.shape)

    tOrig, mOrig = createCube()

    tMat = np.reshape(tOrig, (181, -1))
    tMat = tMat[:, ~np.all(np.isnan(tMat), axis=0)]
    tMat = np.hstack((tMat, mOrig))

    sizePCA = comps * np.sum(tMat.shape)
    sizeTfac = comps * (np.sum(tOrig.shape) + mOrig.shape[1])

    for i, cc in enumerate(comps):
        outt = PCA(tMat, ncomp=cc, missing="fill-em", standardize=False, demean=False, normalize=False)
        recon = outt.scores @ outt.loadings.T
        PCAR2X[i] = np.nanvar(tMat - recon) / np.nanvar(tMat)

        CMTFR2X[i] = perform_CMTF(r=cc).R2X

    ax[0].scatter(comps, CMTFR2X, s=10)
    ax[0].set_ylabel("CMTF R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.0, np.amax(comps) + 0.5)

    ax[1].set_xscale("log", base=2)
    ax[1].plot(sizeTfac, 1.0 - CMTFR2X, ".", label="CMTF")
    ax[1].plot(sizePCA, PCAR2X, ".", label="PCA")
    ax[1].set_ylabel("Normalized Unexplained Variance")
    ax[1].set_xlabel("Size of Factorization")
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlim(2 ** 8, 2 ** 12)
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].legend()

    # Add subplot labels
    subplotLabel(ax)

    return f
