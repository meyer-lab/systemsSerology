"""
This creates Figure 2.
"""

import numpy as np
from statsmodels.multivariate.pca import PCA
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF, calcR2X, tensor_degFreedom
from ..dataImport import createCube
from ..impute import flatten_to_mat
from matplotlib.ticker import ScalarFormatter


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))

    comps = np.arange(1, 12)
    CMTFR2X = np.zeros(comps.shape)
    PCAR2X = np.zeros(comps.shape)
    sizeTfac = np.zeros(comps.shape)

    tOrig, mOrig = createCube()
    tMat = flatten_to_mat(tOrig, mOrig)

    sizePCA = comps * np.sum(tMat.shape)

    for i, cc in enumerate(comps):
        outt = PCA(tMat, ncomp=cc, missing="fill-em", standardize=False, demean=False, normalize=False)
        recon = outt.scores @ outt.loadings.T
        PCAR2X[i] = calcR2X(recon, mIn=tMat)
        tFac = perform_CMTF(r=cc)
        CMTFR2X[i] = tFac.R2X
        sizeTfac[i] = tensor_degFreedom(tFac)

    ax[0].scatter(comps, CMTFR2X, s=10)
    ax[0].set_ylabel("CMTF R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.5, np.amax(comps) + 0.5)

    ax[1].set_xscale("log", base=2)
    ax[1].plot(sizeTfac, 1.0 - CMTFR2X, ".", label="CMTF")
    ax[1].plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax[1].set_ylabel("Normalized Unexplained Variance")
    ax[1].set_xlabel("Size of Reduced Data")
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlim(2 ** 8, 2 ** 12)
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].legend()

    # Scaling matrix
    rats = np.arange(-8, 9, step=0.25)
    tOrig, mOrig = createCube()
    totalR2X = np.zeros(rats.shape)
    CMTFR2X = np.zeros(rats.shape)
    PCAR2X = np.zeros(rats.shape)
    for ii, rat in enumerate(rats):
        mScaled = mOrig * (2.0 ** rat)
        tFac = perform_CMTF(tOrig=tOrig, mOrig=mScaled, r=5)
        totalR2X[ii] = calcR2X(tFac, tOrig, mScaled)
        CMTFR2X[ii] = calcR2X(tFac, tIn=tOrig)
        PCAR2X[ii] = calcR2X(tFac, mIn=mScaled)

    ax[2].plot(rats, totalR2X, label="Total")
    ax[2].plot(rats, PCAR2X, label="Matrix")
    ax[2].plot(rats, CMTFR2X, label="Tensor")
    ax[2].set_ylabel("R2X")
    ax[2].set_xlabel("Matrix scaled")

    def rat2frac(rat):
        if rat >= 0:
            return str(2 ** rat)
        else:
            return '1/' + rat2frac(-rat)
    ax[2].set_xlim(-7.5, 7.5)
    # ax[2].set_ylim(0.8, 1.0)
    ax[2].set_xticks(rats[::8])
    ax[2].set_xticklabels([rat2frac(r) for r in rats[::8]])
    ax[2].legend()

    # Add subplot labels
    subplotLabel(ax)

    return f
