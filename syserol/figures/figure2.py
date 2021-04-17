"""
This creates Figure 2.
"""

import numpy as np
from statsmodels.multivariate.pca import PCA
from tensorly.cp_tensor import _validate_cp_tensor
from .common import subplotLabel, getSetup
from ..tensor import calcR2X, delete_component, perform_CMTF
from ..dataImport import functions, createCube
from matplotlib.ticker import ScalarFormatter


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))

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
    ax[0].set_xlim(0.5, np.amax(comps) + 0.5)

    ax[1].set_xscale("log", base=2)
    ax[1].plot(sizeTfac, 1.0 - CMTFR2X, ".", label="CMTF")
    ax[1].plot(sizePCA, PCAR2X, ".", label="PCA")
    ax[1].set_ylabel("Normalized Unexplained Variance")
    ax[1].set_xlabel("Size of Factorization")
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlim(2 ** 8, 2 ** 12)
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].legend()

    ## Variance explained by each component
    rr = 10
    facT = perform_CMTF(r=rr)
    fullR2X = CMTFR2X[rr-1]
    var_exp = np.zeros(rr)

    for ii in range(rr):
        facTdel = delete_component(facT, np.delete(np.arange(0, rr), ii))
        assert facTdel.rank == 1
        _validate_cp_tensor(facTdel)
        var_exp[ii] = calcR2X(tOrig, mOrig, facTdel)

        assert var_exp[ii] < fullR2X
        assert var_exp[ii] > -1.0

    comps_idx = np.arange(1, 11)
    ax[2].scatter(comps_idx, var_exp, s=10)
    ax[2].set_ylabel("Single component R2X")
    ax[2].set_xlabel("Component index")
    ax[2].set_xticks([x for x in comps_idx])
    ax[2].set_xticklabels([x for x in comps_idx])
    ax[2].set_ylim(-1, 1)
    ax[2].set_xlim(0.5, np.amax(comps_idx) + 0.5)

    # Add subplot labels
    subplotLabel(ax)

    return f
