""" This makes Figure 6. Plot of R2X values"""
import numpy as np
import seaborn as sns
from ..COVID import Tensor4D, dimensionLabel4D
from ..tensor import perform_CMTF, calcR2X, tensor_degFreedom
from .common import getSetup, subplotLabel
from ..impute import flatten_to_mat
from statsmodels.multivariate.pca import PCA
from matplotlib.ticker import ScalarFormatter

def makeFigure():
    ax, f = getSetup((14, 7), (2, 4))
    comps = np.arange(1, 11)

    tensor, _ = Tensor4D()
    R2X = [perform_CMTF(tensor, r=cc).R2X for cc in comps]

    ax[0].axis("off")
    ax[1].scatter(comps, R2X, color="b")
    ax[1].set_ylabel("R2X")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_xticks([x for x in comps])
    ax[1].set_xticklabels([x for x in comps])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0.0, np.amax(comps) + 0.5)

    comps = np.arange(1, 12)
    CMTFR2X = np.zeros(comps.shape)
    PCAR2X = np.zeros(comps.shape)
    sizeTfac = np.zeros(comps.shape)

    tMat = flatten_to_mat(tensor)
    sizePCA = comps * np.sum(tMat.shape)

    for i, cc in enumerate(comps):
        outt = PCA(tMat, ncomp=cc, missing="fill-em", standardize=False, demean=False, normalize=False)
        recon = outt.scores @ outt.loadings.T
        PCAR2X[i] = calcR2X(recon, mIn=tMat)
        tFac = perform_CMTF(tOrig=tensor, r=cc)
        CMTFR2X[i] = tFac.R2X
        sizeTfac[i] = tensor_degFreedom(tFac)

    ax[2].set_xscale("log", base=2)
    ax[2].plot(sizeTfac, 1.0 - CMTFR2X, ".", label="CMTF")
    ax[2].plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax[2].set_ylabel("Normalized Unexplained Variance")
    ax[2].set_xlabel("Size of Reduced Data")
    ax[2].set_ylim(bottom=0.0)
    ax[2].set_xlim(2 ** 8, 2 ** 12)
    ax[2].xaxis.set_major_formatter(ScalarFormatter())
    ax[2].legend()





    ## Colormap

    weeklabels, Rlabels, agLabels = dimensionLabel4D()
    tfac = perform_CMTF(tensor, r=6)

    components = [str(ii + 1) for ii in range(tfac.rank)]
    comp_plot(tfac.factors[0], components, False, "Subjects", ax[4])
    comp_plot(tfac.factors[1], components, agLabels, "Antigens", ax[5])
    comp_plot(tfac.factors[2], components, Rlabels, "Receptors", ax[6])
    comp_plot(tfac.factors[3], components, weeklabels, "Weeks", ax[7])

    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax):
    """ Creates heatmap plots for each input dimension by component. """
    sns.heatmap(factors, cmap="PiYG", center=0, xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel(plotLabel)
