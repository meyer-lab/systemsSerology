""" This makes Figure 6. Plot of R2X values"""
import numpy as np
import seaborn as sns
from ..COVID import Tensor3D, dimensionLabel3D, time_components_df
from ..tensor import perform_CMTF, calcR2X, tensor_degFreedom
from .common import getSetup, subplotLabel
from ..impute import flatten_to_mat
from statsmodels.multivariate.pca import PCA
from matplotlib.ticker import ScalarFormatter


def makeFigure():
    ax, f = getSetup((14, 10), (3, 4))
    comps = np.arange(1, 9)

    tensor, _ = Tensor3D()
    CMTFfacs = [perform_CMTF(tensor, r=cc) for cc in comps]
    CMTFR2X = np.array([f.R2X for f in CMTFfacs])

    ax[0].axis("off")
    ax[1].scatter(comps, CMTFR2X, color="b")
    ax[1].set_ylabel("R2X")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_xticks([x for x in comps])
    ax[1].set_xticklabels([x for x in comps])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0.0, np.amax(comps) + 0.5)

    PCAR2X = np.zeros(comps.shape)
    sizeTfac = np.zeros(comps.shape)

    tMat = flatten_to_mat(tensor)
    sizePCA = comps * np.sum(tMat.shape)

    for i, cc in enumerate(comps):
        outt = PCA(tMat, ncomp=cc, missing="fill-em", standardize=False, demean=False, normalize=False)
        recon = outt.scores @ outt.loadings.T
        PCAR2X[i] = calcR2X(recon, mIn=tMat)
        sizeTfac[i] = tensor_degFreedom(CMTFfacs[i])

    ax[2].set_xscale("log", base=2)
    ax[2].plot(sizeTfac, 1.0 - CMTFR2X, ".", label="CMTF")
    ax[2].plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax[2].set_ylabel("Normalized Unexplained Variance")
    ax[2].set_xlabel("Size of Reduced Data")
    ax[2].set_ylim(bottom=0.0)
    ax[2].set_xlim(2 ** 8, 2 ** 12)
    ax[2].xaxis.set_major_formatter(ScalarFormatter())
    ax[2].legend()

    # Colormap

    Rlabels, agLabels = dimensionLabel3D()
    tfac = CMTFfacs[2]

    components = [str(ii + 1) for ii in range(tfac.rank)]
    comp_plot(tfac.factors[0], components, False, "Samples", ax[4])
    comp_plot(tfac.factors[1], components, agLabels, "Antigens", ax[5])
    comp_plot(tfac.factors[2], components, Rlabels, "Receptors", ax[6])

    time_plot(tfac, ax[7])
    time_plot(tfac, ax[8], condition="Negative")
    time_plot(tfac, ax[9], condition="Moderate")
    time_plot(tfac, ax[10], condition="Severe")
    time_plot(tfac, ax[11], condition="Deceased")

    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax):
    """ Creates heatmap plots for each input dimension by component. """
    sns.heatmap(factors, cmap="PiYG", center=0, xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel(plotLabel)


def time_plot(tfac, ax, condition=None):
    df = time_components_df(tfac, condition=condition)
    sns.regplot(data=df.loc[df["Factors"] == "Comp. 1", :], x="Days", y="Value", ax=ax, lowess=True, color="r",
                marker='.')
    sns.regplot(data=df.loc[df["Factors"] == "Comp. 2", :], x="Days", y="Value", ax=ax, lowess=True, color="g",
                marker='.')
    sns.regplot(data=df.loc[df["Factors"] == "Comp. 3", :], x="Days", y="Value", ax=ax, lowess=True, color="b",
                marker='.')
    sns.regplot(data=df.loc[df["Factors"] == "Comp. 4", :], x="Days", y="Value", ax=ax, lowess=True, color="k",
                marker='.')
    if condition is not None:
        ax.set_title(condition + " only")