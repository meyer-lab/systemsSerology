""" This makes Figure 6. Plot of R2X values"""
import numpy as np
import seaborn as sns
from tensorly.decomposition import parafac
from ..COVID import Tensor3D, dimensionLabel3D, time_components_df
from ..tensor import calcR2X, tensor_degFreedom, cp_normalize, reorient_factors, sort_factors
from .common import getSetup, subplotLabel
from ..impute import flatten_to_mat
from statsmodels.multivariate.pca import PCA
from matplotlib.ticker import ScalarFormatter


def makeFigure():
    ax, f = getSetup((14, 10), (3, 4))
    comps = np.arange(1, 7)

    tensor, _ = Tensor3D()

    CMTFfacs = [parafac(tensor, cc, tol=1e-12, n_iter_max=4000, linesearch=True, orthogonalise=2) for cc in comps]

    # Normalize factors
    CMTFfacs = [cp_normalize(f) for f in CMTFfacs]
    CMTFfacs = [reorient_factors(f) for f in CMTFfacs]
    CMTFfacs = [sort_factors(f) if i > 0 else f for i, f in enumerate(CMTFfacs)]

    # Calculate R2X
    CMTFR2X = np.array([calcR2X(f, tensor) for f in CMTFfacs])
    print(CMTFR2X)

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

    # Flip comp. 2
    tfac.factors[0][:, 1] *= -1
    tfac.factors[2][:, 1] *= -1

    components = [str(ii + 1) for ii in range(tfac.rank)]
    comp_plot(tfac.factors[0], components, False, "Samples", ax[3])
    comp_plot(tfac.factors[1], components, agLabels, "Antigens", ax[4])
    comp_plot(tfac.factors[2], components, Rlabels, "Receptors", ax[5])

    time_plot(tfac, ax[7])
    time_plot(tfac, ax[8], condition="Negative")
    time_plot(tfac, ax[9], condition="Moderate")
    time_plot(tfac, ax[10], condition="Severe")
    time_plot(tfac, ax[11], condition="Deceased")

    #sns.boxplot(data=df.loc[df["week"] == 3, :], x="variable", y="value", hue="group")

    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax):
    """ Creates heatmap plots for each input dimension by component. """
    sns.heatmap(factors, cmap="PiYG", center=0, xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel(plotLabel)


def time_plot(tfac, ax, condition=None):
    df = time_components_df(tfac, condition=condition)
    sns.regplot(data=df.loc[df["Factors"] == "Comp. 1", :], x="days", y="value", ax=ax, lowess=True, color="r",
                marker='.', scatter_kws={"s": 10})
    sns.regplot(data=df.loc[df["Factors"] == "Comp. 2", :], x="days", y="value", ax=ax, lowess=True, color="g",
                marker='.', scatter_kws={"s": 10})
    sns.regplot(data=df.loc[df["Factors"] == "Comp. 3", :], x="days", y="value", ax=ax, lowess=True, color="b",
                marker='.', scatter_kws={"s": 10})
    if condition is not None:
        ax.set_title(condition + " only")