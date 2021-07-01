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
from scipy.optimize import curve_fit


def makeFigure():
    ax, f = getSetup((11, 14), (4, 3))
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

    time_plot(tfac, ax[6])
    time_plot(tfac, ax[7], condition="Negative")
    time_plot(tfac, ax[8], condition="Moderate")
    time_plot(tfac, ax[9], condition="Severe")
    time_plot(tfac, ax[10], condition="Deceased")

    df = time_components_df(tfac)
    sns.boxplot(data=df.loc[df["week"] == 1, :], x="Factors", y="value", hue="group", ax=ax[11])

    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax):
    """ Creates heatmap plots for each input dimension by component. """
    sns.heatmap(factors, cmap="PiYG", center=0, xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel(plotLabel)


def time_plot(tfac, ax, condition=None):
    df = time_components_df(tfac, condition=condition)
    colors = ["r", "g", "b", "c", "m", "y"]
    for ii, comp in enumerate(np.unique(df["Factors"])):
        ndf = df.loc[df["Factors"] == comp, :]
        #sns.regplot(data=df.loc[df["Factors"] == comp, :], x="days", y="value", ax=ax, logistic=True, color=colors[ii],
        #            marker='.', scatter_kws={"s": 10})
        sns.scatterplot(data=ndf, x="days", y="value", ax=ax, palette=[colors[ii]], s=5)
        xs, ys = fit_logsitic(ndf)
        sns.lineplot(x=xs, y=ys, ax=ax, palette=[colors[ii]])
    if condition is not None:
        ax.set_title(condition + " only")


def logistic(x, A, x0, k, C):
    return A / (1 + np.exp(-k * (x - x0))) + C

def fit_logsitic(df):
    xx, yy = df["days"].values, df["value"].values
    initA = (np.max(yy)-np.min(yy)) * 0.6
    initC = np.min(yy)
    initx0 = np.median(xx)
    initk = 0.5 if np.mean(yy[xx<initx0]) < np.mean(yy[xx>initx0]) else -0.5
    popt, _ = curve_fit(logistic, xx, yy, p0=[initA, initx0, initk, initC], ftol=1e-5, maxfev=5000)
    xs = np.arange(0, np.max(xx)+1, step = 0.1)
    return xs, logistic(xs, *popt)