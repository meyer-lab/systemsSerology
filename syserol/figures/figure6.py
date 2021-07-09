""" This makes Figure 6. Plot of R2X values"""
import numpy as np
import seaborn as sns
from tensorly.decomposition import parafac
from ..COVID import Tensor3D, dimensionLabel3D, time_components_df, COVIDpredict, pbsSubtractOriginal
from ..tensor import calcR2X, cp_normalize, reorient_factors, sort_factors
from .common import getSetup, subplotLabel
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
from itertools import groupby


def makeFigure():
    ax, f = getSetup((13, 9), (3, 4))
    comps = np.arange(1, 7)

    tensor, _ = Tensor3D()

    CMTFfacs = [parafac(tensor, cc, tol=1e-10, n_iter_max=1000,
                        linesearch=True, orthogonalise=2) for cc in comps]

    # Normalize factors
    CMTFfacs = [cp_normalize(f) for f in CMTFfacs]
    CMTFfacs = [reorient_factors(f) for f in CMTFfacs]
    CMTFfacs = [sort_factors(f) if i > 0 else f for i,
                f in enumerate(CMTFfacs)]

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

    # Colormap
    Rlabels, agLabels = dimensionLabel3D()
    tfac = CMTFfacs[1]

    # ROC curve
    roc_df, auc = COVIDpredict(tfac)
    roc_sum = roc_df.groupby(['FPR'], as_index=False).agg(
        {'TPR': ['mean', 'std']})

    sns.lineplot(x=roc_sum["FPR"], y=roc_sum["TPR"]
                 ["mean"], color='b', ax=ax[2])
    sns.lineplot(x=[0, 1], y=[0, 1], color="black", ax=ax[2])

    tprs_upper = np.minimum(roc_sum["TPR"]["mean"] + roc_sum["TPR"]["std"], 1)
    tprs_lower = np.maximum(roc_sum["TPR"]["mean"] - roc_sum["TPR"]["std"], 0)
    ax[2].fill_between(roc_sum["FPR"], tprs_lower,
                       tprs_upper, color='grey', alpha=.2)
    ax[2].set_title("Severe vs. Deceased ROC (AUC={}±{})".format(np.around(np.mean(auc), decimals=3),
                                                                 np.around(np.std(auc), decimals=3)))

    components = [str(ii + 1) for ii in range(tfac.rank)]
    comp_plot(tfac.factors[0], components,
              list(pbsSubtractOriginal()['group']), "Samples", ax[3], True)
    comp_plot(tfac.factors[1], components, agLabels, "Antigens", ax[4])
    comp_plot(tfac.factors[2], components, Rlabels, "Receptors", ax[5])

    time_plot(tfac, ax[6])
    time_plot(tfac, ax[7], condition="Negative")
    time_plot(tfac, ax[8], condition="Moderate")
    time_plot(tfac, ax[9], condition="Severe")
    time_plot(tfac, ax[10], condition="Deceased")

    df = time_components_df(tfac)
    sns.boxplot(data=df.loc[df["week"] == 1, :],
                x="Factors", y="value", hue="group", ax=ax[11])
    ax[11].set_title("Components of Week 1")
    ax[11].legend(loc='upper left')

    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax, d=False):
    """ Creates heatmap plots for each input dimension by component. """
    if d:
        b = [list(g) for _, g in groupby(ylabel)]
        newLabels = []
        for i, c in enumerate(b):
            newLabels.append([x + "  " if i == len(c)//2 else "–" if i ==
                              0 or i == len(c) - 1 else "·" for (i, x) in enumerate(c)])

        newLabels = [item for sublist in newLabels for item in sublist]

        sns.heatmap(factors, cmap="PiYG", center=0,
                    xticklabels=xlabel, yticklabels=newLabels, ax=ax)
    else:
        sns.heatmap(factors, cmap="PiYG", center=0,
                    xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_title(plotLabel)


def time_plot(tfac, ax, condition=None):
    df = time_components_df(tfac, condition=condition)
    colors = sns.color_palette("tab10")
    for ii, comp in enumerate(np.unique(df["Factors"])):
        ndf = df.loc[df["Factors"] == comp, :]
        sns.scatterplot(data=ndf, x="days", y="value", ax=ax,
                        palette=[colors[ii]], s=5, hue="Factors")
        xs, ys = fit_logsitic(ndf)
        sns.lineplot(x=xs, y=ys, ax=ax, palette=[colors[ii]])
    if condition is not None:
        ax.set_title(condition + " only")
    else:
        ax.set_title("All samples")


def logistic(x, A, x0, k, C):
    return A / (1 + np.exp(-k * (x - x0))) + C


def fit_logsitic(df):
    xx, yy = df["days"].values, df["value"].values
    initA = (np.max(yy)-np.min(yy)) * 0.6
    initC = np.min(yy)
    initx0 = np.median(xx)
    initk = 0.5 if np.mean(yy[xx < initx0]) < np.mean(
        yy[xx > initx0]) else -0.5
    popt, _ = curve_fit(logistic, xx, yy, p0=[
                        initA, initx0, initk, initC], ftol=1e-5, maxfev=5000)
    xs = np.arange(0, np.max(xx)+1, step=0.1)
    return xs, logistic(xs, *popt)
