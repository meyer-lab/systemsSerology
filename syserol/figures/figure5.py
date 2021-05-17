import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..tensor import perform_CMTF
from ..regression import function_prediction
from ..classify import class_predictions
from ..dataImport import functions
from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))

    tFac = perform_CMTF()
    X = tFac.factors[0]
    ncomp = X.shape[1]
    nboot = 20

    class_df = pd.DataFrame()
    for _ in range(nboot):
        classes = []
        outt = class_predictions(X)
        classes.extend(outt[1])
        classes.extend(outt[2])

        data = {
            "Feature Importance": classes,
            "Component": [str(x) for x in np.arange(1, ncomp + 1).tolist()] * 2,
            "Class": [x for i in [[j] * ncomp for j in ["Controller/Progressor", "Viremic/Non-Viremic"]] for x in i],
        }
        class_df = class_df.append(pd.DataFrame(data), ignore_index=True)

    funcs_df = pd.DataFrame()
    for _ in range(nboot):
        funcs = []
        for function in functions:
            coef = function_prediction(X, resample=True, function=function)[3]
            funcs.extend(coef)
        data = {
            "Feature Importance": funcs,
            "Component": [str(x) for x in np.arange(1, ncomp + 1).tolist()] * 6,
            "Function": [x for i in [[j] * ncomp for j in functions] for x in i],
        }
        funcs_df = funcs_df.append(pd.DataFrame(data), ignore_index=True)

    sns.barplot(x="Component", y="Feature Importance", ci="sd",
                hue="Function", data=funcs_df, errwidth=1, ax=ax[0])
    sns.barplot(x="Component", y="Feature Importance", ci="sd", hue="Class", data=class_df,
                errwidth=2, ax=ax[1], palette=sns.color_palette('magma', n_colors=3))
    # plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1)
    ax[1].set_ylim(-0.75, 1.5)

    # Formatting
    shades = np.arange(-0.5, ncomp - 1, step=2.0)
    for axx in ax:
        for i in shades:
            axx.axvspan(i, i + 1, alpha=0.1, color="grey")
        axx.set_xlim(-0.5, ncomp - 0.5)

    # Add subplot labels
    subplotLabel(ax)

    return f
