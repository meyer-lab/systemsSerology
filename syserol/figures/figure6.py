import numpy as np
import pandas as pd
import seaborn as sns
from ..tensor import perform_CMTF
from ..regression import function_prediction
from ..classify import class_predictions
from ..dataImport import functions
from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))

    tFac, _, _ = perform_CMTF()
    X = tFac.factors[0]
    ncomp = X.shape[1]

    classes = []
    outt = class_predictions(X)
    classes.extend(outt[1][1] / np.max(np.absolute(outt[1][1])))
    classes.extend(outt[2][1] / np.max(np.absolute(outt[2][1])))

    data = {
        "Feature Importance": classes,
        "Component": [str(x) for x in np.arange(1, ncomp + 1).tolist()] * 2,
        "Class": [x for i in [[j] * ncomp for j in ["Controller/Progressor", "Viremic/Non-Viremic"]] for x in i],
    }
    class_df = pd.DataFrame(data)

    funcs = []
    for function in functions:
        coef = function_prediction(X, function=function)[3]
        coef /= np.max(np.absolute(coef))
        funcs.extend(coef)
    data = {
        "Feature Importance": funcs,
        "Component": [str(x) for x in np.arange(1, ncomp + 1).tolist()] * 6,
        "Function": [x for i in [[j] * ncomp for j in functions] for x in i],
    }
    funcs_df = pd.DataFrame(data)

    sns.barplot(x="Component", y="Feature Importance", hue="Function", data=funcs_df, ax=ax[0])
    sns.barplot(x="Component", y="Feature Importance", hue="Class", data=class_df, ax=ax[1])

    # Formatting
    shades = np.arange(-.5, ncomp - 1, step=2.0)
    for axx in ax:
        for i in shades:
            axx.axvspan(i, i + 1, alpha=0.1, color="grey")
        axx.set_xlim(-.5, ncomp - .5)

    # Add subplot labels
    subplotLabel(ax)

    return f
