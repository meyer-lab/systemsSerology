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


    classes = []
    for cidx in range(2):
        perf = class_predictions(X)[cidx]
        perfLO = np.zeros(X.shape[1])

        for ii in range(X.shape[1]):
            XX = np.delete(X.copy(), ii, axis=1)
            perfLO[ii] = class_predictions(XX)[cidx]
        classes.extend(perf - perfLO)
    data = {"Feature Importance": classes, "Component": [str(x) for x in np.arange(1, X.shape[1]+1).tolist()] * 2, "Class": [x for i in [[j] * 10 for j in ["Progression", "Viremia"]] for x in i]}
    class_df = pd.DataFrame(data)

    funcs = []
    for function in functions:
        perf = function_prediction(X, function=function)[2]
        perfLO = np.zeros(X.shape[1])

        for ii in range(X.shape[1]):
            XX = np.delete(X.copy(), ii, axis=1)
            perfLO[ii] = function_prediction(XX, function=function)[2]
        funcs.extend(perf - perfLO)
    data = {"Feature Importance": funcs, "Component": [str(x) for x in np.arange(1, X.shape[1]+1).tolist()] * 6, "Function": [x for i in [[j] * 10 for j in functions] for x in i]}
    funcs_df = pd.DataFrame(data)

    sns.barplot(x="Component", y="Feature Importance", hue="Function", data=funcs_df, ax=ax[0])
    sns.barplot(x="Component", y="Feature Importance", hue="Class", data=class_df, ax=ax[1])
    ax[0].set_ylim(bottom=0.0)
    ax[1].set_ylim(bottom=0.0)

    # Add subplot labels
    subplotLabel(ax)

    return f