"""
This creates Paper Figure 4 - Component prediction weights.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from ..regression import (
    function_prediction
)
from ..dataImport import functions
from ..classify import class_predictions, two_way_classifications
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF

def makeFigure():
    """Plot weighting of components for function and class prediction"""
    #Factor data
    tFac, _, _ = perform_CMTF()
    #Collect function component weights from elastic net prediction
    function_coefs = [function_prediction(tFac, function=f, evaluation="all")[3] for f in functions]
    flat_func_coefs = [func_coef for func in function_coefs for func_coef in func]
    function = [fun for fun in functions for i in range(6)]
    components = [1, 2, 3, 4, 5, 6] * 6
    data = {"Weights": flat_func_coefs, "Function": function, "Component": components}
    function_df = pd.DataFrame(data)

    #Collect classification component weights
    _, _, cp_coef, nv_coef = class_predictions(tFac[1][0])
    components = [1, 2, 3, 4, 5, 6] * 2
    category = ["Progression"] * 6 + ["Viremia"] * 6
    data = {"Weights": [ele for arr in np.hstack([cp_coef, nv_coef]) for ele in arr], "Class": category, "Component": components}
    class_df = pd.DataFrame(data)

    #Plotting
    ax, f = getSetup((10, 5), (1, 2))
    sns.set()
    a = sns.barplot(data=function_df, x="Component", y="Weights", hue="Function", ax=ax[0])
    b = sns.barplot(data=class_df, x="Component", y="Weights", hue="Class", ax=ax[1])
    subplotLabel(ax)

    return f
