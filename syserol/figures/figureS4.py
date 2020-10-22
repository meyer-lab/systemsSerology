"""
This creates Supplementary Figure 4
"""

import pandas as pd
import numpy as np
import seaborn as sns
from ..regression import function_prediction
from ..dataImport import functions
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF


def makeFigure():
    """ Plot Function Prediction Accuracies across Components"""
    components = np.array(np.arange(1, 11))
    comp_accuracies = np.zeros_like(components)
    for i in components:
        func_accuracies = np.zeros(6)
        tensorFac, _, _ = perform_CMTF(None, None, i)
        for j, func in enumerate(functions):
            _, _, accuracy = function_prediction(
                tensorFac, function=func, evaluation="Alter"
            )
            func_accuracies[j] = accuracy
        comp_accuracies[i - 1] = np.mean(func_accuracies)

    data = {"Accuracies": comp_accuracies, "Components": np.arange(1, 11)}
    df = pd.DataFrame(data)

    # Plot
    ax, f = getSetup((7, 4), (1, 1))
    sns.set()
    a = sns.pointplot(y="Accuracies", x="Components", join=False, data=df, ax=ax[0])
    a.set_title("Performance vs. Component #")
    a.set_ylabel("Correlation Coefficient")
    subplotLabel(ax)

    return f
