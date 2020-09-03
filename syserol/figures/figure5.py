"""
This creates Figure 5.
"""
import numpy as np
from syserol.figures.common import subplotLabel, getSetup
from ..regression import noCMTF_function_prediction
from ..dataImport import functions, createCube
from ..tensor import perform_CMTF_def

def makeFigure():
    """ Analyze Prediction Accuracy of 10 Fold Cross Validation Strategy"""
    ax, f = getSetup((10, 10), (3, 2))
    # Plot Actual vs. Predicted Values for each Function
    tensorFac, _, _, _ = perform_CMTF_def(6)
    for i, func in enumerate(functions):
        x, y, accuracy = noCMTF_function_prediction(tensorFac, function=func)
        ax[i].scatter(x, y)
        ax[i].set_xlabel("Actual Values", fontsize=12)
        ax[i].set_ylabel("Predicted Values", fontsize=12)
        m, b = np.polyfit(x, y, 1)  # line of best fit
        ax[i].plot(x, m * x + b, 'k--', color="red")
        ax[i].text(1, 1, f"Accuracy Score: {round(accuracy, 3)}", {"color": "red", "fontsize": 10}, horizontalalignment="right",
                   verticalalignment="bottom", transform=ax[i].transAxes)
        ax[i].set_title(func, fontsize=15)

    subplotLabel(ax)
    return f
