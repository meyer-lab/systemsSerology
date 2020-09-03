"""
This creates Figure 5.
"""
import numpy as np
from .common import subplotLabel, getSetup
from ..regression import noCMTF_function_prediction
from ..dataImport import functions
from ..tensor import perform_CMTF

def makeFigure():
    """ Show predicted vs. actual of decomposition prediction. """
    ax, f = getSetup((10, 10), (3, 2))
    # Plot Actual vs. Predicted Values for each Function
    tensorFac, _, _, _ = perform_CMTF()
    for i, func in enumerate(functions):
        x, y, accuracy = noCMTF_function_prediction(tensorFac, function=func)
        ax[i].scatter(x, y)
        ax[i].set_xlabel("Actual Values")
        ax[i].set_ylabel("Predicted Values")
        m, b = np.polyfit(x, y, 1)  # line of best fit
        ax[i].plot(x, m * x + b, 'k--', color="red")
        ax[i].text(1, 1, f"Accuracy Score: {round(accuracy, 3)}", {"color": "red", "fontsize": 10}, horizontalalignment="right",
                   verticalalignment="bottom", transform=ax[i].transAxes)
        ax[i].set_title(func)
        ax[i].axis('equal')

    subplotLabel(ax)
    return f
