"""
This creates Figure S1.
"""
import numpy as np
from .common import subplotLabel, getSetup
from ..regression import function_prediction
from ..dataImport import functions
from ..tensor import perform_CMTF

def makeFigure():
    """ Show predicted vs. actual of decomposition prediction. """
    ax, f = getSetup((7, 5), (2, 3))
    # Plot Actual vs. Predicted Values for each Function
    tensorFac, _, _ = perform_CMTF()
    all_function_preds = function_prediction(tensorFac, evaluation="all")
    for i, func in enumerate(all_function_preds):
        x, y = func[1], func[2]
        ax[i].scatter(x, y, s=2)
        ax[i].set_xlabel("Actual Values")
        ax[i].set_ylabel("Predicted Values")
        m, b = np.polyfit(x, y, 1)  # line of best fit
        ax[i].plot(x, m * x + b, 'k--', color="red")
        ax[i].text(1, 1, f"Accuracy Score: {round(func[0], 3)}", {"color": "red", "fontsize": 10}, horizontalalignment="right",
                   verticalalignment="bottom", transform=ax[i].transAxes)
        ax[i].set_title(functions[i])
        ax[i].axis('equal')

    subplotLabel(ax)
    return f
