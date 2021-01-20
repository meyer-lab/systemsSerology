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
    for i, func in enumerate(functions):
        x, y, accuracy, _ = function_prediction(tensorFac[1][0], function=func)
        ax[i].scatter(x, y, s=2)
        ax[i].set_xlabel("Actual Values")
        ax[i].set_ylabel("Predicted Values")
        m, b = np.polyfit(x, y, 1)  # line of best fit
        ax[i].plot(x, m * x + b, "k--", color="red")
        ax[i].text(
            0.9,
            0.1,
            f"ρ = {round(accuracy, 3)}",
            {"color": "red", "fontsize": 8},
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax[i].transAxes,
        )
        ax[i].set_title(func)
        lim = max(max(x), max(y))
        ax[i].set_xlim(0, lim * 1.1)
        ax[i].set_ylim(0, lim * 1.1)

    subplotLabel(ax)
    return f
