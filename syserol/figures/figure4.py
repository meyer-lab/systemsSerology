"""
This creates Figure 4.
"""

from syserol.regression import function_elastic_net
from .common import subplotLabel, getSetup
from ..dataImport import functions


def makeFigure():
    """Make Alter Model Prediction Figure for given Function"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 3))

    for i, name in enumerate(functions):
        Measured, Y_pred, _ = function_elastic_net(name)
        ax[i].scatter(Measured, Y_pred)
        ax[i].set_ylabel("Predicted", fontsize=12)
        ax[i].set_xlabel("Measured", fontsize=12)
        ax[i].set_title(f"{name} Predictability", fontsize=15)
        ax[i].plot([Y_pred.min(), Y_pred.max()], [Y_pred.min(), Y_pred.max()], 'k--', lw=4)
        subplotLabel(ax)

    return f
