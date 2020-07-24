"""
This creates Figure 4.
"""

from syserol.model import CMTF_elastic_function_predictions
from .common import subplotLabel, getSetup

def makeFigure():
    """Function Prediction Figures using Our Model with Factorization"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 3))

    functions = ['ADCD', 'ADCC', 'ADNP', 'CD107a', 'IFNy', 'MIP1b']
    for i, name in enumerate(functions):
        Measured, Y_pred, _ = CMTF_elastic_function_predictions(name)
        ax[i].scatter(Measured, Y_pred)
        ax[i].set_ylabel("Predicted", fontsize=12)
        ax[i].set_xlabel("Measured", fontsize=12)
        ax[i].set_title(f"{name} Predictability", fontsize=15)
        ax[i].plot([Y_pred.min(), Y_pred.max()], [Y_pred.min(), Y_pred.max()], 'k--', lw=4)
        subplotLabel(ax)

    return f
