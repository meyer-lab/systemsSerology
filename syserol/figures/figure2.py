"""
This creates Figure 2.
"""

from .common import subplotLabel, getSetup
from syserol.regression import function_elastic_net

def makeFigure():
    """Make Alter Model Prediction Figure for given Function"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 3))
    
    functions = ['ADCD', 'ADCC', 'ADNP', 'CD107a', 'IFNy', 'MIP1b']
    for i, name in enumerate(functions):
        Measured, Y_pred, corr = function_elastic_net(name)
        ax[i].scatter(Measured, Y_pred)
        ax[i].set_ylabel("Predicted", fontsize = 12)
        ax[i].set_xlabel("Measured", fontsize = 12)
        ax[i].set_title(f"{name} Predictability", fontsize = 15)
        ax[i].plot([Y_pred.min(), Y_pred.max()], [Y_pred.min(), Y_pred.max()], 'k--', lw=4)
        subplotLabel(ax)

    return f
