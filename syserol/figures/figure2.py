"""
This creates Figure 2.
"""

from .common import subplotLabel, getSetup

def makeFigure(function):
    """Make Alter Model Prediction Figure for given Function"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))
    
    Measured, Y_pred, corr = function_elastic_net(function)
    ax[0].scatter(Measured, Y_pred)
    ax[0].set_ylabel("Predicted", fontsize = 12)
    ax[0].set_xlabel("Measured", fontsize = 12)
    ax[0].set_title(f"{function} Predictability", fontsize = 15)
    ax[0].plot([Y_pred.min(), Y_pred.max()], [Y_pred.min(), Y_pred.max()], 'k--', lw=4)

    # Add subplot labels
    subplotLabel(ax)

    return f
