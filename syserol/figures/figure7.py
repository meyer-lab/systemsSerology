"""
This creates Figure 7.
"""
import matplotlib
import numpy as np
from syserol.figures.common import subplotLabel, getSetup
from syserol.model import cross_validation
from sklearn.metrics import r2_score

def makeFigure():
    """ Analyze Prediction Accuracy of 10 Fold Cross Validation Strategy"""
    ax, f = getSetup((10, 10), (3, 2))
    matrix = cross_validation()
    functions = ["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]
    # Plot Actual vs. Predicted Values for each Function
    for i in np.arange(6):
        x = matrix[0:181, i]
        y = matrix[0:181, i+6]
        ax[i].scatter(x, y)
        ax[i].set_xlabel("Actual Values", fontsize=12)
        ax[i].set_ylabel("Predicted Values", fontsize=12)
        idx = np.isfinite(x)
        r2 = r2_score(x[idx], y[idx]) # coefficient of determination
        m, b = np.polyfit(x[idx], y[idx], 1) # line of best fit
        ax[i].plot(x, m*x + b, 'k--', color="red")
        ax[i].text(1, 1, fr"$R^2$ Score: {round(r2, 3)}", {"color": "red", "fontsize": 10}, horizontalalignment="right",      
                   verticalalignment="bottom", transform=ax[i].transAxes)
        ax[i].set_title(functions[i], fontsize=15)

    subplotLabel(ax)
    return f