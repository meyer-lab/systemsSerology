"""
This creates Figure 3.
"""
import numpy as np
from .common import subplotLabel, getSetup
from syserol.impute import evaluate_missing


def makeFigure():
    """ Evaluate Handling of Missing Values in Decomposition """
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 1))
    
    Averages, Sums = evaluate_missing()
    
    x = np.arange(1,10)
    ax[0].plot(x, Averages)
    ax[0].set_ylabel("Difference", fontsize = 12)
    ax[0].set_xlabel("Component", fontsize = 12)
    ax[0].set_title("Average Difference Between 100 Original and Reconstructed Values", fontsize = 15)

    ax[1].plot(x, Sums)
    ax[1].set_ylabel("Sum", fontsize = 12)
    ax[1].set_xlabel("Component", fontsize = 12)
    ax[1].set_title("Sum of Differences between 100 Original and Reconstructed Values", fontsize = 15)
    
    # Add subplot labels
    subplotLabel(ax)

    return f
