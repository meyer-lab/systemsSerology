"""
This creates Figure 3.
"""
import numpy as np
from .common import subplotLabel, getSetup
from syserol.impute import evaluate_missing


def makeFigure():
    """ Evaluate Handling of Missing Values in Decomposition """
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))
    
    Sumsqs = evaluate_missing()
    
    x = np.arange(1,10)
    ax[0].plot(x, Sumsqs)
    ax[0].set_ylabel("Normalized Reconstruction Error")
    ax[0].set_xlabel("# Components")
    ax[0].set_ylim(bottom=0.0)
  
    # Add subplot labels
    subplotLabel(ax)

    return f
