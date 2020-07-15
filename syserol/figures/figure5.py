"""
This creates Figure 4.
"""
from .common import subplotLabel, getSetup
from syserol.dataImport import importFunction
from syserol.model import test_predictions
import numpy as np

def makeFigure():
    df, mapped = importFunction()
    ax, f = getSetup((10, 10), (3, 2))
    
    for i, func in enumerate(mapped):
        corrs = test_predictions(func)
        x = np.arange(1, 16)
        ax[i].plot(x, corrs)
        ax[i].set_ylabel("Correlation")
        ax[i].set_xlabel("# Components")
        ax[i].set_title(func)
        ax[i].set_ylim(bottom=0.0)
        
    subplotLabel(ax)
    return f
