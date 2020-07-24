"""
This creates Figure 5.
"""
from syserol.dataImport import importFunction
from syserol.model import test_predictions
from .common import subplotLabel, getSetup
import numpy as np

def makeFigure():
    """ Plot correlation for each function between original and reconstruced glycube matrix across increasing(16) components"""
    _, mapped = importFunction()
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
