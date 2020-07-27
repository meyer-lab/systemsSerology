"""
This creates Figure 7.
"""
import matplotlib
import numpy as np
from syserol.figures.common import subplotLabel, getSetup
from syserol.model import cross_validation, evaluate_diff

def makeFigure():
    """ Analyze Prediction Accuracy of 10 Fold Cross Validation Strategy"""
    x, y, _ = cross_validation()
    Sumsqs, Avg = evaluate_diff()
    ax, f = getSetup((10, 10), (2, 1))
    
    # Plot original values vs CMTF reconstructed values (after original->NaN)
    ax[0].scatter(x, y)
    ax[0].set_ylabel("Reconstructed Values", fontsize=12)
    ax[0].set_xlabel("Original Values", fontsize=12)
    ax[0].set_title("Comparison of Original and Reconstructed Values from 10 Fold Cross Validation Methods", fontsize=15)
    idx = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[idx], y[idx], 1) # line of best fit
    ax[0].plot(x, m*x + b, 'k--', color='black')

    # Plot the Difference Squared for each pair of original&reconstructed values
    # Overlay Average Difference Squared across all pairs
    ax[1].plot(Sumsqs)
    ax[1].axhline(y=Avg, linestyle='-', color="red", label="mean")
    ax[1].set_xlabel("# Values Predicted", fontsize=12)
    ax[1].set_ylabel("Sum of Squared Difference", fontsize=12)
    ax[1].set_title("Sum of Squared Difference between Original and Predicted Values from 10 Fold Cross Validation", fontsize=15)
    ax[1].legend(prop={'size': 15})
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlim(0.0, len(Sumsqs))

    subplotLabel(ax)
    return f