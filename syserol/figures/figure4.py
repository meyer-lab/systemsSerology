"""
This creates Figure 4.
"""

from .common import subplotLabel, getSetup

def makeFigure():
    """ Builds the figure. """
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 3))

    subplotLabel(ax)

    return f
