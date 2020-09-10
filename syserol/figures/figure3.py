"""
This creates Figure 3 for the Paper.
"""
from .common import buildFigure3


def makeFigure():
    """ Generate Figure 3 for Paper, Showing Interpretation of All Data from Decomposed Tensor"""
    return buildFigure3(legends=False, heatmap=True)
