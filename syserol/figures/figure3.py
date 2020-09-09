"""
This creates Figure 3 for the Paper.
"""
from .common import buildFigure3


def makeFigure():
    """ Generate Figure 3 for Paper, Showing Better Interpretation of All Data from Decomposed Tensor"""
    return buildFigure3(legends=True, heatmap=False)
