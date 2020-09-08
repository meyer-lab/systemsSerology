"""
This creates Figure 3 for the Paper, using heatmaps.
"""
from .common import buildFigure3


def makeFigure():
    """ Generate Figure 3 for Paper, Showing Better Interpretation of All Data from Decomposed Tensor, without Legends"""
    return buildFigure3(legends=False, heatmap=True)
