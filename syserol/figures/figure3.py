"""
This creates Figure 3 for the Paper.
"""
import pandas as pd
import numpy as np
import seaborn as sns
from ..tensor import perform_CMTF
from ..dataImport import getAxes, load_file
from .common import buildFigure3


def makeFigure():
    """ Generate Figure 3 for Paper, Showing Better Interpretation of All Data from Decomposed Tensor"""
    return buildFigure3(legends=True, heatmap=False)
