"""
This creates Figure S3.
"""
import numpy as np
import seaborn as sns
import tensorly as tl
from scipy.optimize import least_squares
from .common import subplotLabel, getSetup
from ..dataImport import createCube, getAxes
from ..regression import function_elastic_net
from ..dataImport import importAlterDF
import pandas as pd

import sys


def makeFigure():
    functions = ["ADCD", "ADCC", "ADNP", "CD107a", "IFNγ", "MIP1β"]

    alter = importAlterDF()

    ax, f = getSetup((8, 10), (3, 1))

    c = function_elastic_net("ADCD", n_resample=3)[3]

    coef1 = c[0, :]
    coef2 = c[1, :]
    coef3 = c[2, :]

    for i, coef in enumerate([coef1, coef2, coef3]):

        df = pd.DataFrame(np.reshape(coef, (1, coef.size)),
                          columns=alter.columns.values[1:-6])

        df = df.loc[:, (df != 0).any(axis=0)]

        sns.barplot(data=df, ax=ax[i])

        ax[i].set_xticklabels(df.columns.values,
                              rotation=90, horizontalalignment='right')
        ax[i].set_title("ADCD")

    subplotLabel(ax)

    return f
