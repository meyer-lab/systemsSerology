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

    axs, f = getSetup((8, 6), (3, 2))

    for ii, ax in enumerate(axs):
        function = functions[ii]

        coef = function_elastic_net(function, n_resample=10)[3]

        df = pd.DataFrame(coef,  columns=alter.columns.values[1:-6])

        varianceSorted = df.var().sort_values(ascending=False)

        df = df.reindex(varianceSorted.index, axis=1)

        a = sns.lineplot(data=df.iloc[:, :10].T,
                         ax=ax, markers=True, legend=False)
        a.set_xticklabels(df.iloc[:, :10].columns.values,
                          rotation=60, horizontalalignment='right')
        a.set_title(function)

    subplotLabel(axs)

    return f
