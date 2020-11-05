"""
This creates Figure S2.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, fcg_df
from ..regression import function_prediction
from ..dataImport import functions
from ..tensor import perform_CMTF

def makeFigure():
    """ Compare genotype vs non-genotype specific readings. """
    #Acquire dataframe of antigens
    df_2a = fcg_df("FcgRIIa", "H131", "R131")
    df_3a = fcg_df("FcgRIIIa", "F158", "V158")
    df_3b = fcg_df("FcgRIIIb", "NA1", "SH")


    ax, fig = getSetup((15, 15), (3, 3))
    a = sns.scatterplot(data=df_2a, x=df_2a.columns[0], y=df_2a.columns[1], s=20, ax=ax[0])
    b = sns.scatterplot(data=df_2a, x=df_2a.columns[0], y=df_2a.columns[2], s=20, ax=ax[1])
    c = sns.scatterplot(data=df_2a, x=df_2a.columns[1], y=df_2a.columns[2], s=20, ax=ax[2])
    d = sns.scatterplot(data=df_3a, x=df_3a.columns[0], y=df_3a.columns[1], s=20, ax=ax[3])
    e = sns.scatterplot(data=df_3a, x=df_3a.columns[0], y=df_3a.columns[2], s=20, ax=ax[4])
    f = sns.scatterplot(data=df_3a, x=df_3a.columns[1], y=df_3a.columns[2], s=20, ax=ax[5])
    g = sns.scatterplot(data=df_3b, x=df_3b.columns[0], y=df_3b.columns[1], s=20, ax=ax[6])
    h = sns.scatterplot(data=df_3b, x=df_3b.columns[0], y=df_3b.columns[2], s=20, ax=ax[7])
    i = sns.scatterplot(data=df_3b, x=df_3b.columns[1], y=df_3b.columns[2], s=20, ax=ax[8])
    axes = [a, b, c, d, e, f, g, h, i]
    for axis in axes:
        xlab = axis.get_xlabel()
        ylab = axis.get_ylabel()
        axis.set_xlabel(xlab, fontsize=15)
        axis.set_ylabel(ylab, fontsize=15)
        axis.tick_params(labelsize=10)
    a.set_xlim(-20000)
    a.set_ylim(-20000)
    b.set_xlim(-25000)
    b.set_ylim(-4000)
    c.set_xlim(-25000)
    c.set_ylim(-4000)

    subplotLabel(ax)

    return fig

