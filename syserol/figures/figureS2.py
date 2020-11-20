"""
This creates Figure S2.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from ..regression import function_prediction
from ..dataImport import functions
from ..tensor import perform_CMTF


def fcg_df(receptor, geno1, geno2):
    # Import luminex readings
    test = pd.read_csv("syserol/data/data-luminex.csv")
    geno1 = receptor + "." + geno1
    geno2 = receptor + "." + geno2
    cols = [col for col in test.columns if receptor in col]
    # Set up dictionary for antigens
    dict_receptor = {geno1: [], geno2: [], receptor: []}
    for col in cols:
        if col[:len(geno1)] == geno1:
            dict_receptor[geno1].append(col[len(geno1) + 1:])
        elif col[:len(geno2)] == geno2:
            dict_receptor[geno2].append(col[len(geno2) + 1:])
        elif col[:len(receptor)] == receptor:
            dict_receptor[receptor].append(col[len(receptor) + 1:])
    # Pull values from data and concatenate
    all_antis = []
    for i in range(len(dict_receptor[receptor])):
        test_col = [val for val in cols if val[-len(dict_receptor[receptor][i]):] == dict_receptor[receptor][i]]
        anti = test[test_col]
        if test[test_col].shape[1] == 3:
            anti.columns = [geno1, geno2, receptor]
            all_antis.append(anti)
    df = pd.concat(all_antis)
    return df


def makeFigure():
    """ Compare genotype vs non-genotype specific readings. """
    # Acquire dataframe of antigens
    df_2a = fcg_df("FcgRIIa", "H131", "R131")
    df_3a = fcg_df("FcgRIIIa", "F158", "V158")
    df_3b = fcg_df("FcgRIIIb", "NA1", "SH")

    ax, fig = getSetup((15, 15), (3, 3))
    sns.kdeplot(data=df_2a, x=df_2a.columns[0], y=df_2a.columns[1], ax=ax[0])
    sns.kdeplot(data=df_2a, x=df_2a.columns[0], y=df_2a.columns[2], ax=ax[1])
    sns.kdeplot(data=df_2a, x=df_2a.columns[1], y=df_2a.columns[2], ax=ax[2])
    sns.kdeplot(data=df_3a, x=df_3a.columns[0], y=df_3a.columns[1], ax=ax[3])
    sns.kdeplot(data=df_3a, x=df_3a.columns[0], y=df_3a.columns[2], ax=ax[4])
    sns.kdeplot(data=df_3a, x=df_3a.columns[1], y=df_3a.columns[2], ax=ax[5])
    sns.kdeplot(data=df_3b, x=df_3b.columns[0], y=df_3b.columns[1], ax=ax[6])
    sns.kdeplot(data=df_3b, x=df_3b.columns[0], y=df_3b.columns[2], ax=ax[7])
    sns.kdeplot(data=df_3b, x=df_3b.columns[1], y=df_3b.columns[2], ax=ax[8])

    subplotLabel(ax)

    return fig
