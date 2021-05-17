"""
This creates Figure EV2.
"""
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..regression import function_elastic_net
from ..dataImport import importAlterDF
import pandas as pd
import matplotlib


def makeFigure():
    alter = importAlterDF()

    ax, f = getSetup((8, 10), (3, 1))

    legendDict = {
        "FcgRI": "#9BBF98",
        "FcgRIIa": "#758F4A",
        "FcgRIIb": "#C8C94E",
        "FcgRIII": "#556637",
        "IgG1": "#CBDEF1",
        "IgG2": "#EDA640",
        "IgG3": "#6192C5",
        "IgG4": "#B84632",
        "C1q": "#704F9D",
        "Lectin": "#C098BE",
        "Glycan": "#DE74A6"
    }

    colorDict = {
        "FcgRI": "#9BBF98",
        "FcgRIIa": "#758F4A",
        "FcgRIIb": "#C8C94E",
        "FcgRIII": "#556637",
        "IgG1": "#CBDEF1",
        "IgG2": "#EDA640",
        "IgG3": "#6192C5",
        "IgG4": "#B84632",
        "C1q": "#704F9D",
        "LCA": "#C098BE",
        "MBL": "#C098BE",
        "PNA": "#C098BE",
        "SNA": "#C098BE",
        "VVL": "#C098BE",
        "G0": "#DE74A6",
        "G1": "#DE74A6",
        "G2": "#DE74A6",
        "G2": "#DE74A6",
        "G2": "#DE74A6",
        "F": "#DE74A6",
        "B": "#DE74A6",
        "S": "#DE74A6",
    }

    c = function_elastic_net("ADCD", n_resample=3)[3]

    coef1 = c[0, :]
    coef2 = c[1, :]
    coef3 = c[2, :]

    for i, coef in enumerate([coef1, coef2, coef3]):

        columns = alter.columns.values[1:-6]

        df = pd.DataFrame(np.reshape(coef, (1, coef.size)),
                          columns=columns)

        df = df.loc[:, (df != 0).any(axis=0)]

        palette = {}

        for key1 in columns:
            longestPrefix = ""

            for key2 in colorDict:
                if key1.startswith(key2):
                    if len(key2) > len(longestPrefix):
                        longestPrefix = key2

            if len(longestPrefix) > 0:
                palette[key1] = colorDict[longestPrefix]
            else:
                palette[key1] = "#CCCCCC"

        sns.barplot(data=df, ax=ax[i], palette=palette)

        ax[i].set_xticklabels(df.columns.values,
                              rotation=90, horizontalalignment='center', fontsize=6)
        ax[i].set_title("ADCD")

    patches = [matplotlib.patches.Patch(
        color=v, label=k) for k, v in legendDict.items()]
    ax[2].legend(handles=patches, loc='upper center',
                 bbox_to_anchor=(0, -1.5, 1, 0.1), ncol=11, mode="expand", borderaxespad=0.)

    subplotLabel(ax)

    return f
