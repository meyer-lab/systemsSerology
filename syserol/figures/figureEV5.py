import numpy as np 
import pandas as pd 
import seaborn as sns
from .common import getSetup, subplotLabel
from ..dataImport import importAlterDF

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 1))

    df = importAlterDF(subjects=True)
    gp120 = df.loc[:, df.columns.str.contains("IgG") & df.columns.str.contains("gp120")]
    p24 = df.loc[:, df.columns.str.contains("IgG") & df.columns.str.contains("p24")]
    IgGs = ["IgG1", "IgG2", "IgG3", "IgG4"]

    test = pd.DataFrame(columns=["Class", "IgG", "Value"])
    for ig in IgGs:
        gp_id = gp120.loc[:, gp120.columns.str.contains(ig)].mean(axis=1)
        p24_id = p24.loc[:, p24.columns.str.contains(ig)].mean(axis=1)
        data = {"Class":df["class.etuv"], "IgG":[ig]*181, "Value": gp_id/p24_id}
        test = pd.concat([test, pd.DataFrame(data)])
    
    sns.stripplot(x="Class", y="Value", hue="IgG", data=test, palette="colorblind", ax=ax[0])
    ax[0].set_ylim(-0.5, 2)
    
    subplotLabel(ax)
    return f

