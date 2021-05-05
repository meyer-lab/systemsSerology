import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy.stats import gmean
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
        gp_id = gp120.loc[:, gp120.columns.str.contains(ig)].apply(gmean, 1)  # apply(np.geomean)
        p24_id = p24.loc[:, p24.columns.str.contains(ig)].apply(gmean, 1)
        data = {"Class":df["class.etuv"], "IgG":[ig]*181, "Value": gp_id/p24_id}
        test = pd.concat([test, pd.DataFrame(data)])
    
    # Clip values < 0
    test["Value"] = test["Value"].clip(0)

    # Plot
    sns.boxplot(x="IgG", y="Value", hue="Class", data=test, palette="colorblind", ax=ax[0])
    ax[0].set_ylim(-1, 15)
    
    subplotLabel(ax)
    return f

