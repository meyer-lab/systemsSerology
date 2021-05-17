"""
This creates Figure 6e for the Paper.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from string import ascii_lowercase
from ..tensor import perform_CMTF
from ..dataImport import getAxes, load_file
from matplotlib import gridspec, pyplot as plt


def makeFigure():
    """ Generate Figure 6e for Paper"""
    tensorFac = perform_CMTF()

    f = plt.figure()

    # Gather grouping info
    glycaninf = load_file("meta-glycans")
    glycaninf = glycaninf.replace(to_replace=["false", "b", "f", "g1", "g2", "g0", "s"], value=[
        "", "B", "F", "G1", "G2", "G0", "S"],)
    for i in np.arange(0, len(glycaninf)):
        if "S1" in glycaninf.iloc[i, 0]:
            glycaninf.iloc[i, 2] = "S1"
        if "S2" in glycaninf.iloc[i, 0]:
            glycaninf.iloc[i, 2] = "S2"
    glycaninf["FB"] = glycaninf["f"] + glycaninf["b"]
    glycaninf["GS"] = glycaninf["g"] + glycaninf["s"]
    glycaninf["FB"] = glycaninf["FB"].replace(
        to_replace=[np.nan, ""], value=["Total", "No F or B"])
    glycaninf.loc[19:24, "GS"] = glycaninf.loc[19:24, "glycan"]
    _, detections, antigen = getAxes()
    detections = [x[:2] + "γ" + x[3:] if x[:2]
                  == "Fc" else x for x in detections]
    subjinfo = load_file("meta-subjects")

    rec = pd.DataFrame(tensorFac.factors[1], columns=[
        f"Cmp. {i}" for i in np.arange(1, tensorFac.rank + 1)], index=detections)

    small_rec = rec.loc[['FcγRI', 'IgG2', 'IgG3', 'IgG4', 'MBL', 'VVL']]
    ax = sns.heatmap(small_rec, cmap="PRGn", center=0,
                     yticklabels=True, cbar=True, vmax=0.020, vmin=-0.02)

    ax.text(-0.2, 1.1, 'e', transform=ax.transAxes,
            fontsize=25, fontweight="bold", va="top")

    return f
