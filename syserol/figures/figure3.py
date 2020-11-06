"""
This creates Figure 3 for the Paper.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF
from ..dataImport import getAxes, load_file
from matplotlib import gridspec, pyplot as plt

def makeFigure():
    """ Generate Figure 3 for Paper, Showing Interpretation of All Data from Decomposed Tensor"""
    tensorFac, matrixFac, _ = perform_CMTF()
    heatmap = True
    legends=False
    # Gather tensor data matrices
    subjects = np.squeeze(tensorFac.factors[0])
    receptors = np.squeeze(tensorFac.factors[1])
    antigens = np.squeeze(tensorFac.factors[2])
    glyc = np.squeeze(matrixFac.factors[1])

    # Gather grouping info
    glycaninf = load_file("meta-glycans")
    glycaninf = glycaninf.replace(
        to_replace=["false", "b", "f", "g1", "g2", "g0", "s"],
        value=["", "B", "F", "G1", "G2", "G0", "S"],
    )
    for i in np.arange(0, len(glycaninf)):
        if "S1" in glycaninf.iloc[i, 0]:
            glycaninf.iloc[i, 2] = "S1"
        if "S2" in glycaninf.iloc[i, 0]:
            glycaninf.iloc[i, 2] = "S2"
    glycaninf["FB"] = glycaninf["f"] + glycaninf["b"]
    glycaninf["GS"] = glycaninf["g"] + glycaninf["s"]
    glycaninf["FB"] = glycaninf["FB"].replace(to_replace=[np.nan, ""], value=["Total", "No F or B"])
    glycaninf.loc[19:24, "GS"] = glycaninf.loc[19:24, "glycan"]
    _, detections, antigen = getAxes()
    subjinfo = load_file("meta-subjects")

    if (heatmap==False):
        ax, f = getSetup((8, 8), (3, 4))
        #Build Figure
        index = [0, 2, 4]
        place = [0, 4, 8]
        for i, j in zip(index, place):
            # Subjects
            values1 = subjects[:, i]
            values2 = subjects[:, i + 1]
            data = {
                f"Component {i+1}": values1,
                f"Component {i+2}": values2,
                "Groups": subjinfo["class.etuv"],
            }
            df = pd.DataFrame(data)
            a = sns.scatterplot(
                x=f"Component {i+1}",
                y=f"Component {i+2}",
                hue="Groups",
                data=df,
                palette="Set1",
                legend="brief" if j == 4 and legends==True else False,
                ax=ax[j],
            )

            if j == 4 and legends==True:
                a.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

            # Detections
            values1 = receptors[:, i]
            values2 = receptors[:, i + 1]
            data = {
                f"Component {i+1}": values1,
                f"Component {i+2}": values2,
                "Receptor": detections,
            }
            df = pd.DataFrame(data)
            markers = (
                "o",
                "X",
                "X",
                "X",
                "^",
                "D",
                "D",
                "D",
                "D",
                "D",
                "D",
                "<",
                ">",
                "8",
                "s",
                "P",
                "P",
                "P",
                "P",
                "P",
                "p",
                "d",
            )
            b = sns.scatterplot(
                x=f"Component {i+1}",
                y=f"Component {i+2}",
                hue="Receptor",
                style="Receptor",
                markers=markers,
                data=df,
                palette="Set2",
                legend="brief" if j == 4 and legends==True else False,
                ax=ax[j + 1],
            )
            if j == 4 and legends==True:
                b.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

            # Antigens
            values1 = antigens[:, i]
            values2 = antigens[:, i + 1]
            data = {
                f"Component {i+1}": values1,
                f"Component {i+2}": values2,
                "Antigens": antigen,
            }
            df = pd.DataFrame(data)
            markers = (
                "o",
                "v",
                "^",
                "<",
                ">",
                "8",
                "s",
                "p",
                "*",
                "h",
                "H",
                "D",
                "d",
                "P",
                "X",
                "o",
                "v",
                "^",
                "<",
                ">",
                "8",
                "s",
                "p",
                "*",
                "h",
                "H",
                "D",
                "d",
                "P",
                "X",
                "o",
                "v",
                "^",
                "<",
                ">",
                "8",
                "s",
                "p",
                "*",
                "h",
                "H",
            )
            c = sns.scatterplot(
                x=f"Component {i+1}",
                y=f"Component {i+2}",
                hue="Antigens",
                style="Antigens",
                markers=markers,
                data=df,
                palette="Set3",
                legend="brief" if j == 4 and legends==True else False,
                ax=ax[j + 2],
            )

            if j == 4 and legends==True:
                c.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

            # Glycans
            values1 = glyc[:, i]
            values2 = glyc[:, i + 1]
            data = {
                f"Component {i+1}": values1,
                f"Component {i+2}": values2,
                "G": glycaninf["GS"],
                "FB": glycaninf["FB"],
            }
            df = pd.DataFrame(data)
            d = sns.scatterplot(
                x=f"Component {i+1}",
                y=f"Component {i+2}",
                hue="G",
                style="FB",
                data=df,
                palette="Paired",
                legend="brief" if j == 4 and legends==True else False,
                ax=ax[j + 3],
            )

            if j == 4 and legends==True:
                d.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
        
        for aa in ax:
            aa.axis('equal')
    else:
        f = plt.figure(figsize=(21, 7))
        gs = gridspec.GridSpec(1, 10,
                            width_ratios=[3, 25, 3, 2, 16, 25, 18, 25, 10, 25],
                            wspace=0
                            )
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax4 = plt.subplot(gs[3])
        ax6 = plt.subplot(gs[5])
        ax8 = plt.subplot(gs[7])
        ax10 = plt.subplot(gs[9])

    
        colors = ["blue", "orange", "green", "red"]
        cmap = sns.color_palette(colors)

        subs = pd.DataFrame(subjects, columns = [f"Component {i}" for i in np.arange(1, subjects.shape[1] + 1)], index=subjinfo["class.etuv"])
        rec = pd.DataFrame(receptors, columns = [f"Component {i}" for i in np.arange(1, subjects.shape[1] + 1)], index=detections)
        ant = pd.DataFrame(antigens, columns=[f"Component {i}" for i in np.arange(1, subjects.shape[1] + 1)], index=antigen)
        glycans = pd.DataFrame(glyc, columns=[f"Component {i}" for i in np.arange(1, subjects.shape[1] + 1)], index=glycaninf["glycan"])

        vmin = min(subs.values.min(), rec.values.min(), ant.values.min(), glycans.values.min()) * .75
        vmax = max(subs.values.max(), rec.values.max(), ant.values.max(), glycans.values.max()) * .75

        sns.heatmap(subs, cmap="PRGn", center=0, xticklabels=True, yticklabels=False, cbar_ax=ax4, vmin=vmin, vmax=vmax, ax=ax2)

        sns.heatmap(rec, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=ax6)

        sns.heatmap(ant, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=ax8)

        sns.heatmap(glycans, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=vmin, vmax=vmax, ax=ax10)

        test = pd.DataFrame(subs.index)
        test = test.set_index(["class.etuv"])
        test["Class"] = 0
        test[test.index == "EC"] = 0
        test[test.index == "TP"] = 1
        test[test.index == "UP"] = 2
        test[test.index == "VC"] = 3

        sns.heatmap(test, ax=ax1, cbar_kws=dict(use_gridspec=False, location="left", fraction=.4, pad=.3), yticklabels=False, xticklabels=True, cmap=cmap)
        colorbar = ax1.collections[0].colorbar
        colorbar.set_ticks([0.4, 1.2, 1.9, 2.6])
        colorbar.set_ticklabels(['EC', 'UP', 'TP', "VC"])
        ax1.set_ylabel("")
        ax2.set_ylabel("")
        ax10.set_ylabel("")
        ax1.set_xticklabels(test.columns, rotation=90)
        ax = [ax2, ax6, ax8, ax10]

    ax[0].set_title("Subjects", fontsize=15)
    ax[1].set_title("Receptors", fontsize=15)
    ax[2].set_title("Antigens", fontsize=15)
    ax[3].set_title("Glycans", fontsize=15)

    return f
