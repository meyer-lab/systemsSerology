"""
This file contains functions that are used in multiple figures.
"""
import pandas as pd
import numpy as np
import seaborn as sns
from string import ascii_lowercase
from sklearn.preprocessing import scale
import matplotlib
import svgutils.transform as st
from matplotlib import gridspec, pyplot as plt
from ..tensor import perform_CMTF
from ..dataImport import getAxes, load_file


matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35


def getSetup(figsize, gridd, multz=None, empts=None):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs):
    """ Place subplot labels on figure. """
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.2, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1):
    """ Add cartoon to a figure file. """

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)

    template.append(cartoon)
    template.save(figFile)

def buildFigure3(legends=True, heatmap=False):
    tensorFac, matrixFac, _, _ = perform_CMTF()

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
        ax, f = getSetup((8, 8), (1, 4))
        
        cbar_kws = {"orientation":"horizontal"}
        subs = pd.DataFrame(subjects, columns = [f"Component {i}" for i in np.arange(1, 13)], index=subjinfo["class.etuv"])
        plt = sns.heatmap(subs, cmap="PRGn", center=0, xticklabels=2, yticklabels=50, cbar_kws=cbar_kws, ax=ax[0])
        plt.set_ylabel("")

        rec = pd.DataFrame(receptors, columns = [f"Component {i}" for i in np.arange(1, 13)], index=detections)
        sns.heatmap(rec, cmap="PRGn", center=0, xticklabels=2, yticklabels=True, cbar_kws=cbar_kws, ax=ax[1])

        ant = pd.DataFrame(antigens, columns=[f"Component {i}" for i in np.arange(1, 13)], index=antigen)
        sns.heatmap(ant, cmap="PRGn", center=0, xticklabels=2, yticklabels=True, cbar_kws=cbar_kws, ax=ax[2])

        glycans = pd.DataFrame(glyc, columns=[f"Component {i}" for i in np.arange(1, 13)], index=glycaninf["glycan"])
        a = sns.heatmap(glycans, cmap="PRGn", center=0, xticklabels=2, yticklabels=True, cbar_kws=cbar_kws, ax=ax[3])
        a.set_ylabel("")

    ax[0].set_title("Subjects", fontsize=15)
    ax[1].set_title("Receptors", fontsize=15)
    ax[2].set_title("Antigens", fontsize=15)
    ax[3].set_title("Glycans", fontsize=15)

    return f
