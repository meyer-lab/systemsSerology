"""
This creates Figure 4 for the Paper.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from string import ascii_lowercase
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF
from ..dataImport import getAxes, load_file
from matplotlib import gridspec, pyplot as plt


def makeFigure():
    """ Generate Figure 3 for Paper, Showing Interpretation of All Data from Decomposed Tensor"""
    tensorFac, matrixFac, _ = perform_CMTF()

    # Gather tensor data matrices
    subjects = np.squeeze(tensorFac.factors[0])
    receptors = np.squeeze(tensorFac.factors[1])
    antigens = np.squeeze(tensorFac.factors[2])
    glyc = np.squeeze(matrixFac.factors[1])

    # Gather grouping info
    glycaninf = load_file("meta-glycans")
    glycaninf = glycaninf.replace(to_replace=["false", "b", "f", "g1", "g2", "g0", "s"], value=["", "B", "F", "G1", "G2", "G0", "S"],)
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
    detections = [x[:2] + "γ" + x[3:] if x[:2] == "Fc" else x for x in detections]
    subjinfo = load_file("meta-subjects")

    f = plt.figure(figsize=(21, 7))
    gs = gridspec.GridSpec(1, 10, width_ratios=[3, 25, 3, 2, 16, 25, 18, 25, 10, 25], wspace=0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax4 = plt.subplot(gs[3])
    ax6 = plt.subplot(gs[5])
    ax8 = plt.subplot(gs[7])
    ax10 = plt.subplot(gs[9])

    colors = ["blue", "orange", "green", "red"]
    cmap = sns.color_palette(colors)

    subs = pd.DataFrame(subjects, columns=[f"Cmp. {i}" for i in np.arange(1, subjects.shape[1] + 1)], index=subjinfo["class.etuv"])
    rec = pd.DataFrame(receptors, columns=[f"Cmp. {i}" for i in np.arange(1, subjects.shape[1] + 1)], index=detections)
    ant = pd.DataFrame(antigens, columns=[f"Cmp. {i}" for i in np.arange(1, subjects.shape[1] + 1)], index=antigen)
    glycans = pd.DataFrame(glyc, columns=[f"Cmp. {i}" for i in np.arange(1, subjects.shape[1] + 1)], index=glycaninf["glycan"])

    sns.heatmap(subs, cmap="PRGn", center=0, xticklabels=True, yticklabels=False, cbar_ax=ax4, vmin=-1.0, vmax=1.0, ax=ax2)
    sns.heatmap(rec, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=-1.0, vmax=1.0, ax=ax6)
    sns.heatmap(ant, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=-1.0, vmax=1.0, ax=ax8)
    sns.heatmap(glycans, cmap="PRGn", center=0, yticklabels=True, cbar=False, vmin=-1.0, vmax=1.0, ax=ax10)

    test = pd.DataFrame(subs.index)
    test = test.set_index(["class.etuv"])
    test["Class"] = 0
    test[test.index == "EC"] = 0
    test[test.index == "TP"] = 1
    test[test.index == "UP"] = 2
    test[test.index == "VC"] = 3

    sns.heatmap(
        test, ax=ax1, cbar_kws=dict(use_gridspec=False, location="left", fraction=0.4, pad=0.3), yticklabels=False, xticklabels=True, cmap=cmap
    )
    colorbar = ax1.collections[0].colorbar
    colorbar.set_ticks([0.4, 1.2, 1.9, 2.6])
    colorbar.set_ticklabels(["EC", "UP", "TP", "VC"])
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax10.set_ylabel("")
    ax1.set_xticklabels(test.columns, rotation=90)
    ax = [ax2, ax6, ax8, ax10]

    ax[0].set_title("Subjects", fontsize=15)
    ax[1].set_title("Receptors", fontsize=15)
    ax[2].set_title("Antigens", fontsize=15)
    ax[3].set_title("Glycans", fontsize=15)

    for ii, ax in enumerate(ax):
        ax.text(-0.2, 1.1, ascii_lowercase[ii], transform=ax.transAxes, fontsize=25, fontweight="bold", va="top")

    return f
