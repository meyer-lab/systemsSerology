"""
This creates Figure 3 for the Paper.
"""
import pandas as pd
import numpy as np
import seaborn as sns
from syserol.tensor import perform_CMTF
from syserol.dataImport import createCube, getAxes, load_file
from .common import getSetup


def makeFigure():
    """ Generate Figure 3 for Paper, Showing Better Interpretation of All Data from Decomposed Tensor"""
    ax, f = getSetup((8, 8), (3, 4))

    cube, glyCube = createCube()
    tensorFac, matrixFac, _ = perform_CMTF(cube, glyCube, 6)

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

    functions = ["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]
    index = [0, 2, 4]
    place = [0, 4, 8]
    # Build Figure
    for i, j in zip(index, place):
        # Subjects
        values1 = subjects[:, i]
        values2 = subjects[:, i + 1]
        data = {
            f"Component {i+1} Measurement": values1,
            f"Component {i+2} Measurement": values2,
            "Groups": subjinfo["class.etuv"],
        }
        df = pd.DataFrame(data)
        a = sns.scatterplot(
            x=f"Component {i+1} Measurement",
            y=f"Component {i+2} Measurement",
            hue="Groups",
            data=df,
            palette="Set1",
            legend="brief" if j==4 else False,
            ax=ax[j],
        )
        a.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
        
        # Detections
        values1 = receptors[:, i]
        values2 = receptors[:, i + 1]
        data = {
            f"Component {i+1} Measurement": values1,
            f"Component {i+2} Measurement": values2,
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
            x=f"Component {i+1} Measurement",
            y=f"Component {i+2} Measurement",
            hue="Receptor",
            style="Receptor",
            markers=markers,
            data=df,
            palette="Set2",
            legend="brief" if j==4 else False,
            ax=ax[j + 1],
        )
        b.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

        # Antigens
        values1 = antigens[:, i]
        values2 = antigens[:, i + 1]
        data = {
            f"Component {i+1} Measurement": values1,
            f"Component {i+2} Measurement": values2,
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
            x=f"Component {i+1} Measurement",
            y=f"Component {i+2} Measurement",
            hue="Antigens",
            style="Antigens",
            markers=markers,
            data=df,
            palette="Set3",
            legend="brief" if j==4 else False,
            ax=ax[j + 2],
        )
        c.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

        # Glycans/Functions
        values1 = glyc[:, i]
        values2 = glyc[:, i + 1]
        data = {
            f"Component {i+1} Measurement": values1,
            f"Component {i+2} Measurement": values2,
            "G": np.concatenate((np.array(glycaninf["GS"]), functions)),
            "FB": np.concatenate((np.array(glycaninf["FB"]), ["Function"] * 6)),
        }
        df = pd.DataFrame(data)
        d = sns.scatterplot(
            x=f"Component {i+1} Measurement",
            y=f"Component {i+2} Measurement",
            hue="G",
            style="FB",
            data=df,
            palette="Paired",
            legend="brief" if j==4 else False,
            ax=ax[j + 3],
        )
        d.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

    ax[0].set_title("Subjects", fontsize=15)
    ax[1].set_title("Receptors", fontsize=15)
    ax[2].set_title("Antigens", fontsize=15)
    ax[3].set_title("Glycans/Functions", fontsize=15)

    return f
