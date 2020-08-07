"""
This creates Figure 3 for the Paper.
"""

from syserol.tensor import perform_CMTF
from syserol.dataImport import createCube, getAxes, load_file, importGlycan
from .common import subplotLabel, getSetup
import pandas as pd
import numpy as np
import seaborn as sns


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
        data = {f"Component {i+1} Measurement": values1, f"Component {i+2} Measurement": values2, "Groups": subjinfo["class.etuv"]}
        df = pd.DataFrame(data)
        a = sns.scatterplot(
            x=f"Component {i+1} Measurement", y=f"Component {i+2} Measurement", hue="Groups", data=df, palette="Set1", legend="brief", ax=ax[j]
        )
        a.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

        # Detections
        values1 = receptors[:, i]
        values2 = receptors[:, i + 1]
        data = {f"Component {i+1} Measurement": values1, f"Component {i+2} Measurement": values2, "Groups": detections}
        df = pd.DataFrame(data)
        b = sns.scatterplot(
            x=f"Component {i+1} Measurement", y=f"Component {i+2} Measurement", hue="Groups", data=df, palette="Set2", legend="brief", ax=ax[j + 1]
        )
        b.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

        # Antigens
        values1 = antigens[:, i]
        values2 = antigens[:, i + 1]
        data = {f"Component {i+1} Measurement": values1, f"Component {i+2} Measurement": values2, "Groups": antigen}
        df = pd.DataFrame(data)
        c = sns.scatterplot(
            x=f"Component {i+1} Measurement", y=f"Component {i+2} Measurement", hue="Groups", data=df, palette="Set3", legend="brief", ax=ax[j + 2]
        )
        c.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

        # Glycans/Functions
        values1 = glyc[:, i]
        values2 = glyc[:, i + 1]
        data = {
            f"Component {i+1} Measurement": values1,
            f"Component {i+2} Measurement": values2,
            "Groups": np.concatenate((np.array(glycaninf["glycan"]), functions)),
        }
        df = pd.DataFrame(data)
        d = sns.scatterplot(
            x=f"Component {i+1} Measurement", y=f"Component {i+2} Measurement", hue="Groups", data=df, palette="muted", legend="brief", ax=ax[j + 3]
        )
        d.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

    ax[0].set_title("Subjects", fontsize=15)
    ax[1].set_title("Receptors", fontsize=15)
    ax[2].set_title("Antigens", fontsize=15)
    ax[3].set_title("Glycans/Functions", fontsize=15)

    return f
