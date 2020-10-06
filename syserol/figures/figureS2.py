"""
This creates Figure S2.
"""
import seaborn as sns
import pandas as pd 
import numpy as np
from syserol.tensor import perform_CMTF
from syserol.dataImport import importFunction, AlterIndices
from syserol.figures.common import getSetup, subplotLabel

def makeFigure():
    """ Show ADCC Values with Each Component"""
    ax, f = getSetup((7, 5), (2, 3))

    tensorFac, _, _, _ = perform_CMTF()
    components = tensorFac[1][0] #subjects x components
    df, _ = importFunction()
    subjects = np.in1d(np.arange(181), AlterIndices())
    subjType = [None] * 181
    for i, _ in enumerate(subjType):
        if subjects[i]:
            subjType[i] = "Alter"
        else:
            subjType[i] = "Leftout Alter"

    for i in np.arange(1, 7):
        data = {"ADCC": df["ADCC"], f"Component {i}": components[:, i-1], "Subjects":subjType}
        df = pd.DataFrame(data)
        sns.scatterplot(x=f"Component {i}", y="ADCC", hue="Subjects", data=df, ax=ax[i-1])

    subplotLabel(ax)
    return f