import numpy as np
import pandas as pd
import seaborn as sns
from .common import getSetup, subplotLabel
from ..tensor import perform_CMTF
from ..dataImport import load_file

def makeFigure():
    ax, f = getSetup((6, 2), (1, 3))

    tensorFac, matrixFac, _ = perform_CMTF()
    subjects = np.squeeze(tensorFac.factors[0])

    subjinfo = load_file("meta-subjects")

    index = [0, 2]
    place = [0, 1]
    for i, j in zip(index, place):
        # Subjects
        values1 = subjects[:, i]
        values2 = subjects[:, i+1]
        data = {f"Component {i+1} Measurement": values1, f"Component {i+2} Measurement": values2, "Groups": subjinfo["class.etuv"]}
        df = pd.DataFrame(data)
        xmax = np.amax(np.absolute(values1))
        ymax = np.amax(np.absolute(values2))
        a = sns.scatterplot(x=f"Component {i+1} Measurement", y=f"Component {i+2} Measurement", hue="Groups", data=df, palette="Set1", legend="brief", ax=ax[j])
        a.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        a.set_xlim(-xmax, xmax)
        a.set_ylim(-ymax, ymax)

    subplotLabel(ax)
    return f
