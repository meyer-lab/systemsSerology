"""
This creates Figure S2.
"""
import numpy as np
import seaborn as sns
import tensorly as tl
from .common import subplotLabel, getSetup
from ..dataImport import createCube, getAxes


def makeFigure():
    """ Compare genotype vs non-genotype specific readings. """
    cube, _ = createCube()
    _, detections, _ = getAxes()

    cube = tl.unfold(cube[:, 1:11, :], 1)
    cube = np.delete(cube, 3, axis=1)
    detections = detections[1:11]
    del detections[3]

    # Remove fully missing patients
    missing = np.all(np.isnan(cube), axis=0)
    cube = cube[:, ~missing]

    axs, fig = getSetup((10, 10), (3, 3))

    for ii, ax in enumerate(axs):
        groupi = ii - (ii % 3)
        xi = groupi + [1, 1, 2][ii % 3]
        yi = groupi + [0, 2, 0][ii % 3]

        data = cube[(xi, yi), :]
        miss = np.all(np.isfinite(data), axis=0)
        data = data[:, miss]

        sns.regplot(x=data[0, :], y=data[1, :], ax=ax, lowess=True, scatter_kws={"s": 0.5})
        ax.set_xlabel(detections[xi])
        ax.set_ylabel(detections[yi])

        ax.set_ylim(bottom = -2000)
        ax.set_xlim(left = -2000)

    subplotLabel(axs)

    return fig
