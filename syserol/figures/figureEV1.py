"""
This creates Figure EV1.
"""
import numpy as np
import seaborn as sns
import tensorly as tl
from scipy.optimize import least_squares
from .common import subplotLabel, getSetup
from ..dataImport import createCube, getAxes


def makeFigure():
    """ Compare genotype vs non-genotype specific readings. """
    cube, _ = createCube()
    _, detections, _ = getAxes()

    cube = tl.unfold(cube[:, 1:11, :], 1)
    cube = np.delete(cube, 3, axis=1)
    detections = detections[1:11]
    detections = [x[:2] + "γ" + x[3:] if x[:2] == "Fc" else x for x in detections]
    del detections[3]

    # Remove fully missing subjects
    missing = np.all(np.isnan(cube), axis=0)
    cube = cube[:, ~missing]

    axs, fig = getSetup((8, 8), (3, 3))

    for ii, ax in enumerate(axs):
        groupi = ii - (ii % 3)
        xi = groupi + [1, 1, 2][ii % 3]
        yi = groupi + [0, 2, 0][ii % 3]

        data = cube[(xi, yi), :]
        miss = np.all(np.isfinite(data), axis=0)
        data = data[:, miss]

        ax.scatter(data[0, :], data[1, :], s=0.3)

        def pfunc(x, p):
            return np.power(x, p[0]) * p[1]

        popt = least_squares(lambda x: pfunc(data[0, :], x) - data[1, :], x0=[1.0, 1.0], jac="3-point")
        linx = np.linspace(0.0, np.amax(data[0, :]), num=100)
        liny = pfunc(linx, popt.x)
        ax.plot(linx, liny, "r-")

        ax.set_xlabel(detections[xi])
        ax.set_ylabel(detections[yi])
        ax.set_xticks(ax.get_xticks().tolist())
        ax.set_xticklabels(ax.get_xticks().tolist(), rotation=20, ha="right")
        ax.set_ylim(bottom=-2000, top=180000)
        ax.set_xlim(left=-2000, right=180000)

    subplotLabel(axs)

    return fig
