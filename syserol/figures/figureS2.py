"""
This creates Figure S2.
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

        ax.scatter(data[0, :], data[1, :], s=0.3)

        func = lambda x: data[0, :] * x[0] / (data[0, :] * (1 - x[1]) + x[2]) - data[1, :]
        diffscale = np.mean(data[1, :]) / np.mean(data[0, :])
        popt = least_squares(func, x0 = [diffscale, 1.0, 1.0], jac="3-point")
        xout = popt.x
        linx = np.linspace(0.0, np.amax(data[0, :]), num=100)
        liny = linx * xout[0] / (linx * (1 - xout[1]) + xout[2])
        ax.plot(linx, liny, 'r-')

        ax.set_xlabel(detections[xi])
        ax.set_ylabel(detections[yi])

        ax.set_ylim(bottom = -2000)
        ax.set_xlim(left = -2000)

    subplotLabel(axs)

    return fig
