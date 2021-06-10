""" This makes Figure 6. Plot of R2X values"""
import numpy as np
import seaborn as sns
from ..COVID import Tensor4D, dimensionLabel4D
from ..tensor import perform_CMTF
from .common import getSetup, subplotLabel

def makeFigure():
    ax, f = getSetup((7, 9), (3, 2))
    comps = np.arange(1, 11)

    tensor, _ = Tensor4D()
    R2X = [perform_CMTF(tensor, r=cc).R2X for cc in comps]

    ax[0].axis("off")
    ax[1].scatter(comps, R2X, color="b")
    ax[1].set_ylabel("R2X")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_xticks([x for x in comps])
    ax[1].set_xticklabels([x for x in comps])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0.0, np.amax(comps) + 0.5)

    ## Colormap

    weeklabels, Rlabels, agLabels = dimensionLabel4D()
    tfac = perform_CMTF(tensor, r=6)

    components = [str(ii + 1) for ii in range(tfac.rank)]
    comp_plot(tfac.factors[0], components, False, "Subjects", ax[2])
    comp_plot(tfac.factors[1], components, agLabels, "Antigens", ax[3])
    comp_plot(tfac.factors[2], components, Rlabels, "Receptors", ax[4])
    comp_plot(tfac.factors[3], components, weeklabels, "Weeks", ax[5])

    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax):
    """ Creates heatmap plots for each input dimension by component. """
    sns.heatmap(factors, cmap="PiYG", center=0, xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel(plotLabel)
