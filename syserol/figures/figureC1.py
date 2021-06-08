import seaborn as sns
from ..COVID import Tensor4D, dimensionLabel4D
from ..tensor import cp_decomp
from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((6, 7), (4, 1))
    tensor, _ = Tensor4D()
    weeklabels, receptorslabels, antigenslabels = dimensionLabel4D()

    assert tensor.shape[0] == len(weeklabels)
    assert tensor.shape[1] == len(antigenslabels)
    assert tensor.shape[3] == len(receptorslabels)

    _, tfac = cp_decomp(tensor, 6)

    components = [str(ii + 1) for ii in range(tfac.rank)]
    comp_plot(tfac.factors[0], components, weeklabels, "Weeks", ax[0])
    comp_plot(tfac.factors[1], components, antigenslabels, "Antigens", ax[1])
    comp_plot(tfac.factors[2], components, False, "Subjects", ax[2])
    comp_plot(tfac.factors[3], components, receptorslabels, "Receptors", ax[3])
    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax):
    """Creates heatmap plots for each input dimension by component"""
    sns.heatmap(factors, cmap="PiYG", center=0, xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel(plotLabel)
    ax.set_title(plotLabel + " by Component")
