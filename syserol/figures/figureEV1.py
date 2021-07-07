import numpy as np
from sklearn.linear_model import LogisticRegression
from syserol.tensor import perform_CMTF
from syserol.dataImport import load_file
from syserol.figures.common import getSetup, subplotLabel
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))

    tFac = perform_CMTF()
    X = tFac.factors[0]

    df = load_file("meta-subjects")
    Y1 = (df["class.cp"] == "controller").astype(int)  # control 1, progress 0
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic 1, nonviremic 0

    make_decision_plot(ax[0], X, Y2, title="Viremic/Nonviremic", black="Nonviremic",
                       white="Viremic", xaxis=1, yaxis=2)
    make_decision_plot(ax[1], X, Y1, title="Controller/Progressor", black="Progressor",
                       white="Controller", xaxis=1, yaxis=4)

    # Add subplot labels
    subplotLabel(ax)

    return f


def make_decision_plot(ax, X, y, title, black, white, xaxis, yaxis):
    """ Make one decision plot. Only works with 2D data. """
    X = X[:, np.array([xaxis-1, yaxis-1])]
    xx = np.linspace(-1.05, 1.05, 100)
    xx, yy = np.meshgrid(xx, xx.T)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    classifier = LogisticRegression(penalty="none")

    # Fit and get model probabilities
    classifier.fit(X, y)
    probas = classifier.predict_proba(Xfull)

    ax.imshow(probas[:, 0].reshape((100, 100)), extent=(-1.05,
              1.05, -1.05, 1.05), origin='lower', cmap="crest")
    blk = ax.scatter(X[y == 0, 0], X[y == 0, 1],
                     marker='.', c='k', edgecolor='k')
    wht = ax.scatter(X[y == 1, 0], X[y == 1, 1],
                     marker='.', c='w', edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel("Component " + str(xaxis))
    ax.set_ylabel("Component " + str(yaxis))
    ax.legend([Patch(facecolor='black', edgecolor='grey'), Patch(
        facecolor='w', edgecolor='grey'), Patch(
        facecolor=sns.color_palette('crest'), edgecolor='grey'), Patch(
        facecolor='w', edgecolor='grey'), ], [black, white], framealpha=0.99)
