"""
This creates Figure 3.
"""
import pandas as pd
import numpy as np
from .common import subplotLabel, getSetup
from ..impute import evaluate_missing, increase_missing


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))
    rep = 10

    comps = np.arange(1, 11)
    df = pd.concat([pd.DataFrame({'Components': comps, 'R2X': evaluate_missing(comps, 15, chords=True)[0]})
                    for _ in range(rep)], axis=0)
    df = df.groupby('Components').agg({'R2X': ['mean', 'std']})
    Q2Xchord = df['R2X']['mean']
    Q2Xerrors = df['R2X']['std']
    ax[0].scatter(comps, Q2Xchord)
    ax[0].errorbar(comps, Q2Xchord, yerr=Q2Xerrors, fmt='none')
    ax[0].set_ylabel("Q2X of Imputation")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)

    df = pd.concat([pd.DataFrame(np.vstack((evaluate_missing(comps, 15, chords=False, PCAcompare=True)[0:2], comps)).T,
                                 columns=['CMTF', 'PCA', 'Components']) for _ in range(rep)], axis=0)
    df = df.groupby(['Components']).agg(['mean', 'std'])

    CMTFR2X = df['CMTF']['mean']
    CMTFErr = df['CMTF']['std']
    PCAR2X = df['PCA']['mean']
    PCAErr = df['PCA']['std']

    ax[1].plot(comps - 0.1, CMTFR2X, ".", label="CMTF")
    ax[1].plot(comps + 0.1, PCAR2X, ".", label="PCA")
    ax[1].errorbar(comps - 0.1, CMTFR2X, yerr=CMTFErr, fmt='none', ecolor='b')
    ax[1].errorbar(comps + 0.1, PCAR2X, yerr=PCAErr, fmt='none', ecolor='darkorange')
    ax[1].set_ylabel("Q2X of Imputation")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_xticks([x for x in comps])
    ax[1].set_xticklabels([x for x in comps])
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc=4)

    comps = np.arange(10, 11)
    df = pd.concat([pd.DataFrame(np.vstack(increase_missing(comps, PCAcompare=True)[0:3]).T,
                                 columns=['CMTF', 'PCA', 'missing']) for _ in range(rep)])
    df = df.groupby(['missing']).agg(['mean', 'std']).reset_index()

    missing = df['missing']
    CMTFR2X = df['CMTF']['mean']
    CMTFErr = df['CMTF']['std']
    PCAR2X = df['PCA']['mean']
    PCAErr = df['PCA']['std']

    ax[2].plot(missing, CMTFR2X, ".", label="CMTF")
    ax[2].plot(missing, PCAR2X, ".", label="PCA")
    ax[2].errorbar(missing, CMTFR2X, yerr=CMTFErr, fmt='none', ecolor='b')
    ax[2].errorbar(missing, PCAR2X, yerr=PCAErr, fmt='none', ecolor='darkorange')
    ax[2].set_yscale("log")
    ax[2].set_ylabel("Q2X of Imputation")
    ax[2].set_xlabel("Fraction Missing")
    ax[2].set_xlim(0.4, 1)
    ax[2].set_ylim(0, 1)
    ax[2].legend(loc=3)

    # Add subplot labels
    subplotLabel(ax)

    return f
