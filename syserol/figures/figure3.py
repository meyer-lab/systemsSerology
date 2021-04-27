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
    comps = np.arange(1, 2)

    df = pd.concat([pd.DataFrame(evaluate_missing(comps, 15, chords=True)[0]) for _ in range(rep)], axis=1) 
    Q2Xchord = df.mean(axis = 1)
    Q2Xerrors = df.std(axis = 1)
    ax[0].scatter(comps, Q2Xchord)
    ax[0].errorbar(comps, Q2Xchord, yerr = Q2Xerrors)
    ax[0].set_ylabel("Q2X of Imputation")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)

    df = pd.concat([pd.DataFrame(evaluate_missing(comps, 15, chords=False, PCAcompare=True)[0:2]) for _ in range(rep)], axis=0) # PCA will be on odd rows, Tfac on even, each row is a rep, column is a comp
    # CMTFR2X, PCAR2X, _ = evaluate_missing(comps, 15, chords=False, PCAcompare=True)
    CMTFR2X = df.iloc[::2].mean(axis=0)
    CMTFErr = df.iloc[::2].std(axis=0)
    PCAR2X = df.iloc[1::2].mean(axis=0)
    PCAErr = df.iloc[1::2].std(axis=0)
    ax[1].plot(comps, CMTFR2X, ".", label="CMTF")
    ax[1].plot(comps, PCAR2X, ".", label="PCA")
    ax[1].errorbar(comps, CMTFR2X, yerr = CMTFErr)
    ax[1].errorbar(comps, PCAR2X, yerr = PCAErr)
    ax[1].set_ylabel("Q2X of Imputation")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_xticks([x for x in comps])
    ax[1].set_xticklabels([x for x in comps])
    ax[1].set_ylim(0, 1)
    ax[1].legend()

    df = pd.concat([pd.DataFrame(increase_missing(comps,PCAcompare=True)[0:3]) for _ in range(rep)])
    # Rows index 0 are the CMTF Q2Xs across increasing missing values, indexed 1 are PCA. Indexed 2 are the missing fractions. 
    CMTFR2X = df.loc[0].mean(axis=0)
    CMTFErr = df.loc[0].std(axis=0)
    PCAR2X  = df.loc[1].mean(axis=0)
    PCAErr = df.loc[1].std(axis=0)
    print(CMTFErr, PCAErr)
    missing = df.iloc[2]
    ax[2].plot(missing, CMTFR2X, ".", label="CMTF")
    ax[2].plot(missing, PCAR2X, ".", label="PCA")
    ax[2].errorbar(missing, CMTFR2X, yerr=CMTFErr)
    ax[2].errorbar(missing, PCAR2X, yerr=PCAErr)
    ax[2].set_ylabel("Q2X of Imputation")
    ax[2].set_xlabel("Fraction Missing")
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
