"""
This creates Figure 3.
"""
import pandas as pd
import numpy as np
from .common import subplotLabel, getSetup
from ..impute import evaluate_missing


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 3), (1, 2))
    rep = 10

    comps = np.arange(1, 11)

    try:
        chords_df = pd.read_csv('syserol/data/fig3_chords_df.csv')
    except FileNotFoundError:
        print("Building chords...")
        # Imputing chords dataframe
        chords_df = pd.concat([pd.DataFrame({'Components': comps, 'R2X': evaluate_missing(comps, 15, chords=True)[0]})
                               for _ in range(rep)], axis=0)
        chords_df.to_csv('syserol/data/fig3_chords_df.csv', index=False)
    chords_df = chords_df.groupby('Components').agg({'R2X': ['mean', 'sem']})

    try:
        single_df = pd.read_csv('syserol/data/fig3_single_df.csv')
    except FileNotFoundError:
        print("Building singles...")
        # Single imputations dataframe
        single_df = pd.concat([pd.DataFrame(np.vstack((evaluate_missing(comps, 15, chords=False)[0:2], comps)).T,
                                            columns=['CMTF', 'PCA', 'Components']) for _ in range(rep)], axis=0)
        single_df.to_csv('syserol/data/fig3_single_df.csv', index=False)
    single_df = single_df.groupby(['Components']).agg(['mean', 'sem'])


    Q2Xchord = chords_df['R2X']['mean']
    Q2Xerrors = chords_df['R2X']['sem']
    ax[0].scatter(comps, Q2Xchord)
    ax[0].errorbar(comps, Q2Xchord, yerr=Q2Xerrors, fmt='none')
    ax[0].set_ylabel("Q2X of Imputation")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)

    CMTFR2X = single_df['CMTF']['mean']
    CMTFErr = single_df['CMTF']['sem']
    PCAR2X = single_df['PCA']['mean']
    PCAErr = single_df['PCA']['sem']
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

    # Add subplot labels
    subplotLabel(ax)

    return f
