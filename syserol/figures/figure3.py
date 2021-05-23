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

    comps = np.arange(1, 6)

    try:
        chords_df = pd.read_csv('syserol/data/fig3_chords_df.csv')
    except:
        print("Building chords...")
        # Imputing chords dataframe
        chords_df = pd.concat([pd.DataFrame({'Components': comps, 'R2X': evaluate_missing(comps, 15, chords=True)[0]})
                                for _ in range(rep)], axis=0)
        chords_df.to_csv('syserol/data/fig3_chords_df.csv', index=False)
    chords_df = chords_df.groupby('Components').agg({'R2X': ['mean', 'std']})


    try:
        single_df = pd.read_csv('syserol/data/fig3_single_df.csv')
    except:
        print("Building singles...")
        # Single imputations dataframe
        single_df = pd.concat([pd.DataFrame(np.vstack((evaluate_missing(comps, 15, chords=False)[0:2], comps)).T,
                                 columns=['CMTF', 'PCA', 'Components']) for _ in range(rep)], axis=0)
        single_df.to_csv('syserol/data/fig3_single_df.csv', index=False)
    single_df = single_df.groupby(['Components']).agg(['mean', 'std'])


    try:
        increasing_df = pd.read_csv('syserol/data/fig3_increasing_df.csv')
    except:
        print("Building increasing...")
        # Increasing imputations dataframe
        rep = 1
        comps = 5
        increasing_df = pd.concat([pd.DataFrame(np.vstack(increase_missing(comps)[0:3]).T,
                                    columns=['CMTF', 'PCA', 'missing']) for _ in range(rep)])
        increasing_df.to_csv('syserol/data/fig3_increasing_df.csv', index=False)


    Q2Xchord = chords_df['R2X']['mean']
    Q2Xerrors = chords_df['R2X']['std']
    ax[0].scatter(comps, Q2Xchord)
    ax[0].errorbar(comps, Q2Xchord, yerr=Q2Xerrors, fmt='none')
    ax[0].set_ylabel("Q2X of Imputation")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)

    CMTFR2X = single_df['CMTF']['mean']
    CMTFErr = single_df['CMTF']['std']
    PCAR2X = single_df['PCA']['mean']
    PCAErr = single_df['PCA']['std']
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

    increasing_df = increasing_df.groupby(['missing']).agg(['mean', 'std']).reset_index()
    missing = increasing_df['missing']
    CMTFR2X = increasing_df['CMTF']['mean']
    CMTFErr = increasing_df['CMTF']['std']
    PCAR2X = increasing_df['PCA']['mean']
    PCAErr = increasing_df['PCA']['std']
    ax[2].plot(missing, CMTFR2X, ".", label="CMTF")
    ax[2].plot(missing, PCAR2X, ".", label="PCA")
    if np.any(PCAErr) and np.any(CMTFErr):
        ax[2].errorbar(missing, CMTFR2X, yerr=CMTFErr, fmt='none', ecolor='b')
        ax[2].errorbar(missing, PCAR2X, yerr=PCAErr, fmt='none', ecolor='darkorange')
    ax[2].set_ylabel("Q2X of Imputation")
    ax[2].set_xlabel("Fraction Missing")
    ax[2].set_xlim(0.4, 1)
    ax[2].set_ylim(0.4, 1)
    ax[2].legend(loc=3)

    # Add subplot labels
    subplotLabel(ax)

    return f
