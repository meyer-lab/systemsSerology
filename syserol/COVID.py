"""Import Zohar data, tensor formation, plotting raw data."""
import numpy as np
import pandas as pd


def pbsSubtractOriginal():
    """Paper Background subtract, will keep all rows for any confusing result"""
    Cov = pd.read_csv("syserol/data/ZoharCovData.csv")
    # 23 (0-> 23) is the start of IgG1_S
    r, c = Cov.shape
    Demographics = Cov.iloc[:, 0:23]
    Serology = Cov.iloc[:, 23:c]
    pbsRow = Serology.iloc[[r - 4]]
    BackgroundSub = pd.DataFrame(Serology.values - pbsRow.values, columns=Serology.columns)
    Complete = pd.concat([Demographics, BackgroundSub], axis=1)
    Complete = Complete.loc[np.isfinite(Complete["patient_ID"]), :]
    return Complete


def to_slice(subjects, df):
    df.set_index('patient_ID')
    df = df.reindex(subjects)

    _, receptorslabels, antigenslabels = dimensionLabel4D()
    tensor = np.zeros((len(antigenslabels), len(subjects), len(receptorslabels)))

    for rii, recp in enumerate(receptorslabels):
        dfR = df.loc[:, df.columns.str.contains(recp)]

        for aii, anti in enumerate(antigenslabels):
            dfAR = dfR.loc[:, dfR.columns.str.endswith(anti)]

            if dfAR.size == 0:
                tensor[aii, :, rii] = np.nan
            else:
                tensor[aii, :, rii] = np.squeeze(dfAR.values)

    return tensor


def Tensor4D():
    """ Create a 4D Tensor (Time (Weeks), Antigen, Receptor, Sample) """
    df = pbsSubtractOriginal()
    subjects = pd.unique(df["patient_ID"])

    # 3D Tensor for Week 1, 2, 3
    tensor1 = to_slice(subjects, df.loc[df["days"] < 7, :])
    tensor2 = to_slice(subjects, df.loc[(df["days"] > 7) & (df["days"] < 15), :])
    tensor3 = to_slice(subjects, df.loc[(df["days"] > 14) & (df["days"] < 22), :])
    tensor4 = to_slice(subjects, df.loc[(df["days"] > 21) & (df["days"] < 29), :])
    tensor5 = to_slice(subjects, df.loc[(df["days"] > 28) & (df["days"] < 36), :])

    assert tensor1.shape == tensor2.shape
    assert tensor1.shape == tensor3.shape
    assert tensor1.shape == tensor4.shape
    assert tensor1.shape == tensor5.shape

    # Create Tensor 4
    return np.stack((tensor1, tensor2, tensor3, tensor4, tensor5)), subjects


def dimensionLabel4D():
    """Returns labels for receptor and antigens, included in the 4D tensor"""
    weekLabel = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
    receptorLabel = [
        "IgG1",
        "IgG2",
        "IgG3",
        "IgG4"
        "IgA1",
        "IgA2",
        "IgM",
        "FcR_alpha",
        "FcR2A",
        "FcR2B",
        "FcR3A",
        "FcR3B",
        "ADCP",
        "ADNP",
        "ADCD",
        "ADNKA_CD107a",
        "ADNKA_MIP1b",
    ]
    antigenLabel = ["S", "RBD", "N", "S1", "S2", "S1 Trimer", "Flu", "NL63", "HKU1"]
    return weekLabel, receptorLabel, antigenLabel









