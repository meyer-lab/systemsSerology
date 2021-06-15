"""Import Zohar data, tensor formation, plotting raw data."""
import numpy as np
import pandas as pd


def pbsSubtractOriginal():
    """ Paper Background subtract, will keep all rows for any confusing result. """
    Cov = pd.read_csv("syserol/data/ZoharCovData.csv", index_col=0)
    # 23 (0-> 23) is the start of IgG1_S
    Demographics = Cov.iloc[:, 0:23]
    Serology = Cov.iloc[:, 23::]
    Serology -= Serology.loc["PBS"].values.squeeze()
    df = pd.concat([Demographics, Serology], axis=1)
    df = df.loc[np.isfinite(df["patient_ID"]), :]
    df["week"] = np.array(df["days"] // 7 + 1.0, dtype=int)
    return df.set_index("patient_ID")


def to_slice(subjects, df):
    _, Rlabels, AgLabels = dimensionLabel4D()
    tensor = np.full((len(subjects), len(AgLabels), len(Rlabels)), np.nan)
    missing = 0

    for rii, recp in enumerate(Rlabels):
        for aii, anti in enumerate(AgLabels):
            try:
                dfAR = df[recp + "_" + anti]
                dfAR = dfAR.groupby(by="patient_ID").mean()
                dfAR = dfAR.reindex(subjects)
                tensor[:, aii, rii] = dfAR.values
            except KeyError:
                #print(recp + "_" + anti)
                missing += 1

    return tensor


def Tensor4D():
    """ Create a 4D Tensor (Time (Weeks), Antigen, Receptor, Sample) """
    df = pbsSubtractOriginal()
    subjects = pd.unique(df.index)

    # 3D Tensor for Week 1, 2, 3
    tensors = [to_slice(subjects, df.loc[df["week"] == ii, :]) for ii in range(1, 5)]

    # Create Tensor 4
    tensor = np.stack(tensors, axis=3)
    idxs = np.any(np.isfinite(tensor), axis=(1,2,3))

    return tensor[idxs, :], subjects[idxs]


def dimensionLabel4D():
    """Returns labels for receptor and antigens, included in the 4D tensor"""
    weekLabel = ["Week 1", "Week 2", "Week 3", "Week 4"]
    receptorLabel = [
        "IgG1",
        "IgG2",
        "IgG3",
        "IgA1",
        "IgA2",
        "IgM",
        "FcRalpha",
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
    antigenLabel = ["S", "RBD", "N", "S1", "S2", "S1 Trimer", "flu_mix", "NL63", "HKU1"]
    return weekLabel, receptorLabel, antigenLabel
