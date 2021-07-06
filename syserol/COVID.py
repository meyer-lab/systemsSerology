"""Import Zohar data, tensor formation, plotting raw data."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold


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
    df["patient_ID"] = df["patient_ID"].astype('int32')
    return df.set_index("patient_ID")


def to_slice(subjects, df):
    Rlabels, AgLabels = dimensionLabel3D()
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


def Tensor3D():
    """ Create a 3D Tensor (Antigen, Receptor, Sample in time) """
    df = pbsSubtractOriginal()
    Rlabels, AgLabels = dimensionLabel3D()

    tensor = np.full((len(df), len(AgLabels), len(Rlabels)), np.nan)
    missing = 0

    for rii, recp in enumerate(Rlabels):
        for aii, anti in enumerate(AgLabels):
            try:
                dfAR = df[recp + "_" + anti]
                tensor[:, aii, rii] = dfAR.values
            except KeyError:
                missing += 1

    tensor = np.clip(tensor, 10.0, None)
    tensor = np.log10(tensor)

    # Mean center each measurement
    tensor -= np.nanmean(tensor, axis=0)

    return tensor, np.array(df.index)


def dimensionLabel3D():
    """Returns labels for receptor and antigens, included in the 4D tensor"""
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
        "FcR3B"
    ]
    antigenLabel = ["S", "RBD", "N", "S1", "S2", "S1 Trimer"]
    return receptorLabel, antigenLabel


def COVIDpredict(tfac):
    df = pbsSubtractOriginal()
    subjj = df["group"].isin(["Severe", "Deceased"])

    X = tfac.factors[0][subjj, :]
    y = pd.factorize(df.loc[subjj, "group"])[0]

    kf = KFold(n_splits=5, shuffle=True)
    outt = pd.DataFrame(columns=["fold", "FPR", "TPR"])
    for ii, (train, test) in enumerate(kf.split(X)):
        model = LogisticRegression().fit(X[train], y[train])
        y_score = model.predict_proba(X[test])
        fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
        outt = pd.concat([outt, pd.DataFrame(data={"fold": [ii+1] * len(fpr), "FPR": fpr, "TPR": tpr})])

    return outt


def time_components_df(tfac, condition=None):
    subj = pbsSubtractOriginal()
    df = pd.DataFrame(tfac.factors[0])
    comp_names = ["Comp. " + str((i + 1)) for i in range(tfac.factors[0].shape[1])]
    df.columns = comp_names
    df['days'] = subj['days'].values
    df['group'] = subj['group'].values
    df['week'] = subj['week'].values
    if condition is not None:
        df = df.loc[(subj["group"] == condition).values, :]
    df = df.dropna()
    df = pd.melt(df, id_vars=['days', 'group', 'week'], value_vars=comp_names)
    df.rename(columns={'variable': 'Factors'}, inplace=True)
    return df
