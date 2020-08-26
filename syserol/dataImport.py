""" Data import and processing. """
from functools import lru_cache
from os.path import join, dirname
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(
        join(path_here, "syserol/data/" + name + ".csv"), delimiter=",", comment="#"
    )

    return data


def importLuminex(antigen=None):
    """ Import the Luminex measurements. Subset if only a specific antigen is needed. """
    df = load_file("data-luminex")
    df = pd.melt(df, id_vars=["subject"])

    if antigen is not None:
        df = df[df["variable"].str.contains(antigen)]
        df["variable"] = df["variable"].str.replace("." + antigen, "")

        # Filter out bad antigen matches
        df = df[~df["variable"].str.contains("235")]
        df = df[~df["variable"].str.contains("244")]
        df = df[~df["variable"].str.contains("Kif")]
        df = df[~df["variable"].str.contains("delta3711")]

    return df


def importGlycan():
    """ Import the glycan measurements. """
    df = load_file("data-glycan-gp120")
    dfAxis = load_file("meta-glycans")
    df = pd.melt(df, id_vars=["subject"])

    glycan = dfAxis["glycan"].to_list()

    return glycan, df


def importIGG():
    """ Import the IgG measurements. """
    df = load_file("data-luminex-igg")
    df = pd.melt(df, id_vars=["subject"])

    df["variable"] = df["variable"].str.replace("IgG.", "")

    return df


def getAxes():
    """ Get each of the axes over which the data is measured. """
    subjects = load_file("meta-subjects")
    detections = load_file("meta-detections")
    antigen = load_file("meta-antigens")

    subjects = subjects["subject"].to_list()
    detections = detections["detection"].to_list()
    antigen = antigen["antigen"].to_list()

    return subjects, detections, antigen


functions = ["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]


def importFunction():
    """ Import functional data. """
    subjects, _, _ = getAxes()
    df = load_file("data-function")
    df_a = pd.DataFrame({"subject": subjects})

    df = df_a.merge(df, on="subject", how="left")

    idnum = [0, 1, 2, 3, 4, 5]
    mapped = dict(zip(functions, idnum))

    return df, mapped


@lru_cache()
def createCube():
    """ Import the data and assemble the antigen cube. """
    subjects, detections, antigen = getAxes()
    cube = np.full([len(subjects), len(detections), len(antigen)], np.nan)

    IGG = importIGG()
    glycan, dfGlycan = importGlycan()
    glyCube = np.full([len(subjects), len(glycan)], np.nan)

    for k, curAnti in enumerate(antigen):
        lumx = importLuminex(curAnti)

        for i, curSubj in enumerate(subjects):
            subjLumx = lumx[lumx["subject"] == curSubj]
            subjGly = dfGlycan[dfGlycan["subject"] == curSubj]

            for _, row in subjGly.iterrows():
                j = glycan.index(row["variable"])
                glyCube[i, j] = row["value"]

            for _, row in subjLumx.iterrows():
                j = detections.index(row["variable"])
                cube[i, j, k] = row["value"]

    # Add IgG data on the end as another detection
    for i, curSubj in enumerate(subjects):
        subjLumx = IGG[IGG["subject"] == curSubj]

        for _, row in subjLumx.iterrows():
            k = antigen.index(row["variable"])
            cube[i, -1, k] = row["value"]

    # We probably want to do some sort of normalization, but I'm not sure what yet
    cube = cube - np.nanmean(cube, axis=(0, 2))[np.newaxis, :, np.newaxis]
    cube = cube / np.nanstd(cube, axis=(0, 2))[np.newaxis, :, np.newaxis]
    glyCube = glyCube - np.nanmean(glyCube, axis=0)[np.newaxis, :]
    glyCube = glyCube / np.nanstd(glyCube, axis=0)[np.newaxis, :]

    print("Missingness fraction: " + str(np.mean(np.isnan(cube))))

    # Check that there are no slices with completely missing data
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(cube), axis=(1, 2)))

    return cube, glyCube
