""" Data import and processing. """
from os.path import join, dirname
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "syserol/data/" + name + ".csv"), delimiter=",")

    return data


def importLuminex(antigen=None):
    """ Import the Luminex measurements. Subset if only a specific antigen is needed. """
    df = load_file("data-luminex")
    df = pd.melt(df, id_vars=["subject"])

    if antigen is not None:
        df = df[df["variable"].str.contains(antigen)]
        df["variable"] = df["variable"].str.replace("." + antigen, "")

    return df


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

    subjects = subjects["subject"]
    detections = detections["detection"]
    antigen = antigen["antigen"]

    return subjects, detections, antigen


def createCube():
    """ Import the data and assemble the antigen cube. """
    subjects, detections, antigen = getAxes()
    cube = np.full([len(subjects), len(detections) + 1, len(antigen)], np.nan)

    IGG = importIGG()

    for k, curAnti in enumerate(antigen):
        lumx = importLuminex(curAnti)

        for i, curSubj in enumerate(subjects):
            subjLumx = lumx[lumx["subject"] == curSubj]

            for j, curDet in enumerate(detections):
                if subjLumx["variable"].isin([curDet]).any():
                    cube[i, j, k] = subjLumx.loc[subjLumx["variable"] == curDet, "value"]

    # Add IgG data on the end as another detection
    for i, curSubj in enumerate(subjects):
        subjLumx = IGG[IGG["subject"] == curSubj]

        for k, curAnti in enumerate(antigen):
            if subjLumx["variable"].isin([curAnti]).any():
                cube[i, -1, k] = subjLumx.loc[subjLumx["variable"] == curAnti, "value"]

    return cube
