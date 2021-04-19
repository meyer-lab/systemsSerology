"""
Unit test file.
"""
import numpy as np
import pandas as pd
import pytest
from tensorly.cp_tensor import _validate_cp_tensor
from ..tensor import perform_CMTF, delete_component, calcR2X
from ..regression import make_regression_df
from ..classify import class_predictions_df
from ..dataImport import createCube


def test_R2X():
    """ Test to ensure R2X for higher components is larger. """
    arr = []
    for i in range(1, 5):
        facT = perform_CMTF(r=i)
        assert np.all(np.isfinite(facT.factors[0]))
        assert np.all(np.isfinite(facT.factors[1]))
        assert np.all(np.isfinite(facT.factors[2]))
        assert np.all(np.isfinite(facT.mFactor))
        arr.append(facT.R2X)
    for j in range(len(arr) - 1):
        assert arr[j] < arr[j + 1]
    # confirm R2X is >= 0 and <=1
    assert np.min(arr) >= 0
    assert np.max(arr) <= 1


def test_delete():
    """ Test deleting a component results in a valid tensor. """
    tOrig, mOrig = createCube()
    facT = perform_CMTF(r=10)

    fullR2X = calcR2X(tOrig, mOrig, facT)

    for ii in range(facT.rank):
        facTdel = delete_component(facT, ii)
        _validate_cp_tensor(facTdel)

        delR2X = calcR2X(tOrig, mOrig, facTdel)

        assert delR2X < fullR2X
        assert delR2X > -1.0


@pytest.mark.parametrize("resample", [False, True])
def test_prediction_dfs(resample):
    """ Test that we can assemble the prediction dataframes. """
    tFac = perform_CMTF(r=3)

    # Function Prediction DataFrame, Figure 5A
    functions_df = make_regression_df(tFac[1][0], resample=resample)

    # Class Predictions DataFrame, Figure 5B
    classes = class_predictions_df(tFac[1][0], resample=resample)

    assert isinstance(functions_df, pd.DataFrame)
    assert isinstance(classes, pd.DataFrame)
