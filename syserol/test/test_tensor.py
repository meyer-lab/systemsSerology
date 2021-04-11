"""
Unit test file.
"""
import numpy as np
import pandas as pd
import pytest
from tensorly.cp_tensor import _validate_cp_tensor
from ..tensor import perform_CMTF, delete_component
from ..regression import make_regression_df
from ..classify import class_predictions_df


def test_R2X():
    """ Test to ensure R2X for higher components is larger. """
    arr = []
    for i in range(1, 5):
        facT, facM, tensorR2X = perform_CMTF(r=i)
        assert np.all(np.isfinite(facT.factors[0]))
        assert np.all(np.isfinite(facT.factors[1]))
        assert np.all(np.isfinite(facT.factors[2]))
        assert np.allclose(facT.factors[0], facM.factors[0])
        arr.append(tensorR2X)
    for j in range(len(arr) - 1):
        assert arr[j] < arr[j + 1]
    # confirm R2X is >= 0 and <=1
    assert np.min(arr) >= 0
    assert np.max(arr) <= 1


def test_delete():
    """ Test deleting a component results in a valid tensor. """
    facT, facM, _ = perform_CMTF(r=5)
    facT = delete_component(facT, 2)
    facM = delete_component(facM, 3)

    _validate_cp_tensor(facT)
    _validate_cp_tensor(facM)


@pytest.mark.parametrize("resample", [False, True])
def test_prediction_dfs(resample):
    """ Test that we can assemble the prediction dataframes. """
    tFac = perform_CMTF(r=3)[0]

    # Function Prediction DataFrame, Figure 5A
    functions_df = make_regression_df(tFac[1][0], resample=resample)

    # Class Predictions DataFrame, Figure 5B
    classes = class_predictions_df(tFac[1][0], resample=resample)

    assert isinstance(functions_df, pd.DataFrame)
    assert isinstance(classes, pd.DataFrame)
