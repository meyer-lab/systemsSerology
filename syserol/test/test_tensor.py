"""
Unit test file.
"""
import numpy as np
import pandas as pd
import pytest
import tensorly as tl
from tensorly.cp_tensor import _validate_cp_tensor
from tensorly.random import random_cp
from ..tensor import perform_CMTF, delete_component, calcR2X, buildGlycan, sort_factors, cp_to_vec, buildTensors, cp_decomp
from ..regression import make_regression_df
from ..classify import class_predictions_df
from ..dataImport import createCube
from ..COVID import Tensor4D


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


def test_cp():
    """ Test that the CP decomposition code works. """
    tensor, _ = Tensor4D()
    facT = cp_decomp(tensor, 6)


def test_delete():
    """ Test deleting a component results in a valid tensor. """
    tOrig, mOrig = createCube()
    facT = perform_CMTF(r=4)

    fullR2X = calcR2X(facT, tOrig, mOrig)

    for ii in range(facT.rank):
        facTdel = delete_component(facT, ii)
        _validate_cp_tensor(facTdel)

        delR2X = calcR2X(facTdel, tOrig, mOrig)

        assert delR2X < fullR2X
        assert delR2X > -1.0


def test_sort():
    """ Test that sorting does not affect anything. """
    tOrig, mOrig = createCube()

    tFac = random_cp(tOrig.shape, 3)
    tFac.mFactor = np.random.randn(mOrig.shape[1], 3)
    tFac.mWeights = np.ones(3)

    R2X = calcR2X(tFac, tOrig, mOrig)
    tRec = tl.cp_to_tensor(tFac)
    mRec = buildGlycan(tFac)

    tFac = sort_factors(tFac)
    sR2X = calcR2X(tFac, tOrig, mOrig)
    stRec = tl.cp_to_tensor(tFac)
    smRec = buildGlycan(tFac)

    np.testing.assert_allclose(R2X, sR2X)
    np.testing.assert_allclose(tRec, stRec)
    np.testing.assert_allclose(mRec, smRec)


def test_vec():
    """ Test that making a vector and then reconstructing works. """
    tOrig, mOrig = createCube()

    tFac = random_cp(tOrig.shape, 3, normalise_factors=False)
    tFac.mFactor = np.random.randn(mOrig.shape[1], 3)
    tFac.mWeights = np.ones(3)

    tFacNew = buildTensors(cp_to_vec(tFac), tOrig, mOrig, tFac.rank)

    for ii in range(3):
        np.testing.assert_allclose(tFac.factors[ii], tFacNew.factors[ii])

    np.testing.assert_allclose(calcR2X(tFac, tOrig, mOrig), calcR2X(tFacNew, tOrig, mOrig))


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
