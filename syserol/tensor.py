"""
Tensor decomposition methods
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from jax.config import config
from scipy.optimize import minimize
from numpy.random import randn
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.kruskal_tensor import KruskalTensor, kruskal_normalise
from .dataImport import createCube

tl.set_backend('jax')
config.update("jax_enable_x64", True)

def calcR2X(data, factor):
    """ Calculate R2X. """
    tensorErr = np.nanvar(tl.kruskal_to_tensor(factor) - data)
    return 1.0 - tensorErr / np.nanvar(data)


def reorient_factors(tensorFac, matrixFac):
    """ This function ensures that factors are negative on at most one direction. """
    for jj in range(1, len(tensorFac)):
        # Calculate the sign of the current factor in each component
        means = np.sign(np.mean(tensorFac[jj], axis=0))

        # Update both the current and last factor
        tensorFac[0] *= means[np.newaxis, :]
        matrixFac[0] *= means[np.newaxis, :]
        matrixFac[1] *= means[np.newaxis, :]
        tensorFac[jj] *= means[np.newaxis, :]
    return tensorFac, matrixFac


def buildTensors(pIn, tensor, matrix, r):
    """ Use parameter vector to build kruskal tensors. """
    assert tensor.shape[0] == matrix.shape[0]
    nA = tensor.shape[0]*r
    nB = tensor.shape[1]*r
    nC = tensor.shape[2]*r

    A = jnp.reshape(pIn[:nA], (tensor.shape[0], r))
    B = jnp.reshape(pIn[nA:nA+nB], (tensor.shape[1], r))
    C = jnp.reshape(pIn[nA+nB:nA+nB+nC], (tensor.shape[2], r))
    G = jnp.reshape(pIn[nA+nB+nC:], (matrix.shape[1], r))

    return KruskalTensor((None, [A, B, C])), KruskalTensor((None, [A, G]))


def cost(pIn, tensor, matrix, tmask, mmask, r):
    tensF, matF = buildTensors(pIn, tensor, matrix, r)
    tDiff = (tl.kruskal_to_tensor(tensF) - tensor) * (1 - tmask)
    mDiff = (tl.kruskal_to_tensor(matF) - matrix) * (1 - mmask)
    cost = jnp.linalg.norm(tDiff) # Tensor cost
    cost += jnp.linalg.norm(mDiff) # Matrix cost
    cost += 10 * jnp.linalg.norm(pIn, ord=9)
    return cost


def perform_CMTF(tensorIn=None, matrixIn=None, r=8):
    """ Perform CMTF decomposition. """
    if tensorIn is None:
        tensorIn, matrixIn = createCube()

    tmask = np.isnan(tensorIn)
    mmask = np.isnan(matrixIn)
    tensorIn[tmask] = 0.0
    matrixIn[mmask] = 0.0

    cost_jax = jit(cost, static_argnums=(1, 2, 3, 4, 5))
    cost_grad = jit(grad(cost, 0), static_argnums=(1, 2, 3, 4, 5))

    facInit = parafac(tensorIn, r, mask=tmask, n_iter_max=30, orthogonalise=True)
    x0 = np.concatenate((np.ravel(facInit.factors[0]), np.ravel(facInit.factors[1]), np.ravel(facInit.factors[2])))
    x0 = np.concatenate((x0, randn(matrixIn.shape[1] * r)))

    res = minimize(cost_jax, x0, method='CG', jac=cost_grad, args=(tensorIn, matrixIn, tmask, mmask, r), options={"gtol": 1e-7, "disp": True, "maxiter": 5000})
    tensorFac, matrixFac = buildTensors(res.x, tensorIn, matrixIn, r)
    tensorFac = kruskal_normalise(tensorFac)
    matrixFac = kruskal_normalise(matrixFac)

    # Reorient the later tensor factors
    tensorFac.factors, matrixFac.factors = reorient_factors(tensorFac.factors, matrixFac.factors)

    tensor_R2XX = calcR2X(tensorIn, tensorFac)
    matrix_R2XX = calcR2X(matrixIn, matrixFac)

    for ii in range(3):
        tensorFac.factors[ii] = np.array(tensorFac.factors[ii])
    for ii in range(2):
        matrixFac.factors[ii] = np.array(matrixFac.factors[ii])

    return tensorFac, matrixFac, tensor_R2XX, matrix_R2XX
