import numpy as np

import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.kruskal_tensor import KruskalTensor


# Authors: Isabell Lehmann <isabell.lehmann94@outlook.de>

# License: BSD 3 clause


def solve_least_squares(A, B):
    # solve ||B-AX||^2 (AX = B -> X = A^+ @ B), with A^+: pseudo inverse
    return np.transpose(np.linalg.lstsq(A, B, rcond=-1)[0])


def coupled_matrix_tensor_3d_factorization(X, Y, rank, mask_3d=None, mask_matrix=None, init, verbose=False):
    """
    Calculates a coupled matrix and tensor factorization of 3rd order tensor and matrix which are
    coupled in first mode.

    Assume you have tensor_3d = [[lambda; A, B, C]] and matrix = [[gamma; A, V]], which are
    coupled in 1st mode. With coupled matrix and tensor factorization (CTMF), the normalized
    factor matrices A, B, C for the CP decomposition of X, the normalized matrix V and the
    weights lambda_ and gamma are found. This implementation only works for a coupling in the
    first mode.

    Solution is found via alternating least squares (ALS) as described in Figure 5 of
    @article{acar2011all,
      title={All-at-once optimization for coupled matrix and tensor factorizations},
      author={Acar, Evrim and Kolda, Tamara G and Dunlavy, Daniel M},
      journal={arXiv preprint arXiv:1105.3422},
      year={2011}
    }

    Notes
    -----
    In the paper, the columns of the factor matrices are not normalized and therefore weights are
    not included in the algorithm.

    Parameters
    ----------
    tensor_3d : tl.tensor or Kruskal tensor
        3rd order tensor X = [[A, B, C]]
    matrix : tl.tensor or Kruskal tensor
        matrix that is coupled with tensor in first mode: Y = [[A, V]]
    rank : int
        rank for CP decomposition of X

    Returns
    -------
    tensor_3d_pred, matrix_pred : Kruskal tensors
        tensor_3d_pred = [[lambda; A,B,C]], matrix_pred = [[gamma; A,V]]
    """

    assert tl.is_tensor(X)
    assert tl.is_tensor(Y)

    # initialize values
    A, B, C = kruskal.factors

    V = solve_least_squares(A, Y)
    gamma = np.linalg.norm(V, axis=0)
    V /= gamma
    lambda_ = tl.ones(rank)

    # Mask both based on starting point
    if mask_3d is not None:
        X = X * mask_3d + tl.kruskal_tensor.kruskal_to_tensor((lambda_, [A, B, C])) * (1 - mask_3d)
    if mask_matrix is not None:
        Y = Y * mask_matrix + tl.kruskal_tensor.kruskal_to_tensor((gamma, [A, V])) * (1 - mask_matrix)

    error_old = np.linalg.norm(X - tl.kruskal_tensor.kruskal_to_tensor((lambda_, [A, B, C]))) + np.linalg.norm(Y - tl.kruskal_tensor.kruskal_to_tensor((gamma, [A, V])))

    # alternating least squares
    # note that the order of the khatri rao product is reversed since tl.unfold has another order
    # than assumed in paper
    for iteration in range(10 ** 4):
        A = solve_least_squares(
            np.transpose(np.concatenate((np.dot(np.diag(lambda_), np.transpose(khatri_rao([B, C]))), np.dot(np.diag(gamma), np.transpose(V))), axis=1)),
            np.transpose(np.concatenate((tl.unfold(X, 0), Y), axis=1)),
        )
        norm_A = np.linalg.norm(A, axis=0)
        A /= norm_A
        lambda_ *= norm_A
        gamma *= norm_A

        B = solve_least_squares(np.dot(khatri_rao([A, C]), np.diag(lambda_)), np.transpose(tl.unfold(X, 1)))
        norm_B = np.linalg.norm(B, axis=0)
        B /= norm_B
        lambda_ *= norm_B

        C = solve_least_squares(np.dot(khatri_rao([A, B]), np.diag(lambda_)), np.transpose(tl.unfold(X, 2)))
        norm_C = np.linalg.norm(C, axis=0)
        C /= norm_C
        lambda_ *= norm_C

        V = solve_least_squares(np.dot(A, np.diag(gamma)), Y)
        norm_V = np.linalg.norm(V, axis=0)
        V /= norm_V
        gamma *= norm_V

        # Perform masking
        if mask_3d is not None:
            X = X * mask_3d + tl.kruskal_tensor.kruskal_to_tensor((lambda_, [A, B, C])) * (1 - mask_3d)
        if mask_matrix is not None:
            Y = Y * mask_matrix + tl.kruskal_tensor.kruskal_to_tensor((gamma, [A, V])) * (1 - mask_matrix)

        error_new = np.linalg.norm(X - tl.kruskal_tensor.kruskal_to_tensor((lambda_, [A, B, C]))) + np.linalg.norm(Y - tl.kruskal_tensor.kruskal_to_tensor((gamma, [A, V])))
        decr = error_old - error_new

        if verbose and iteration % 10 == 0:
            print("iteration {}, reconstruction error: {}, decrease = {}".format(iteration, error_new, decr))

        if iteration > 0 and (tl.abs(decr) <= 1e-9 or error_new < 1e-5):
            break

        error_old = error_new

    return KruskalTensor((lambda_, [A, B, C])), KruskalTensor((gamma, [A, V]))
