import numpy as np

import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.kruskal_tensor import KruskalTensor


# Authors: Isabell Lehmann <isabell.lehmann94@outlook.de>

# License: BSD 3 clause


def solve_least_squares(A, B):
    # solve ||B-AX||^2 (AX = B -> X = A^+ @ B), with A^+: pseudo inverse
    return np.transpose(np.linalg.lstsq(A, B, rcond=-1)[0])


def coupled_matrix_tensor_3d_factorization(
    X, Y, mask_3d, mask_matrix, init, verbose=False
):
    """
    Calculates a coupled matrix and tensor factorization of 3rd order tensor and matrix which are
    coupled in first mode.

    Assume you have tensor_3d = [[A, B, C]] and matrix = [[A, V]], which are
    coupled in 1st mode. With coupled matrix and tensor factorization (CTMF), the
    factor matrices A, B, C for the CP decomposition of X and the matrix V are found.
    This implementation only works for a coupling in the first mode.

    Solution is found via alternating least squares (ALS) as described in Figure 5 of
    Acar et al, arXiv, 2011.

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
        tensor_3d_pred = [[A,B,C]], matrix_pred = [[A,V]]
    """

    assert tl.is_tensor(X)
    assert tl.is_tensor(Y)

    # initialize values
    A, B, C = init.factors

    # alternating least squares
    # note that the order of the khatri rao product is reversed since tl.unfold has another order
    # than assumed in paper
    for iteration in range(10 ** 4):
        V = solve_least_squares(A, Y)
        A = solve_least_squares(
            np.transpose(
                np.concatenate(
                    (np.transpose(khatri_rao([B, C])), np.transpose(V)), axis=1
                )
            ),
            np.transpose(np.concatenate((tl.unfold(X, 0), Y), axis=1)),
        )

        B = solve_least_squares(khatri_rao([A, C]), np.transpose(tl.unfold(X, 1)))
        C = solve_least_squares(khatri_rao([A, B]), np.transpose(tl.unfold(X, 2)))

        # Perform masking
        if mask_3d is not None:
            X = X * mask_3d + tl.kruskal_tensor.kruskal_to_tensor((None, [A, B, C])) * (
                1 - mask_3d
            )
        if mask_matrix is not None:
            Y = Y * mask_matrix + tl.kruskal_tensor.kruskal_to_tensor(
                (None, [A, V])
            ) * (1 - mask_matrix)

        error_new = np.linalg.norm(
            X - tl.kruskal_tensor.kruskal_to_tensor((None, [A, B, C]))
        ) + np.linalg.norm(Y - tl.kruskal_tensor.kruskal_to_tensor((None, [A, V])))

        if iteration > 5:
            decr = error_old - error_new

            if verbose and iteration % 10 == 0:
                print(
                    "iteration {}, reconstruction error: {}, decrease = {}".format(
                        iteration, error_new, decr
                    )
                )

            if iteration > 0 and (tl.abs(decr) <= 1e-9 or error_new < 1e-5):
                break

        error_old = error_new

    return KruskalTensor((None, [A, B, C])), KruskalTensor((None, [A, V]))
