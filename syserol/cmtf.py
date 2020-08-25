import numpy as np

import tensorly as tl
from tensorly.kruskal_tensor import KruskalTensor, kruskal_to_tensor


def coupled_matrix_tensor_3d_factorization(Y, mask_matrix, init, verbose=False):
    """
    Calculate the glycosylation matrix components corresponding to the patient components from the tensor.
    """
    assert tl.is_tensor(Y)

    # initialize values
    A = init.factors[0]

    # alternating least squares
    for iteration in range(10 ** 4):
        V = np.transpose(np.linalg.lstsq(A, Y, rcond=-1)[0])

        # Perform masking
        if mask_matrix is not None:
            Y = Y * mask_matrix + kruskal_to_tensor((None, [A, V])) * (1 - mask_matrix)

        error_new = np.linalg.norm(Y - kruskal_to_tensor((None, [A, V])))

        if iteration > 5:
            decr = error_old - error_new

            if verbose and iteration % 10 == 0:
                print(f"iteration {iteration}, reconstruction error: {error_new}, decrease = {decr}")

            if iteration > 0 and (tl.abs(decr) <= 1e-9 or error_new < 1e-5):
                break

        error_old = error_new

    return KruskalTensor((None, [A, V]))
