""" Evaluate the ability of CP to impute data. """

import numpy as np
from .dataImport import createCube
from .tensor import impute


def evalMissing(nComp = 1, numSample = 100):
	""" Evaluate how well factorization imputes missing values. """
	cube = createCube()

	orig = []
	recon = []

	idxs = np.argwhere(np.isfinite(cube))

	for ii in range(numSample):
		i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]

		orig.append(cube[i, j, k])
		cubeTemp = np.copy(cube)
		cubeTemp[i, j, k] = np.nan

		tensorR = impute(cubeTemp, nComp)

		recon.append(tensorR[i, j, k])
		print(len(recon))
		print(np.corrcoef(orig, recon)[0, 1])

		if len(recon) > 100:
			break

	return orig, recon
