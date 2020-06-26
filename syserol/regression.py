""" Regression methods. """
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from .dataImport import createCube, importFunction
from .tensor import perform_decomposition


def patientComponents(nComp = 1, n_splits = 10):
	""" Generate factorization on cross-validation. """
	cube = createCube()
	cube_reconfig = np.transpose(cube, (1, 2, 0))
	Y = importFunction()['ADCC']
	Y_pred = np.empty(Y.shape)

	kf = KFold(n_splits=n_splits)

	for train_index, test_index in tqdm(kf.split(np.squeeze(cube[:, :, 0]))):
		cube_train = cube[train_index, :, :]

		# Normal decomposition of training data
		factor_train = perform_decomposition(cube_train, nComp)

		# Now decompose with full dataset, but only vary patient matrix
		factor_train_fixed = [factor_train[1], factor_train[2]]
		factor_full = perform_decomposition(cube_reconfig, nComp, fixed = factor_train_fixed)

		# Extract matrices for patients
		patient_test = factor_full[2][test_index, :]
		patient_train = factor_train[0]

		# Setup Y and remove missing values
		Y_train = Y[train_index]
		idxx = np.isfinite(Y_train)
		Y_train = Y_train[idxx]
		patient_train = patient_train[idxx, :]

		# Fit the model
		model = Lasso(alpha = 0.01)
		model.fit(patient_train, Y_train)

		# Predict the test set
		Y_pred[test_index] = model.predict(patient_test)


	idxx = np.isfinite(Y)
	Y = Y[idxx]
	Y_pred = Y_pred[idxx]

	print(r2_score(Y, Y_pred))
