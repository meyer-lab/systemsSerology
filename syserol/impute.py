""" Evaluate the ability of CP to impute data. """

import numpy as np
import random as rd
from statistics import mean 
from .dataImport import createCube
from .tensor import perform_decomposition, impute, R2X



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

def evaluate_missing():
    Cube, GlyCube = createCube()
    orig = Cube #copy of original for comparison
    indices = list()
    #replace 200 real values with missing values
    for num in range(200):
        i = 0
        j = 0
        k = 0
        while True:
            i = rd.randint(0,180) #pick a subject
            j = rd.randint(0,21) #pick a detection 
            k = rd.randint(0,40) #pick an antigen 
            if not np.isnan(Cube[i,j,k]):
                break

        Cube[i,j,k] = np.nan #make that value a NaN    
        indices.append((i,j,k)) #store its index
    
    
    #check differences between original and recon values for different number of components
    Averages = list()
    Sums = list()
    for comp in np.arange(1,10):
        recon = impute(Cube, comp)
        Diff = list()
        for ind in range(len(indices)):
            index = indices[ind]
            origVal = orig[index]
            reconVal = recon[index]
            err = abs(reconVal - origVal) #find difference between values
            Diff.append(err)
        Avg = mean(Diff)
        Sum = sum(Diff)
        print("The average difference for", comp, "components is:", Avg, "and the Sum is:", Sum)
        Averages.append(Avg)
        Sums.append(Sum)

    return Averages, Sums