#!/usr/bin/env python3
import pickle

from syserol.dataImport import createCube
from syserol.tensor import perform_CMTF


if __name__ == "__main__":
    cube, glyCube = createCube()
    
    for ii in range(1, 24):
        output = perform_CMTF(cube, glyCube, ii)
        pickle.dump(output, open( "factors" + str(ii) + ".p", "wb" ) )
