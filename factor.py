#!/usr/bin/env python3
import sys
import pickle

from syserol.dataImport import createCube
from syserol.tensor import perform_CMTF


if __name__ == "__main__":
    cube, glyCube = createCube()
    facT, facM = perform_CMTF(cube, glyCube, int(sys.argv[1]))

    pickle.dump((facT, facM), open( "factors.p", "wb" ) )