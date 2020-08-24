#!/usr/bin/env python3

import pickle
from syserol.model import Function_Prediction_10FoldCV

if __name__ == "__main__":
    matrix = Function_Prediction_10FoldCV(10)
    pickle.dump(matrix, open("crossValidationMatrix.p", "wb") )
