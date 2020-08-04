
import pickle
from syserol.model import cross_validation

if __name__ == "__main__":
    matrix = cross_validation()
    pickle.dumb(matrix, open("crossValidationMatrix.p", "wb"))
    