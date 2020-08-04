
import pickle
from syserol.model import Function_Prediction_10FoldCV

if __name__ == "__main__":
    matrix = Function_Prediction_10FoldCV(components=10)
    pickle.dumb(matrix, open("crossValidationMatrix.p", "wb"))
