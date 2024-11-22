import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

class naive_bayes_algorithm:
    pass

class spam_detector:
    def __init__(self,polynomial_degree: int=1):
        self.polynomial_degree = polynomial_degree
        self.coefficient_vector = None
        self.error_train = None
    
    def vandermonde(self, x: np.ndarray):
        vandermonde_matrix=np.zeros((len(x), self.polynomial_degree + 1))
        for i in range(self.polynomial_degree + 1):
            vandermonde_matrix[:, i] = x ** i
        return vandermonde_matrix
    
    def train_model(self, data, prediction):
        design_matrix_X=self.vandermonde(data)
        m1=design_matrix_X.T@design_matrix_X
        b1=design_matrix_X.T@prediction
        self.coefficient_vector= np.linalg.solve(m1,b1)
        self.error_train= design_matrix_X@self.coefficient_vector -prediction
        self.mean_error_train= mean_absolute_error(prediction,design_matrix_X@self.coefficient_vector)
        
    
    def predict(self, X):
        prediction=np.zeros(len(X))
        for i in range(self.polynomial_degree+1):
            prediction=prediction+self.coefficient_vector[i]*(X**i)
        return prediction