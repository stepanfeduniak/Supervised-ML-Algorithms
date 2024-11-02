import numpy as np
import pandas as pd
class PolynomialRegressor:
    def __init__(self,polynomial_degree: int=1):
        self.polynomial_degree = polynomial_degree
        self.coefficient_vector = None
    
    def vandermonde(self, x: np.ndarray):
        vandermonde_matrix=np.zeros((len(x), self.polynomial_degree + 1))
        for i in range(self.polynomial_degree + 1):
            vandermonde_matrix[:, i] = x ** i
        return vandermonde_matrix
    
    def train_model(self, data, prediction):
        design_matrix_X=self.vandermonde(data)
        print(design_matrix_X)
        m1=design_matrix_X.T@design_matrix_X
        b1=design_matrix_X.T@prediction
        coefficient_vector= np.linalg.solve(m1,b1)
        error_matrix= design_matrix_X@coefficient_vector -prediction
        print(coefficient_vector)
        print(error_matrix)

    
        
        
