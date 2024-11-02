import numpy as np
import pandas as pd
class PolynomialRegressor:
    polynomial_degree=1
    def __init__(self,polynomial_degree):
        self.polynomial_degree=polynomial_degree
    
    def train_model(self, data, prediction):
        data_array = data
        prediction_data_array = prediction
        #Creating an Vandermonde matrix
        design_matrix_X=np.zeros((len(data), self.polynomial_degree + 1))
        for i in range(self.polynomial_degree + 1):
            design_matrix_X[:, i] = data_array ** i
        print(design_matrix_X)
        m1=design_matrix_X.T@design_matrix_X
        b1=design_matrix_X.T@prediction
        coefficient_vector= np.linalg.solve(m1,b1)
        error_matrix= design_matrix_X@coefficient_vector -prediction
        print(coefficient_vector)
        print(error_matrix)

    
        
        
