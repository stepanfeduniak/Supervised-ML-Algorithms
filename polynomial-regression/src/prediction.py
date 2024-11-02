from polynomial_regressor import PolynomialRegressor
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True, precision=4) 

x = np.array([1, 12, 33, 441])
y = np.array([1, 12, 3, 4])
model1 = PolynomialRegressor(2)
model1.train_model(x, y)