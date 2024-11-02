from polynomial_regressor import PolynomialRegressor
import numpy as np
np.set_printoptions(suppress=True, precision=4) 

x = np.array([1, 33, 3, 4])
y = np.array([1, 12, 3, 4])
model1 = PolynomialRegressor(1)
model1.train_model(x, y)