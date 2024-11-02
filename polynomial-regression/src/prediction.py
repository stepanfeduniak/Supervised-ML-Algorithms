from polynomial_regressor import PolynomialRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True, precision=4) 

fish_df=pd.read_csv("./polynomial-regression/data/fishing.csv")
y=fish_df["Weight"]
X=fish_df["Length1"]
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.8, random_state = 1)
models_mean_absolute_error_train=[]
models_mean_absolute_error_test=[]
for i in range(7):
    model_x = PolynomialRegressor(i+1)
    model_x.train_model(train_X, train_y)
    prediction_x=model_x.predict(val_X)
    models_mean_absolute_error_train.append(model_x.mean_error_train)
    models_mean_absolute_error_test.append(mean_absolute_error(val_y, prediction_x))

best_model_n=models_mean_absolute_error_test.index(min(models_mean_absolute_error_test))+1

print(models_mean_absolute_error_train)
print(models_mean_absolute_error_test)
print(best_model_n)
model1 = PolynomialRegressor(best_model_n)
model1.train_model(train_X, train_y)

prediction=model1.predict(val_X)
print(model1.coefficient_vector)
mean_error=mean_absolute_error(val_y, prediction)
print(model1.mean_error_train)
print(mean_error)

#Visualisation

x_range = np.linspace(min(X), max(X), 100)
y = model1.predict(x_range)

plt.figure(figsize=(10, 6))
plt.scatter(train_X, train_y, color='black', label='Original Data')
plt.plot(x_range, y, label=f'Best Polynomial Degree {best_model_n}', color='blue')
plt.xlabel("Length1")
plt.ylabel("Weight")
plt.title(f"Train Data (Degree {best_model_n})")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(val_X, val_y, color='black', label='Validation Data')
plt.plot(x_range, y, label=f'Best Polynomial Degree {best_model_n}', color='blue')
plt.xlabel("Length1")
plt.ylabel("Weight")
plt.title(f"Validation set related to trained polynomial (Degree {best_model_n})")
plt.legend()
plt.show()

