class LinearRegressor:
    def __init__(self, analysed_data, predicted_data):
        self.analysed_data=analysed_data
        self.predicted_data=predicted_data
class PolynomialRegressor:
    polynomial_degree=1
    def __init__(self,polynomial_degree, analysed_data, predicted_data):
        self.analysed_data=analysed_data
        self.predicted_data=predicted_data
        self.polynomial_degree=polynomial_degree
