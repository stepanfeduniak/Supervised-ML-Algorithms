import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class NullTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.fillna('null')
class Tokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert each list of words to a column of tokens
        return pd.DataFrame(X.apply(lambda x: x.split()).tolist())