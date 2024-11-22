from naive_bayes import spam_detector
from collections import defaultdict
import codecs
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from Preprocessors.preprocessor import NullTransformer, Tokenizer
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from math import log
# Set print options
np.set_printoptions(suppress=True, precision=4)

# Load data
df1 = pd.read_csv("./naive_bayes/data/spam.csv",encoding='ISO-8859-1')
df2 = pd.read_csv("./naive_bayes/data/spam_ham_dataset.csv",encoding='ISO-8859-1')

df1_cleaned = df1[['v1', 'v2']].rename(columns={'v1': 'Label', 'v2': 'Message'})
df2_cleaned = df2.rename(columns={'Category': 'Label', 'Message': 'Message'})

merged_df = pd.concat([df1_cleaned, df2_cleaned], ignore_index=True)

spam_detector_machine=spam_detector(merged_df)
spam_detector_machine.fit("ham","spam","Label","Message")
text_to_check="Wassup bro, I just won 1000 dollars by betting on football"
result=spam_detector_machine.predict(text_to_check)
print(result)