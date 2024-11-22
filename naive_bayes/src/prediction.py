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
spam_df = pd.read_csv("./naive_bayes/data/spam.csv",encoding='ISO-8859-1')

# Define the pipeline for the Message column
spam_messages = spam_df[spam_df['v1'] == 'spam']['v2']
ham_messages = spam_df[spam_df['v1'] == 'ham']['v2']

# Tokenize the messages (using simple split for now)
spam_words = spam_messages.apply(lambda x: x.split()).explode()
ham_words = ham_messages.apply(lambda x: x.split()).explode()

# Step 2: Create a dictionary to hold the word counts
word_counts = defaultdict(lambda: [0,0])  # Default dictionary where [in_spam, not_in_spam]

# Step 3: Count the occurrences of each word in spam and ham messages
for word in spam_words:
    word_counts[word][0] += 1  # Increment the 'in_spam' count

for word in ham_words:
    word_counts[word][1] += 1  # Increment the 'not_in_spam' count
#for word, counts in word_counts.items():
 #   if counts[0]>counts[1]:
  #      print(f"{word}: {counts[0]}, {counts[1]}")



# Test message and prior probabilities (already given)
test_message = "Wassup bro"
coef_bayes_spam = (len(spam_messages) / (len(spam_messages) + len(ham_messages)))
coef_bayes_ham = (len(ham_messages) / (len(spam_messages) + len(ham_messages)))
check_message = test_message.split()

# Initialize the log probabilities
log_prob_spam = log(coef_bayes_spam)  # Start with the log of prior probability of spam
log_prob_ham = log(coef_bayes_ham)    # Start with the log of prior probability of ham

# Step 1: For each word in the message, calculate the likelihood of that word in spam and ham
for word in check_message:
    # Get the spam and ham counts for the word (defaulting to 0 if the word is not found)
    in_spam = word_counts.get(word, [0, 0])[0]
    not_in_spam = word_counts.get(word, [0, 0])[1]
    
    # Avoid division by zero by adding 1 (Laplace smoothing)
    prob_word_given_spam = (in_spam+1 ) / (len(spam_words) + len(word_counts))  # Laplace smoothing
    prob_word_given_ham = (not_in_spam+1) / (len(ham_words) + len(word_counts))  # Laplace smoothing
    print((log(prob_word_given_ham),log(prob_word_given_spam)))
    # Step 2: Update the log probabilities for spam and ham
    log_prob_spam += log(prob_word_given_spam)
    log_prob_ham += log(prob_word_given_ham)

# Step 3: Compare the log probabilities and classify the message
print(len(word_counts))
print(log_prob_ham)
print(log_prob_spam)

if log_prob_spam > log_prob_ham:
    print("The message is classified as Spam")
else:
    print("The message is classified as Ham")


