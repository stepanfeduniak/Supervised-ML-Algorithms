import numpy as np
import pandas as pd
from collections import defaultdict
from math import log
class naive_bayes_algorithm:
    pass

class spam_detector:
    def __init__(self,df):
        self.df = df
        
    def fit(self,type_1,type_2,classyfying_column,data_column):
        self.spam_messages = self.df[self.df[classyfying_column] == type_2][data_column]
        self.ham_messages = self.df[self.df[classyfying_column] == type_1][data_column]
        spam_words=self.tokenise(self.spam_messages)
        ham_words=self.tokenise(self.ham_messages)
        self.value_list=self.word_counter(spam_words,ham_words)
        print("Model is fit")
    
    def word_counter(self, a,b):
        word_counts = defaultdict(lambda: [0,0])
        for word in a:
            word_counts[word][0] += 1
        for word in b:
            word_counts[word][1] += 1
        return word_counts
    
    def predict(self,phrase):
        coef_bayes_spam = (len(self.spam_messages) / (len(self.spam_messages) + len(self.ham_messages)))
        coef_bayes_ham = (len(self.ham_messages) / (len(self.spam_messages) + len(self.ham_messages)))
        check_message = phrase.split()
        log_prob_spam = log(coef_bayes_spam) 
        log_prob_ham = log(coef_bayes_ham)   
        for word in check_message:
            in_spam = self.value_list.get(word, [0, 0])[0]
            not_in_spam = self.value_list.get(word, [0, 0])[1]
            prob_word_given_spam = (in_spam+1 ) / (len(self.spam_messages) + len(self.value_list)) 
            prob_word_given_ham = (not_in_spam+1) / (len(self.ham_messages) + len(self.value_list)) 
            log_prob_spam += log(prob_word_given_spam)
            log_prob_ham += log(prob_word_given_ham)
        if log_prob_spam > log_prob_ham:
            return 'Spam'
        else:
            return 'Ham'




    def tokenise(self,X):
        a= X.apply(lambda x: x.split()).explode()
        return a

