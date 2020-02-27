"""
##### CLASSIFICATION #####

This module contains our functions for classification analysis.

"""


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def get_train_test_split(df, test_size=.25):

    X = df.loc[:, ['neg', 'neu', 'pos', 'compound', 'review_fulltext']]
    # X = df.drop(columns=['star_rating', 'review_class'])
    y = df.loc[:, 'review_class']
    
    return train_test_split(X, y, test_size=test_size)
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # smote = SMOTE()

    # X_train, y_train = smote.fit_sample(X_train, y_train) 

    # return X_train, X_test, y_train, y_test
