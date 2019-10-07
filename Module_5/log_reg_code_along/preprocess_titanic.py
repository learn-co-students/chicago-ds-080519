import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def initial_cleanup(df):
    
    df.drop(['Cabin', 'PassengerId', 'Name'], axis= 1, inplace= True)
    #impute mean age
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    #drop nas
    df.dropna(inplace = True)
    df.drop_duplicates(inplace = True)
    
    return df

def dummy_titanic(X):
    X_cat = X.select_dtypes(include='object')
    for feature in list(X_cat):
        dummies = pd.get_dummies(X_cat[feature], drop_first = True, prefix = f'{feature}')
        X_cat = pd.merge(dummies,X_cat, left_index= True, right_index= True)
        print(X_cat.shape)
        X_cat.drop(feature, axis= 1, inplace= True)
    
    X_num = X.select_dtypes(exclude='object')
    X = pd.merge(X_cat,X_num, right_index = True, left_index = True)

    return X

def scale_titanic(X):
    # scale the data
    X = X
    mm = MinMaxScaler()
    for feature in list(X):
        X[feature] = mm.fit_transform(X[[feature]])
    return X

def preprocess_titanic(df):
    df = initial_cleanup(df)
    
    X = df.drop('Survived', axis = 1)
    y = df.Survived
    
    X = dummy_titanic(X)
    X = scale_titanic(X)
    
    return [X,y] 