import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def initial_cleanup(df):
    
    """Remove unecessary columns PassengerId and Name, as well as Cabin, 
    which has only 204/891 non-na values.
    """
    
    df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis= 1, inplace= True)
    #impute mean age
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    #drop nas
    df.dropna(inplace = True)
    df.drop_duplicates(inplace = True)
    
    return df

def dummy_titanic(X):
    """Transform the four object columns left after dropping into binary and dummies.
    Feature count jumps from 9 to 687, mainly because of huge number of ticket options.
    """
    X_cat = X.select_dtypes(include='object')
    for feature in list(X_cat):
        dummies = pd.get_dummies(X_cat[feature], drop_first = True, prefix = f'{feature}')
        X_cat = pd.merge(dummies,X_cat, left_index= True, right_index= True)

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 42)
    return X_train, X_test, y_train, y_test 

def preprocess_titanic_no_tts(df):
    df = initial_cleanup(df)
    
    X = df.drop('Survived', axis = 1)
    y = df.Survived
    
    X = dummy_titanic(X)
    X = scale_titanic(X)
    

    return X,y