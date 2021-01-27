import os
import pickle
import numpy as np
import random as rd
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from pilotscripts import config

def load_data(config, preprocessing=True):
    if preprocessing:
        dataframe = pd.read_excel(config.datafile)
        this_var_appears,next_vars_disappear = dropout(config.dropout_criterion_var, 
                                                        config.dropout_criterion_nextvars, dataframe)
        dataframe['drop'] = next_vars_disappear
        feature_sets = config.label_col+config.categorical_features+ config.continuous_features
        selected_frame = dataframe[this_var_appears][feature_sets]
        new_frame = selected_frame.dropna()
        #(new_frame.head())
        #print(new_frame.columns,new_frame.isna().sum(axis=0), print(new_frame['drop']))
        features, labels = encode_features(new_frame, category_features=config.categorical_features, 
                                            continuous_feautres=config.continuous_features,
                                            labels=config.label_col)
        _ = data_split(features,labels, save=True)

    train = load_pickle(config.datapath+'train.pickle')
    test = load_pickle(config.datapath+'test.pickle')
    return train, test

def encode_features(frame, category_features=[], continuous_feautres=[], labels=[]):

    """
    Given the dataframe, process the features
    1. for categorical features: dummy coding
    2. for continuous features: standardizing 
    
    ------------------
    Parameters:
    frame:                  dataframe
    categorical_features:   list of column names in the dataframe needed to be dummy coded
    continuous_feautres :   list of column names in the dataframe needed to be standardized
    labels:                 labels needed to be coded

    """ 

    features = []
    lbs = []
    for var in category_features+continuous_feautres+labels:
        #print(var, labels, frame.columns)
        if var in labels:
            #print(var)
            lbs = pd.get_dummies(frame[var], drop_first=True)
            #print(frame.shape, frame[var].value_counts(), lbs)
        elif var in category_features:
            features.append(pd.get_dummies(frame[var], prefix=var))
        else:
            #print(var)
            features.append((frame[var]-frame[var].mean())/frame[var].std())
    features = pd.concat(features,axis=1)

    return features, lbs


def data_split(features, labels,random_seed=42, save=True):

    """
    Given the dataframe, split the data for train and test set
    
    ------------------
    Parameters:
    features: a matrix
    labels:   a vector

    Optional parameters:
    random seed
    whether saveing the split

    Implicit parameters:
    config.datapath 
    """ 

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=random_seed)
    train = {'data': X_train, 'label': y_train}
    test = {'data': X_test, 'label': y_test}

    if save:
        os.makedirs(config.datapath,exist_ok=True)
        with open(config.datapath+'train.pickle','wb') as f:
            pickle.dump(train,f)
        with open(config.datapath+'test.pickle','wb') as f:
            pickle.dump(test, f)
    return train, test


def dropout(this_var, next_vars,frame):
    
    """
    Given the dataframe, a variable, calculate whether next time it drops out or not
    
    ------------------
    Parameters:
    this_var:   current measure of the variable
    next_vars:  following measures of the variable
    frame:      data frame
    """ 

    if not next_vars:
        return
    this_var_appear = ~frame[this_var].isna()
    next_vars_disappear= frame[next_vars].isna().sum(axis=1) == len(next_vars)
   
    return this_var_appear.values, next_vars_disappear.values

def sample_parameters(params_dict):
    """
    Given a paramter dictionary, return sampled parameter
    ------------------
    Parameters:
    params_dict: parameters of dictionary
    """ 
    return {param:rd.choice(params_dict[param]) for param in params_dict}

def load_pickle(filename):
    with open(filename,'rb') as f:
        data= pickle.load(f)
    return data
