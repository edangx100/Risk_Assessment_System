from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from pickle import dump
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import json
from preprocess import preprocess_data


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    #Preprocess data
    df_x, df_y, encoder = preprocess_data(df, None)

    # split into training and test data
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20)

    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='ovr', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))

    # Preperation to save model and encoders
    model_filename = os.path.join(model_path, "trainedmodel.pkl")
    encoder_filename = os.path.join(model_path, "encoder.pkl")
    #write the trained model to your workspace in a file called trainedmodel.pkl
    dump(model, open(model_filename, 'wb'))
    dump(encoder, open(encoder_filename, 'wb'))


if __name__ == "__main__":
    train_model()