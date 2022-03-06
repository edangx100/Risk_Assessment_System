from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from pickle import load
import os
from sklearn import metrics
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
output_folder_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])  


#################Function for model scoring
def score_model(production=False):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    
    if production:
        model_path = prod_deployment_path
    else: 
        model_path = output_model_path
    
    # preperation for pickle loading
    model_filename = os.path.join(model_path, "trainedmodel.pkl")
    encoder_filename = os.path.join(model_path, "encoder.pkl")
    # Load trained model and encoders
    model = load(open(model_filename, 'rb'))
    encoder = load(open(encoder_filename, 'rb'))

    if production:
        df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    else:
        df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    df_x, df_y, _ = preprocess_data(df, encoder)

    y_pred = model.predict(df_x)
    f1_score = metrics.f1_score(df_y, y_pred)
    
    with open(os.path.join(output_model_path, "latestscore.txt"), "w") as f:
        f.write(str(f1_score) + "\n")
    
    return f1_score


if __name__ == "__main__":
    score_model()