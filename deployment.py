from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copy


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    source_file = os.path.join(dataset_csv_path, "ingestedfiles.txt")    
    dest_file = os.path.join(prod_deployment_path, "ingestedfiles.txt")   
    copy(source_file, dest_file)

    source_file = os.path.join(output_model_path, "encoder.pkl")    
    dest_file = os.path.join(prod_deployment_path, "encoder.pkl")   
    copy(source_file, dest_file)

    source_file = os.path.join(output_model_path, "trainedmodel.pkl")    
    dest_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")   
    copy(source_file, dest_file)

    source_file = os.path.join(output_model_path, "latestscore.txt")    
    dest_file = os.path.join(prod_deployment_path, "latestscore.txt")   
    copy(source_file, dest_file)

if __name__ == '__main__':
    store_model_into_pickle()