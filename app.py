from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model
import json
import os


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    dataset_filepath = request.json.get('dataset_filepath')
    #call the prediction function you created in Step 3
    y_pred, _ = model_predictions(dataset_filepath)
    return str(y_pred)   #add return value for prediction outputs
    

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    score = score_model()
    return str(score)   #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary = dataframe_summary()
    return str(summary)   #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    missing_check = missing_data()
    exe_time = execution_time()
    dependency_check = outdated_packages_list() 

    #add return value for all diagnostics
    return str(f"\nmissing_data: {missing_check}" + f"execution_time: {exe_time}" + f"\noutdated_packages: {dependency_check}")   
    

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
