
import pandas as pd
import numpy as np
import timeit
import os
import json
from pickle import load
from preprocess import preprocess_data
import subprocess
import sys


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


##################Function to get model predictions
def model_predictions(dataset_filepath):
    #read the deployed model and a test dataset, calculate predictions
    
    # preperation for pickle loading
    model_filename = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    encoder_filename = os.path.join(prod_deployment_path, "encoder.pkl")
    # Load trained model and encoders
    model = load(open(model_filename, 'rb'))
    encoder = load(open(encoder_filename, 'rb'))

    if dataset_filepath is None: 
        dataset_filepath = "testdata.csv"
    dataset_filepath = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(dataset_filepath)

    df_x, df_y, _ = preprocess_data(df, encoder)

    y_pred = model.predict(df_x)

    return y_pred, df_y  #return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    dataset_path = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(dataset_path)

    numeric_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
        ]

    stats_list = []
    for column in numeric_columns:
        stats_list.append( {column: {"mean": df[column].mean()}} )
        stats_list.append( {column: {"median": df[column].median()}} )
        stats_list.append( {column: {"standard deviation": df[column].std()}} )
    
    return stats_list  #return value should be a list containing all summary statistics


##################Function to check for missing data
def missing_data():
    #check for missing data
    dataset_path = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(dataset_path)

    missing_percent_list = []
    for column in df.columns:
        num_nan = df[column].isna().sum()
        num_total = len(df[column].index)

        missing_percent_list.append( {column : str(int(num_nan/num_total*100))+"%"} )

    return missing_percent_list


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    time_list = []

    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing = timeit.default_timer() - starttime
    time_list.append(timing)

    starttime = timeit.default_timer()
    os.system('python training.py')
    timing = timeit.default_timer() - starttime
    time_list.append(timing)

    return time_list  #return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)


if __name__ == '__main__':
    y_pred, df_y = model_predictions(None)
    print(f'Number of prediction: {len(y_pred)}')
    print(f'Number of test data: {len(df_y)}')
    
    stats_list = dataframe_summary()
    print(f'Satistics list: {stats_list}')
    
    missing_percent_list = missing_data()
    print(f'Check missing data: {missing_percent_list}')
    
    time_list = execution_time()
    print(f'Execution time: {time_list}')
    
    outdated_packages = outdated_packages_list()
    print(f'Outdated packages check: {outdated_packages}')




    
