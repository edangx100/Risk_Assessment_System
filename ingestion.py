import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file

    final_df = pd.DataFrame(columns=['corporation','lastmonth_activity','lastyear_activity','number_of_employees','exited'])
    
    current_filepath = os.getcwd() + '/'
    filenames = os.listdir(current_filepath + input_folder_path)
    for filename in filenames:
        current_df = pd.read_csv(current_filepath + input_folder_path + '/' + filename)
        final_df = final_df.append(current_df).reset_index(drop=True)

    final_df.drop_duplicates(inplace=True)

    final_df.to_csv(f"{output_folder_path}/finaldata.csv", index=False)

    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as f:
        for line in filenames:
            f.write(line + "\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
