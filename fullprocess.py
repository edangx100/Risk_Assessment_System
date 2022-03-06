
import json
import os
import ast
import numpy as np
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting


##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = os.path.join(config["input_folder_path"])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path']) 


##################Check and read new data
#first, read ingestedfiles.txt
ingestedfiles_list =[]
with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as f:
    for line in f:
        ingestedfiles_list.append(line.rstrip())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = False
for filename in os.listdir(input_folder_path):
    filepath = input_folder_path + "/" + filename 
    if filepath not in ingestedfiles_list:
        new_files = True


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if new_files == False:
    print("No new data. Exiting.")
    exit(0)

# run data ingestion if new data exist
ingestion.merge_multiple_dataframe()  


##################Checking for model drift
# model updated using new data under 'output_model_path' folder 
scoring.score_model(production=True) 

#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as f:
    score_old_list = ast.literal_eval(f.read())
score_old = np.max(score_old_list)

with open(os.path.join(output_model_path, "latestscore.txt"), "r") as f:
    score_new_list = ast.literal_eval(f.read())
score_new = np.max(score_new_list)


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if score_new >= score_old:
    print(f"New model score: {score_new} \n Previous model score:{score_old} \n No model drift. Exiting." )    
    exit(0)

print(f"New model score: {score_new} \n Previous model score:{score_old} \n Model drift detected. Retrain model." )  
training.train_model()


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle()


##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python diagnostics.py')
reporting.score_model()

# assume app.py is ran, call apis
os.system('python apicalls.py ')




