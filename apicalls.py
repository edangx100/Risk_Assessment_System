import requests
import os
import json

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = os.path.join(config['output_model_path'])


#Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000"


#Call each API endpoint and store the responses
response1 = requests.post(f"{URL}/prediction", json={"dataset_filepath": "testdata.csv"}).text
response2 = requests.get(f"{URL}/scoring").text
response3 = requests.get(f"{URL}/summarystats").text
response4 = requests.get(f"{URL}/diagnostics").text

#combine all API responses
responses = response1 + "\n\n" + response2 + "\n\n" + response3 + "\n\n" + response4

#write the responses to your workspace
with open(os.path.join(output_model_path, "apireturns.txt"), "w") as api_returns_file:
    api_returns_file.write(responses)


