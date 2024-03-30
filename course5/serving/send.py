import requests
import json

url = "http://localhost:5000/predict"

data = {
    {
        "CHAS":{
            "0":0
        },
        "RM":{
            "0":6.575
        },
        "TAX":{
            "0":296.0
        },
        "PTRATIO":{
            "0":15.3
        },
        "B":{
            "0":396.9
        },
        "LSTAT":{
            "0":4.98
        },
    }

input_data = json.dumps(data)
headers = {"Content-Type": "application/json"}

resp = requests.post(url, input_data, headers=headers)
print(resp.text)
