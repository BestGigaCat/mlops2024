import os
import io
import boto3
import json
import csv

ENDPOINT_NAME = "CustomerChurn"
runtime= boto3.client('runtime.sagemaker')

def handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    print(event)
    data = json.loads(json.dumps(event))
    print(data)
    payload = data['body']
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=str(payload))
    print(response)
    # more feature extractions for complicated cases
    result = json.loads(response['Body'].read().decode())
    
    cloudwatch = boto3.client('cloudwatch')
    response = cloudwatch.put_metric_data(
        MetricData = [{
                'MetricName': 'Result',
                'Dimensions': [
                    {
                        'Name': 'REGISTERED_SERVICE',
                        'Value': 'InferenceService'
                    },
                    {
                        'Name': 'APP_VERSION',
                        'Value': '1.0'
                    },
                ],
                'Unit': 'None',
                'Value': result
            }],
            Namespace='MLApp'
    )
    print(result)
    
    return result
