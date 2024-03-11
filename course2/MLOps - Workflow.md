```python
import sys

!{sys.executable} -m pip install --upgrade stepfunctions
```

    Collecting stepfunctions
      Downloading stepfunctions-2.3.0.tar.gz (67 kB)
         |████████████████████████████████| 67 kB 546 kB/s             
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: sagemaker>=2.1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from stepfunctions) (2.116.0)
    Requirement already satisfied: boto3>=1.14.38 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from stepfunctions) (1.23.10)
    Requirement already satisfied: pyyaml in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from stepfunctions) (5.4.1)
    Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from boto3>=1.14.38->stepfunctions) (0.5.0)
    Requirement already satisfied: botocore<1.27.0,>=1.26.10 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from boto3>=1.14.38->stepfunctions) (1.26.10)
    Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from boto3>=1.14.38->stepfunctions) (0.10.0)
    Requirement already satisfied: numpy<2.0,>=1.9.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (1.19.5)
    Requirement already satisfied: smdebug-rulesconfig==1.0.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (1.0.1)
    Requirement already satisfied: protobuf<4.0,>=3.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (3.15.2)
    Requirement already satisfied: pathos in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (0.2.8)
    Requirement already satisfied: protobuf3-to-dict<1.0,>=0.1.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (0.1.5)
    Requirement already satisfied: importlib-metadata<5.0,>=1.4.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (3.7.0)
    Requirement already satisfied: packaging>=20.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (21.3)
    Requirement already satisfied: attrs<23,>=20.3.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (20.3.0)
    Requirement already satisfied: schema in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (0.7.5)
    Requirement already satisfied: pandas in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (1.1.5)
    Requirement already satisfied: google-pasta in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.1.0->stepfunctions) (0.2.0)
    Requirement already satisfied: urllib3<1.27,>=1.25.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from botocore<1.27.0,>=1.26.10->boto3>=1.14.38->stepfunctions) (1.26.8)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from botocore<1.27.0,>=1.26.10->boto3>=1.14.38->stepfunctions) (2.8.1)
    Requirement already satisfied: zipp>=0.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from importlib-metadata<5.0,>=1.4.0->sagemaker>=2.1.0->stepfunctions) (3.4.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from importlib-metadata<5.0,>=1.4.0->sagemaker>=2.1.0->stepfunctions) (4.0.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from packaging>=20.0->sagemaker>=2.1.0->stepfunctions) (2.4.7)
    Requirement already satisfied: six>=1.9 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from protobuf<4.0,>=3.1->sagemaker>=2.1.0->stepfunctions) (1.15.0)
    Requirement already satisfied: pytz>=2017.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pandas->sagemaker>=2.1.0->stepfunctions) (2021.1)
    Requirement already satisfied: pox>=0.3.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pathos->sagemaker>=2.1.0->stepfunctions) (0.3.0)
    Requirement already satisfied: ppft>=1.6.6.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pathos->sagemaker>=2.1.0->stepfunctions) (1.6.6.4)
    Requirement already satisfied: dill>=0.3.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pathos->sagemaker>=2.1.0->stepfunctions) (0.3.4)
    Requirement already satisfied: multiprocess>=0.70.12 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pathos->sagemaker>=2.1.0->stepfunctions) (0.70.12.2)
    Requirement already satisfied: contextlib2>=0.5.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from schema->sagemaker>=2.1.0->stepfunctions) (0.6.0.post1)
    Building wheels for collected packages: stepfunctions
      Building wheel for stepfunctions (setup.py) ... [?25ldone
    [?25h  Created wheel for stepfunctions: filename=stepfunctions-2.3.0-py2.py3-none-any.whl size=78153 sha256=ddbfa3b8d8e8bfd7e29acf9b94e8171d326b359fc0acdb44e01334798c9623b3
      Stored in directory: /home/ec2-user/.cache/pip/wheels/4b/75/8e/166fad54033824bc17b9118d091d34eaab1837565aa5f0f57e
    Successfully built stepfunctions
    Installing collected packages: stepfunctions
    Successfully installed stepfunctions-2.3.0



```python
import uuid
import logging
import stepfunctions
import boto3
import sagemaker

from sagemaker.amazon.amazon_estimator import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.s3 import S3Uploader
from stepfunctions import steps
from stepfunctions.steps import TrainingStep, ModelStep
from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow

# Set up sagemaker connection session
session = sagemaker.Session()
# Set up logging level
stepfunctions.set_stream_logger(level=logging.INFO)

# Set up AWS python SDK session
region = boto3.Session().region_name
# Set up S3 bucket location 
bucket = session.default_bucket()
# Generate random UUID for this run
id = uuid.uuid4().hex

# Create a unique name for the AWS Glue job to be created. If you change the
# default name, you may need to change the Step Functions execution role.
job_name = "glue-customer-churn-etl-{}".format(id)

# Create a unique name for the AWS Lambda function to be created. If you change
# the default name, you may need to change the Step Functions execution role.
function_name = "query-training-status-{}".format(id)
```


```python
# paste the role'ARN we created for this course 
workflow_execution_role = "arn:aws:iam::846634201516:role/MLOpsCourseRole"

# SageMaker Execution Role
# You can use sagemaker.get_execution_role() if running inside sagemaker's notebook instance
sagemaker_execution_role = (
    sagemaker.get_execution_role()
)
```


```python
session = sagemaker.Session()
# In the case you cannot get the right default bucket, you can specify a S3 location you created manually
bucket = session.default_bucket()
print(bucket)
```

    sagemaker-us-west-2-846634201516



```python
# Same role we created in homework 1
glue_role = "arn:aws:iam::846634201516:role/MLOpsCourseRole"
```


```python
# Same role we created in homework 1
lambda_role = "arn:aws:iam::846634201516:role/MLOpsCourseRole"
```


```python
# Name anything you want
project_name = "ml_deploy"

# Copy customer churn csv data into this notebook instance
# Then use the following code to copy your local CSV to S3 location for model training
data_source = S3Uploader.upload(
    local_path="./data/customer-churn.csv",
    desired_s3_uri="s3://{}/{}".format(bucket, project_name),
    sagemaker_session=session,
)

train_prefix = "train"
val_prefix = "validation"

# Train and validation dataset location in S3
train_data = "s3://{}/{}/{}/".format(bucket, project_name, train_prefix)
validation_data = "s3://{}/{}/{}/".format(bucket, project_name, val_prefix)
```


```python
# Copy glue_etl.py to notebook instance
# Upload glue script to S3 bucket
glue_script_location = S3Uploader.upload(
    local_path="./code/glue_etl.py",
    desired_s3_uri="s3://{}/{}".format(bucket, project_name),
    sagemaker_session=session,
)
glue_client = boto3.client("glue")

# create a ETL job in Glue to split training and validation dataset
response = glue_client.create_job(
    Name=job_name,
    Description="PySpark job to extract the data and split in to training and validation data sets",
    Role=glue_role,  # you can pass your existing AWS Glue role here if you have used Glue before
    ExecutionProperty={"MaxConcurrentRuns": 2},
    Command={"Name": "glueetl", "ScriptLocation": glue_script_location, "PythonVersion": "3"},
    DefaultArguments={"--job-language": "python"},
    GlueVersion="3.0",
    WorkerType="Standard",
    NumberOfWorkers=2,
    Timeout=60,
)
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/boto3/compat.py:88: PythonDeprecationWarning: Boto3 will no longer support Python 3.6 starting May 30, 2022. To continue receiving service updates, bug fixes, and security updates please upgrade to Python 3.7 or later. More information can be found here: https://aws.amazon.com/blogs/developer/python-support-policy-updates-for-aws-sdks-and-tools/
      warnings.warn(warning, PythonDeprecationWarning)



```python
import zipfile

# Model validation pipeline
# Copy query_training_status.py to lcoal instance notebook
zip_name = "query_training_status.zip"
lambda_source_code = "./code/query_training_status.py"

# Zip the script
zf = zipfile.ZipFile(zip_name, mode="w")
zf.write(lambda_source_code, arcname=lambda_source_code.split("/")[-1])
zf.close()

# Copy zipped script to S3 for lambda use
S3Uploader.upload(
    local_path=zip_name,
    desired_s3_uri="s3://{}/{}".format(bucket, project_name),
    sagemaker_session=session,
)
```




    's3://sagemaker-us-west-2-846634201516/ml_deploy/query_training_status.zip'




```python
# Create lambda client
lambda_client = boto3.client("lambda")

# Create a lambda function for model result validation
response = lambda_client.create_function(
    FunctionName=function_name,
    Runtime="python3.9",
    Role=lambda_role,
    Handler="query_training_status.lambda_handler",
    Code={"S3Bucket": bucket, "S3Key": "{}/{}".format(project_name, zip_name)},
    Description="Queries a SageMaker training job and return the results.",
    Timeout=15,
    MemorySize=128,
)
```


```python
# Retrive XGBoost algorithm container for training purpose
container = sagemaker.image_uris.retrieve("xgboost", region, "latest")

# Create XGBoost estimator (model) with m4.xlarge instance type
xgb = sagemaker.estimator.Estimator(
    container,
    sagemaker_execution_role,
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    output_path="s3://{}/{}/output".format(bucket, project_name),
)

# Set initial hyperparameter configurations
xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    objective="binary:logistic",
    eval_metric="error",
    num_round=100,
)
```


```python
# Specify model training execution configurations
# SageMaker expects unique names for each job, model and endpoint.
# If these names are not unique the execution will fail.
execution_input = ExecutionInput(
    schema={
        "TrainingJobName": str,
        "GlueJobName": str,
        "ModelName": str,
        "EndpointName": str,
        "LambdaFunctionName": str,
    }
)
```


```python
# Start glue data ETL job run
etl_step = steps.GlueStartJobRunStep(
    "Extract, Transform, Load",
    parameters={
        "JobName": execution_input["GlueJobName"],
        "Arguments": {
            "--S3_SOURCE": data_source,
            "--S3_DEST": "s3a://{}/{}/".format(bucket, project_name),
            "--TRAIN_KEY": train_prefix + "/",
            "--VAL_KEY": val_prefix + "/",
        },
    },
)
```


```python
# Define model training setp
training_step = steps.TrainingStep(
    "Model Training",
    estimator=xgb,
    data={
        "train": TrainingInput(train_data, content_type="text/csv"),
        "validation": TrainingInput(validation_data, content_type="text/csv"),
    },
    job_name=execution_input["TrainingJobName"],
    wait_for_completion=True,
)
```


```python
# Define model store / register step
model_step = steps.ModelStep(
    "Save Model",
    model=training_step.get_expected_model(),
    model_name=execution_input["ModelName"],
    result_path="$.ModelStepResults",
)
```


```python
# Define lambda for model validation step
lambda_step = steps.compute.LambdaStep(
    "Query Training Results",
    parameters={
        "FunctionName": execution_input["LambdaFunctionName"],
        "Payload": {"TrainingJobName.$": "$.TrainingJobName"},
    },
)
```


```python
# Name accuracy check in AWS StepFunction
check_accuracy_step = steps.states.Choice("Accuracy > 90%")
```


```python
# Configure model deployment endpoint
endpoint_config_step = steps.EndpointConfigStep(
    "Create Model Endpoint Config",
    endpoint_config_name=execution_input["ModelName"],
    model_name=execution_input["ModelName"],
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
)
```


```python
# Update Model Endpoint
endpoint_step = steps.EndpointStep(
    "Update Model Endpoint",
    endpoint_name=execution_input["EndpointName"],
    endpoint_config_name=execution_input["ModelName"],
    # If you want continuous training in existing pipeline, need to modify this to true
    update=False,
)
```


```python
# Name fail critera in AWS StepFunction
fail_step = steps.states.Fail(
    "Model Accuracy Too Low", comment="Validation accuracy lower than threshold"
)
```


```python
# Define logic for model validation
threshold_rule = steps.choice_rule.ChoiceRule.NumericLessThan(
    variable=lambda_step.output()["Payload"]["trainingMetrics"][0]["Value"], value=0.1
)

check_accuracy_step.add_choice(rule=threshold_rule, next_step=endpoint_config_step)
check_accuracy_step.default_choice(next_step=fail_step)
```


```python
# Define step function end
endpoint_config_step.next(endpoint_step)
```




    Update Model Endpoint EndpointStep(resource='arn:aws:states:::sagemaker:createEndpoint', parameters={'EndpointConfigName': <stepfunctions.inputs.placeholders.ExecutionInput object at 0x7f8754c5b208>, 'EndpointName': <stepfunctions.inputs.placeholders.ExecutionInput object at 0x7f8754c5b390>}, type='Task')




```python
# Chain model training automation as a pipeline
workflow_definition = steps.Chain(
    [etl_step, training_step, model_step, lambda_step, check_accuracy_step]
)
```


```python
# Define workflow in AWS StepFunction
workflow = Workflow(
    name="MyInferenceRoutine_{}".format(id),
    definition=workflow_definition,
    role=workflow_execution_role,
    execution_input=execution_input,
)
```


```python
# Genearate DAG in graph in AWS StepFunction
workflow.render_graph()
```





<link rel="stylesheet" type="text/css" href="https://do0of8uwbahzz.cloudfront.net/graph.css">
<div id="graph-116" class="workflowgraph">

    <svg></svg>

</div>

<script type="text/javascript">

require.config({
    paths: {
        sfn: "https://do0of8uwbahzz.cloudfront.net/sfn",
    }
});

require(['sfn'], function(sfn) {
    var element = document.getElementById('graph-116')

    var options = {
        width: parseFloat(getComputedStyle(element, null).width.replace("px", "")),
        height: 600,
        layout: 'LR',
        resizeHeight: true
    };

    var definition = {"StartAt": "Extract, Transform, Load", "States": {"Extract, Transform, Load": {"Parameters": {"JobName.$": "$$.Execution.Input['GlueJobName']", "Arguments": {"--S3_SOURCE": "s3://sagemaker-us-west-2-846634201516/ml_deploy/customer-churn.csv", "--S3_DEST": "s3a://sagemaker-us-west-2-846634201516/ml_deploy/", "--TRAIN_KEY": "train/", "--VAL_KEY": "validation/"}}, "Resource": "arn:aws:states:::glue:startJobRun.sync", "Type": "Task", "Next": "Model Training"}, "Model Training": {"Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync", "Parameters": {"AlgorithmSpecification": {"TrainingImage": "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest", "TrainingInputMode": "File"}, "OutputDataConfig": {"S3OutputPath": "s3://sagemaker-us-west-2-846634201516/ml_deploy/output"}, "StoppingCondition": {"MaxRuntimeInSeconds": 86400}, "ResourceConfig": {"VolumeSizeInGB": 30, "InstanceCount": 1, "InstanceType": "ml.m4.xlarge"}, "RoleArn": "arn:aws:iam::846634201516:role/AmazonMWAA-SageMaker-Role", "InputDataConfig": [{"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://sagemaker-us-west-2-846634201516/ml_deploy/train/", "S3DataDistributionType": "FullyReplicated"}}, "ContentType": "text/csv", "ChannelName": "train"}, {"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://sagemaker-us-west-2-846634201516/ml_deploy/validation/", "S3DataDistributionType": "FullyReplicated"}}, "ContentType": "text/csv", "ChannelName": "validation"}], "HyperParameters": {"max_depth": "5", "eta": "0.2", "gamma": "4", "min_child_weight": "6", "subsample": "0.8", "objective": "binary:logistic", "eval_metric": "error", "num_round": "100"}, "TrainingJobName.$": "$$.Execution.Input['TrainingJobName']", "DebugHookConfig": {"S3OutputPath": "s3://sagemaker-us-west-2-846634201516/ml_deploy/output"}}, "Type": "Task", "Next": "Save Model"}, "Save Model": {"ResultPath": "$.ModelStepResults", "Parameters": {"ExecutionRoleArn": "arn:aws:iam::846634201516:role/AmazonMWAA-SageMaker-Role", "ModelName.$": "$$.Execution.Input['ModelName']", "PrimaryContainer": {"Environment": {}, "Image": "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest", "ModelDataUrl.$": "$['ModelArtifacts']['S3ModelArtifacts']"}}, "Resource": "arn:aws:states:::sagemaker:createModel", "Type": "Task", "Next": "Query Training Results"}, "Query Training Results": {"Parameters": {"FunctionName.$": "$$.Execution.Input['LambdaFunctionName']", "Payload": {"TrainingJobName.$": "$.TrainingJobName"}}, "Resource": "arn:aws:states:::lambda:invoke", "Type": "Task", "Next": "Accuracy > 90%"}, "Accuracy > 90%": {"Type": "Choice", "Choices": [{"Variable": "$['Payload']['trainingMetrics'][0]['Value']", "NumericLessThan": 0.1, "Next": "Create Model Endpoint Config"}], "Default": "Model Accuracy Too Low"}, "Model Accuracy Too Low": {"Comment": "Validation accuracy lower than threshold", "Type": "Fail"}, "Create Model Endpoint Config": {"Resource": "arn:aws:states:::sagemaker:createEndpointConfig", "Parameters": {"EndpointConfigName.$": "$$.Execution.Input['ModelName']", "ProductionVariants": [{"InitialInstanceCount": 1, "InstanceType": "ml.m4.xlarge", "ModelName.$": "$$.Execution.Input['ModelName']", "VariantName": "AllTraffic"}]}, "Type": "Task", "Next": "Update Model Endpoint"}, "Update Model Endpoint": {"Resource": "arn:aws:states:::sagemaker:createEndpoint", "Parameters": {"EndpointConfigName.$": "$$.Execution.Input['ModelName']", "EndpointName.$": "$$.Execution.Input['EndpointName']"}, "Type": "Task", "End": true}}};
    var elementId = '#graph-116';

    var graph = new sfn.StateMachineGraph(definition, elementId, options);
    graph.render();
});

</script>





```python
# Create workflow in AWS StepFunction
workflow.create()
```

    [32m[INFO] Workflow created successfully on AWS Step Functions.[0m





    'arn:aws:states:us-west-2:846634201516:stateMachine:MyInferenceRoutine_1efcdd1db5e74f7d946f01f3ec4fcbaa'




```python
# Execute training automation workflow and pass parameters
execution = workflow.execute(
    inputs={
        "TrainingJobName": "regression-{}".format(id),  # Each Sagemaker Job requires a unique name,
        "GlueJobName": job_name,
        "ModelName": "CustomerChurn-{}".format(id),  # Each Model requires a unique name,
        "EndpointName": "CustomerChurn",  # Each Endpoint requires a unique name
        "LambdaFunctionName": function_name,
    }
)
```

    [32m[INFO] Workflow execution started successfully on AWS Step Functions.[0m



```python
execution.render_progress()
```





<link rel="stylesheet" type="text/css" href="https://do0of8uwbahzz.cloudfront.net/graph.css">
<div id="graph-126" class="workflowgraph">

    <style>
        .graph-legend ul {
            list-style-type: none;
            padding: 10px;
            padding-left: 0;
            margin: 0;
            position: absolute;
            top: 0;
            background: transparent;
        }

        .graph-legend li {
            margin-left: 10px;
            display: inline-block;
        }

        .graph-legend li > div {
            width: 10px;
            height: 10px;
            display: inline-block;
        }

        .graph-legend .success { background-color: #2BD62E }
        .graph-legend .failed { background-color: #DE322F }
        .graph-legend .cancelled { background-color: #DDDDDD }
        .graph-legend .in-progress { background-color: #53C9ED }
        .graph-legend .caught-error { background-color: #FFA500 }
    </style>
    <div class="graph-legend">
        <ul>
            <li>
                <div class="success"></div>
                <span>Success</span>
            </li>
            <li>
                <div class="failed"></div>
                <span>Failed</span>
            </li>
            <li>
                <div class="cancelled"></div>
                <span>Cancelled</span>
            </li>
            <li>
                <div class="in-progress"></div>
                <span>In Progress</span>
            </li>
            <li>
                <div class="caught-error"></div>
                <span>Caught Error</span>
            </li>
        </ul>
    </div>

    <svg></svg>
    <a href="https://console.aws.amazon.com/states/home?region=us-west-2#/executions/details/arn:aws:states:us-west-2:846634201516:execution:MyInferenceRoutine_1efcdd1db5e74f7d946f01f3ec4fcbaa:261bd9b5-fac0-4e6e-a9c0-2e8b4b87bd1e" target="_blank"> Inspect in AWS Step Functions </a>
</div>

<script type="text/javascript">

require.config({
    paths: {
        sfn: "https://do0of8uwbahzz.cloudfront.net/sfn",
    }
});

require(['sfn'], function(sfn) {
    var element = document.getElementById('graph-126')

    var options = {
        width: parseFloat(getComputedStyle(element, null).width.replace("px", "")),
        height: 1000,
        layout: 'LR',
        resizeHeight: true
    };

    var definition = {"StartAt": "Extract, Transform, Load", "States": {"Extract, Transform, Load": {"Parameters": {"JobName.$": "$$.Execution.Input['GlueJobName']", "Arguments": {"--S3_SOURCE": "s3://sagemaker-us-west-2-846634201516/ml_deploy/customer-churn.csv", "--S3_DEST": "s3a://sagemaker-us-west-2-846634201516/ml_deploy/", "--TRAIN_KEY": "train/", "--VAL_KEY": "validation/"}}, "Resource": "arn:aws:states:::glue:startJobRun.sync", "Type": "Task", "Next": "Model Training"}, "Model Training": {"Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync", "Parameters": {"AlgorithmSpecification": {"TrainingImage": "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest", "TrainingInputMode": "File"}, "OutputDataConfig": {"S3OutputPath": "s3://sagemaker-us-west-2-846634201516/ml_deploy/output"}, "StoppingCondition": {"MaxRuntimeInSeconds": 86400}, "ResourceConfig": {"VolumeSizeInGB": 30, "InstanceCount": 1, "InstanceType": "ml.m4.xlarge"}, "RoleArn": "arn:aws:iam::846634201516:role/AmazonMWAA-SageMaker-Role", "InputDataConfig": [{"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://sagemaker-us-west-2-846634201516/ml_deploy/train/", "S3DataDistributionType": "FullyReplicated"}}, "ContentType": "text/csv", "ChannelName": "train"}, {"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://sagemaker-us-west-2-846634201516/ml_deploy/validation/", "S3DataDistributionType": "FullyReplicated"}}, "ContentType": "text/csv", "ChannelName": "validation"}], "HyperParameters": {"max_depth": "5", "eta": "0.2", "gamma": "4", "min_child_weight": "6", "subsample": "0.8", "objective": "binary:logistic", "eval_metric": "error", "num_round": "100"}, "TrainingJobName.$": "$$.Execution.Input['TrainingJobName']", "DebugHookConfig": {"S3OutputPath": "s3://sagemaker-us-west-2-846634201516/ml_deploy/output"}}, "Type": "Task", "Next": "Save Model"}, "Save Model": {"ResultPath": "$.ModelStepResults", "Parameters": {"ExecutionRoleArn": "arn:aws:iam::846634201516:role/AmazonMWAA-SageMaker-Role", "ModelName.$": "$$.Execution.Input['ModelName']", "PrimaryContainer": {"Environment": {}, "Image": "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest", "ModelDataUrl.$": "$['ModelArtifacts']['S3ModelArtifacts']"}}, "Resource": "arn:aws:states:::sagemaker:createModel", "Type": "Task", "Next": "Query Training Results"}, "Query Training Results": {"Parameters": {"FunctionName.$": "$$.Execution.Input['LambdaFunctionName']", "Payload": {"TrainingJobName.$": "$.TrainingJobName"}}, "Resource": "arn:aws:states:::lambda:invoke", "Type": "Task", "Next": "Accuracy > 90%"}, "Accuracy > 90%": {"Type": "Choice", "Choices": [{"Variable": "$['Payload']['trainingMetrics'][0]['Value']", "NumericLessThan": 0.1, "Next": "Create Model Endpoint Config"}], "Default": "Model Accuracy Too Low"}, "Model Accuracy Too Low": {"Comment": "Validation accuracy lower than threshold", "Type": "Fail"}, "Create Model Endpoint Config": {"Resource": "arn:aws:states:::sagemaker:createEndpointConfig", "Parameters": {"EndpointConfigName.$": "$$.Execution.Input['ModelName']", "ProductionVariants": [{"InitialInstanceCount": 1, "InstanceType": "ml.m4.xlarge", "ModelName.$": "$$.Execution.Input['ModelName']", "VariantName": "AllTraffic"}]}, "Type": "Task", "Next": "Update Model Endpoint"}, "Update Model Endpoint": {"Resource": "arn:aws:states:::sagemaker:createEndpoint", "Parameters": {"EndpointConfigName.$": "$$.Execution.Input['ModelName']", "EndpointName.$": "$$.Execution.Input['EndpointName']"}, "Type": "Task", "End": true}}};
    var elementId = '#graph-126';
    var events = { 'events': [{"timestamp": 1710119487.945, "type": "ExecutionStarted", "id": 1, "previousEventId": 0, "executionStartedEventDetails": {"input": "{\n    \"TrainingJobName\": \"regression-1efcdd1db5e74f7d946f01f3ec4fcbaa\",\n    \"GlueJobName\": \"glue-customer-churn-etl-1efcdd1db5e74f7d946f01f3ec4fcbaa\",\n    \"ModelName\": \"CustomerChurn-1efcdd1db5e74f7d946f01f3ec4fcbaa\",\n    \"EndpointName\": \"CustomerChurn\",\n    \"LambdaFunctionName\": \"query-training-status-1efcdd1db5e74f7d946f01f3ec4fcbaa\"\n}", "inputDetails": {"truncated": false}, "roleArn": "arn:aws:iam::846634201516:role/MLOpsCourseRole"}}, {"timestamp": 1710119487.983, "type": "TaskStateEntered", "id": 2, "previousEventId": 0, "stateEnteredEventDetails": {"name": "Extract, Transform, Load", "input": "{\n    \"TrainingJobName\": \"regression-1efcdd1db5e74f7d946f01f3ec4fcbaa\",\n    \"GlueJobName\": \"glue-customer-churn-etl-1efcdd1db5e74f7d946f01f3ec4fcbaa\",\n    \"ModelName\": \"CustomerChurn-1efcdd1db5e74f7d946f01f3ec4fcbaa\",\n    \"EndpointName\": \"CustomerChurn\",\n    \"LambdaFunctionName\": \"query-training-status-1efcdd1db5e74f7d946f01f3ec4fcbaa\"\n}", "inputDetails": {"truncated": false}}}, {"timestamp": 1710119487.983, "type": "TaskScheduled", "id": 3, "previousEventId": 2, "taskScheduledEventDetails": {"resourceType": "glue", "resource": "startJobRun.sync", "region": "us-west-2", "parameters": "{\"Arguments\":{\"--S3_SOURCE\":\"s3://sagemaker-us-west-2-846634201516/ml_deploy/customer-churn.csv\",\"--S3_DEST\":\"s3a://sagemaker-us-west-2-846634201516/ml_deploy/\",\"--TRAIN_KEY\":\"train/\",\"--VAL_KEY\":\"validation/\"},\"JobName\":\"glue-customer-churn-etl-1efcdd1db5e74f7d946f01f3ec4fcbaa\"}"}}, {"timestamp": 1710119488.089, "type": "TaskStarted", "id": 4, "previousEventId": 3, "taskStartedEventDetails": {"resourceType": "glue", "resource": "startJobRun.sync"}}, {"timestamp": 1710119488.26, "type": "TaskSubmitted", "id": 5, "previousEventId": 4, "taskSubmittedEventDetails": {"resourceType": "glue", "resource": "startJobRun.sync", "output": "{\"JobRunId\":\"jr_c7279ffd510a171a95bc34e2dcd2b4a46b2d150d088bd6a46cf42a650ce7bed4\",\"SdkHttpMetadata\":{\"AllHttpHeaders\":{\"Connection\":[\"keep-alive\"],\"x-amzn-RequestId\":[\"94d02ffc-a4ee-4d08-906b-dea5ac8df4f2\"],\"Content-Length\":[\"82\"],\"Date\":[\"Mon, 11 Mar 2024 01:11:28 GMT\"],\"Content-Type\":[\"application/x-amz-json-1.1\"]},\"HttpHeaders\":{\"Connection\":\"keep-alive\",\"Content-Length\":\"82\",\"Content-Type\":\"application/x-amz-json-1.1\",\"Date\":\"Mon, 11 Mar 2024 01:11:28 GMT\",\"x-amzn-RequestId\":\"94d02ffc-a4ee-4d08-906b-dea5ac8df4f2\"},\"HttpStatusCode\":200},\"SdkResponseMetadata\":{\"RequestId\":\"94d02ffc-a4ee-4d08-906b-dea5ac8df4f2\"},\"JobName\":\"glue-customer-churn-etl-1efcdd1db5e74f7d946f01f3ec4fcbaa\"}", "outputDetails": {"truncated": false}}}] };

    var graph = new sfn.StateMachineExecutionGraph(definition, events, elementId, options);
    graph.render();
});

</script>





```python
execution.list_events()
```




    [{'timestamp': datetime.datetime(2024, 1, 26, 5, 27, 29, 477000, tzinfo=tzlocal()),
      'type': 'ExecutionStarted',
      'id': 1,
      'previousEventId': 0,
      'executionStartedEventDetails': {'input': '{\n    "TrainingJobName": "regression-672358213b124ad7b7f7af810b8ea8e7",\n    "GlueJobName": "glue-customer-churn-etl-672358213b124ad7b7f7af810b8ea8e7",\n    "ModelName": "CustomerChurn-672358213b124ad7b7f7af810b8ea8e7",\n    "EndpointName": "CustomerChurn",\n    "LambdaFunctionName": "query-training-status-672358213b124ad7b7f7af810b8ea8e7"\n}',
       'inputDetails': {'truncated': False},
       'roleArn': 'arn:aws:iam::846634201516:role/MLOpsCourseRole'}},
     {'timestamp': datetime.datetime(2024, 1, 26, 5, 27, 29, 509000, tzinfo=tzlocal()),
      'type': 'TaskStateEntered',
      'id': 2,
      'previousEventId': 0,
      'stateEnteredEventDetails': {'name': 'Extract, Transform, Load',
       'input': '{\n    "TrainingJobName": "regression-672358213b124ad7b7f7af810b8ea8e7",\n    "GlueJobName": "glue-customer-churn-etl-672358213b124ad7b7f7af810b8ea8e7",\n    "ModelName": "CustomerChurn-672358213b124ad7b7f7af810b8ea8e7",\n    "EndpointName": "CustomerChurn",\n    "LambdaFunctionName": "query-training-status-672358213b124ad7b7f7af810b8ea8e7"\n}',
       'inputDetails': {'truncated': False}}},
     {'timestamp': datetime.datetime(2024, 1, 26, 5, 27, 29, 509000, tzinfo=tzlocal()),
      'type': 'TaskScheduled',
      'id': 3,
      'previousEventId': 2,
      'taskScheduledEventDetails': {'resourceType': 'glue',
       'resource': 'startJobRun.sync',
       'region': 'us-west-2',
       'parameters': '{"Arguments":{"--S3_SOURCE":"s3://sagemaker-us-west-2-846634201516/ml_deploy/customer-churn.csv","--S3_DEST":"s3a://sagemaker-us-west-2-846634201516/ml_deploy/","--TRAIN_KEY":"train/","--VAL_KEY":"validation/"},"JobName":"glue-customer-churn-etl-672358213b124ad7b7f7af810b8ea8e7"}'}},
     {'timestamp': datetime.datetime(2024, 1, 26, 5, 27, 29, 582000, tzinfo=tzlocal()),
      'type': 'TaskStarted',
      'id': 4,
      'previousEventId': 3,
      'taskStartedEventDetails': {'resourceType': 'glue',
       'resource': 'startJobRun.sync'}}]




```python
workflow.list_executions(html=True)
```





<style>

.table-widget {
    width: 100%;
    font-size: 14px;
    line-height: 28px;
    color: #545b64;
    border-spacing: 0;
    background-color: #fff;
    border-color: grey;
    background: #fafafa;
}

.table-widget thead th {
    text-align: left !important;
    color: #879596;
    padding: 0.3em 2em;
    border-bottom: 1px solid #eaeded;
    min-height: 4rem;
    line-height: 28px;
}

.table-widget thead th:first-of-type {
}

.table-widget td {
    overflow-wrap: break-word;
    padding: 0.4em 2em;
    line-height: 28px;
    text-align: left !important;
    background: #fff;
    border-bottom: 1px solid #eaeded;
    border-top: 1px solid transparent;
}

.table-widget td:before {
    content: "";
    height: 3rem;
}

a {
    cursor: pointer;
    text-decoration: none !important;
    color: #007dbc;
}

a:hover {
    text-decoration: underline !important;
}

a.disabled {
    color: black;
    cursor: default;
    pointer-events: none;
}

.hide {
    display: none;
}

pre {
    white-space: pre-wrap;
}


* {
    box-sizing: border-box;
}

.table-widget {
    min-width: 100%;
    font-size: 14px;
    line-height: 28px;
    color: #545b64;
    border-spacing: 0;
    background-color: #fff;
    border-color: grey;
    background: #fafafa;
}

.table-widget thead th {
    text-align: left !important;
    color: #879596;
    padding: 0.3em 2em;
    border-bottom: 1px solid #eaeded;
    min-height: 4rem;
    line-height: 28px;
}

.table-widget td {
    /* padding: 24px 18px; */
    padding: 0.4em 2em;
    line-height: 28px;
    text-align: left !important;
    background: #fff;
    border-bottom: 1px solid #eaeded;
    border-top: 1px solid transparent;
}

.table-widget td:before {
    content: "";
    height: 3rem;
}

.table-widget .clickable-cell {
    cursor: pointer;
}

.hide {
    display: none;
}

.triangle-right {
    width: 0;
    height: 0;
    border-top: 5px solid transparent;
    border-left: 8px solid #545b64;
    border-bottom: 5px solid transparent;
    margin-right: 5px;
}

a.awsui {
    text-decoration: none !important;
    color: #007dbc;
}

a.awsui:hover {
    text-decoration: underline !important;
}

</style>
<table class="table-widget">
    <thead>
        <tr>
            <th>Name</th>
            <th>Status</th>
            <th>Started</th>
            <th>End Time</th>
        </tr>
    </thead>
    <tbody>

<tr class="awsui-table-row">
    <td>
        <a href="https://console.aws.amazon.com/states/home?region=us-west-2#/executions/details/arn:aws:states:us-west-2:846634201516:execution:MyInferenceRoutine_1efcdd1db5e74f7d946f01f3ec4fcbaa:261bd9b5-fac0-4e6e-a9c0-2e8b4b87bd1e" target="_blank" class="awsui">261bd9b5-fac0-4e6e-a9c0-2e8b4b87bd1e</a>
    </td>
    <td>RUNNING</td>
    <td>Mar 11, 2024 01:11:27.945 AM</td>
    <td>-</td>
</tr>

    </tbody>
</table>





```python
# Validate endpoint
endpoint_name="CustomerChurn"
sagemaker_runtime = boto3.client(
    "sagemaker-runtime", region_name='us-west-2')

response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name, 
    ContentType='text/csv',
    Body = "2.0,400.0,0.38571846040122537,2.0,4.177940384158745,0.0,3.745462710628048,250.0,3.699591756294294,1.0,11.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0"
)
print(response['Body'].read().decode('utf-8'))
```

    0.8917055130004883



```python
# Clean up resource, note does not include delete SageMaker endpoint
# lambda_client.delete_function(FunctionName=function_name)
# glue_client.delete_job(JobName=job_name)
# workflow.delete()
```


```python

```
