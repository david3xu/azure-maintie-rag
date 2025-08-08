Key
Placeholder
Value
Example
Since all the
documents are in
the root of the
container this
should be the
document name.
dataset
{DATASET}
The test set to
which this file will
go to when split
before training. See
How to train a
model for more
information on how
your data is split.
Possible values for
this field are Train
and Test .
Train
Once you send your API request, you’ll receive a 202  response indicating that the job was
submitted correctly. In the response headers, extract the operation-location  value. It will be
formatted like this:
rest
{JOB-ID}  is used to identify your request, since this operation is asynchronous. You’ll use this
URL to get the import job status.
Possible error scenarios for this request:
The selected resource doesn't have proper permissions for the storage account.
The storageInputContainerName  specified doesn't exist.
Invalid language code is used, or if the language code type isn't string.
multilingual  value is a string and not a boolean.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
NAME}/import/jobs/{JOB-ID}?api-version={API-VERSION}
Get import job status
\nUse the following GET request to get the status of your importing your project. Replace the
placeholder values below with your own values.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating your API
request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name of your project. This value is
case-sensitive.
myProject
{JOB-ID}
The ID for locating your model's training
status. This value is in the location  header
value you received in the previous step.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are calling. The
value referenced here is for the latest
version released. See Model lifecycle to
learn more about other available API
versions.
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
After importing your project, you only have copied the project's assets and metadata and
assets. You still need to train your model, which will incur usage on your account.
Request URL
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
NAME}/import/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Train your model
\nReplace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Submit a POST request using the following URL, headers, and JSON body to submit a training
job. Replace the placeholder values below with your own values.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating your API
request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name of your project. This value is
case-sensitive.
myProject
{API-
VERSION}
The version of the API you are calling. The
value referenced here is for the latest
version released. See Model lifecycle to
learn more about other available API
versions.
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the following JSON in your request body. The model will be given the {MODEL-NAME}  once
training is complete. Only successful training jobs will produce models.
Submit training job
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/:train?api-
version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Request body
\nJSON
Key
Placeholder
Value
Example
modelLabel
{MODEL-NAME}
The model name that will be assigned to your
model once trained successfully.
myModel
trainingConfigVersion
{CONFIG-
VERSION}
This is the model version that will be used to
train the model.
2022-05-01
evaluationOptions
Option to split your data across training and
testing sets.
{}
kind
percentage
Split methods. Possible values are percentage  or
manual . See How to train a model for more
information.
percentage
trainingSplitPercentage
80
Percentage of your tagged data to be included
in the training set. Recommended value is 80 .
80
testingSplitPercentage
20
Percentage of your tagged data to be included
in the testing set. Recommended value is 20 .
20
Once you send your API request, you’ll receive a 202  response indicating that the job was
submitted correctly. In the response headers, extract the location  value. It will be formatted
like this:
rest
{
"modelLabel": "{MODEL-NAME}",
"trainingConfigVersion": "{CONFIG-VERSION}",
"evaluationOptions": {
"kind": "percentage",
"trainingSplitPercentage": 80,
"testingSplitPercentage": 20
}
}
ﾉ
Expand table
７ Note
The trainingSplitPercentage  and testingSplitPercentage  are only required if Kind  is set
to percentage  and the sum of both percentages should be equal to 100.
\n{JOB-ID}  is used to identify your request, since this operation is asynchronous. You can use this
URL to get the training status.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to get the status of your model's training progress. Replace the
placeholder values below with your own values.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating your API
request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name of your project. This value is
case-sensitive.
myProject
{JOB-ID}
The ID for locating your model's training
status. This value is in the location  header
value you received in the previous step.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are calling. The
value referenced here is for the latest
version released. See Model lifecycle to
learn more about other available API
versions.
2022-05-01
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
NAME}/train/jobs/{JOB-ID}?api-version={API-VERSION}
Get training status
Request URL
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
NAME}/train/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
\nUse the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Once you send the request, you’ll get the following response.
JSON
This is the step where you make your trained model available form consumption via the
runtime prediction API
.
ﾉ
Expand table
Response Body
{
  "result": {
    "modelLabel": "{MODEL-NAME}",
    "trainingConfigVersion": "{CONFIG-VERSION}",
    "estimatedEndDateTime": "2022-04-18T15:47:58.8190649Z",
    "trainingStatus": {
      "percentComplete": 3,
      "startDateTime": "2022-04-18T15:45:06.8190649Z",
      "status": "running"
    },
    "evaluationStatus": {
      "percentComplete": 0,
      "status": "notStarted"
    }
  },
  "jobId": "{JOB-ID}",
  "createdDateTime": "2022-04-18T15:44:44Z",
  "lastUpdatedDateTime": "2022-04-18T15:45:48Z",
  "expirationDateTime": "2022-04-25T15:44:44Z",
  "status": "running"
}
Deploy your model
 Tip
\nReplace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Submit a PUT request using the following URL, headers, and JSON body to submit a
deployment job. Replace the placeholder values below with your own values.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating your API
request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name of your project. This value is
case-sensitive.
myProject
{DEPLOYMENT-
NAME}
The name of your deployment. This value
is case-sensitive.
staging
{API-
VERSION}
The version of the API you are calling.
The value referenced here is for the latest
version released. See Model lifecycle to
learn more about other available API
versions.
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the same deployment name as your primary project for easier maintenance and
minimal changes to your system to handle redirecting your traffic.
Submit deployment job
{Endpoint}/language/authoring/analyze-
text/projects/{projectName}/deployments/{deploymentName}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
\nUse the following JSON in the body of your request. Use the name of the model you to assign
to the deployment.
JSON
Key
Placeholder
Value
Example
trainedModelLabel
{MODEL-
NAME}
The model name that will be assigned to your
deployment. You can only assign successfully trained
models. This value is case-sensitive.
myModel
Once you send your API request, you’ll receive a 202  response indicating that the job was
submitted correctly. In the response headers, extract the operation-location  value. It will be
formatted like this:
rest
{JOB-ID}  is used to identify your request, since this operation is asynchronous. You can use this
URL to get the deployment status.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to query the status of the deployment job. You can use the URL
you received from the previous step, or replace the placeholder values below with your own
values.
rest
Request body
{
  "trainedModelLabel": "{MODEL-NAME}"
}
ﾉ
Expand table
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
NAME}/deployments/{DEPLOYMENT-NAME}/jobs/{JOB-ID}?api-version={API-VERSION}
Get the deployment status
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
\nPlaceholder
Value
Example
{ENDPOINT}
The endpoint for authenticating your API
request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name of your project. This value is
case-sensitive.
myProject
{DEPLOYMENT-
NAME}
The name of your deployment. This value
is case-sensitive.
staging
{JOB-ID}
The ID for locating your model's training
status. This is in the location  header
value you received in the previous step.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are calling.
The value referenced here is for the latest
version released. See Model lifecycle to
learn more about other available API
versions.
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Once you send the request, you will get the following response. Keep polling this endpoint
until the status parameter changes to "succeeded". You should get a 200  code to indicate the
success of the request.
JSON
NAME}/deployments/{DEPLOYMENT-NAME}/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Response Body
{
    "jobId":"{JOB-ID}",
    "createdDateTime":"{CREATED-TIME}",
\nWithin your system, at the step where you call runtime prediction API
 check for the response
code returned from the submit task API. If you observe a consistent failure in submitting the
request, this could indicate an outage in your primary region. Failure once doesn't mean an
outage, it may be transient issue. Retry submitting the job through the secondary resource you
have created. For the second request use your {SECONDARY-ENDPOINT}  and {SECONDARY-RESOURCE-
KEY} , if you have followed the steps above, {PROJECT-NAME}  and {DEPLOYMENT-NAME}  would be
the same so no changes are required to the request body.
In case you revert to using your secondary resource you will observe slight increase in latency
because of the difference in regions where your model is deployed.
Maintaining the freshness of both projects is an important part of the process. You need to
frequently check if any updates were made to your primary project so that you move them
over to your secondary project. This way if your primary region fails and you move into the
secondary region you should expect similar model performance since it already contains the
latest updates. Setting the frequency of checking if your projects are in sync is an important
choice. We recommend that you do this check daily in order to guarantee the freshness of data
in your secondary model.
Use the following url to get your project details, one of the keys returned in the body indicates
the last modified date of the project. Repeat the following step twice, one for your primary
project and another for your secondary project and compare the timestamp returned for both
of them to check if they are out of sync.
Use the following GET request to get your project details. Replace the placeholder values
below with your own values.
rest
    "lastUpdatedDateTime":"{UPDATED-TIME}",
    "expirationDateTime":"{EXPIRATION-TIME}",
    "status":"running"
}
Changes in calling the runtime
Check if your projects are out of sync
Get project details
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}?api-version=