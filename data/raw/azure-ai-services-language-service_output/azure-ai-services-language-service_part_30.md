Use the following URL when you create your API request. Replace the placeholder values with
your own values.
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
The name for your project. This value is
case sensitive and must match the
project name in the JSON file that you're
importing.
EmailAppDemo
{API-
VERSION}
The version of the API that you're calling.
2023-04-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
The JSON body you send is similar to the following example. For more information about the
JSON object, see the reference documentation.
JSON
Request URL
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/:import?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Body
{
  "projectFileVersion": "{API-VERSION}",
  "stringIndexType": "Utf16CodeUnit",
\n  "metadata": {
    "projectKind": "Conversation",
    "settings": {
      "confidenceThreshold": 0.7
    },
    "projectName": "{PROJECT-NAME}",
    "multilingual": true,
    "description": "Trying out CLU",
    "language": "{LANGUAGE-CODE}"
  },
  "assets": {
    "projectKind": "Conversation",
    "intents": [
      {
        "category": "intent1"
      },
      {
        "category": "intent2"
      }
    ],
    "entities": [
      {
        "category": "entity1"
      }
    ],
    "utterances": [
      {
        "text": "text1",
        "dataset": "{DATASET}",
        "intent": "intent1",
        "entities": [
          {
            "category": "entity1",
            "offset": 5,
            "length": 5
          }
        ]
      },
      {
        "text": "text2",
        "language": "{LANGUAGE-CODE}",
        "dataset": "{DATASET}",
        "intent": "intent2",
        "entities": []
      }
    ]
  }
}
ﾉ
Expand table
\nKey
Placeholder
Value
Example
{API-
VERSION}
The version of
the API that
you're calling.
2023-04-01
projectName
{PROJECT-NAME}
The name of your project. This value is case sensitive.
EmailAppDemo
language
{LANGUAGE-CODE}
A string that specifies the language code for the
utterances used in your project. If your project is a
multilingual project, choose the language code of
most of the utterances.
en-us
multilingual
true
A Boolean value that enables you to have documents
in multiple languages in your dataset. When your
model is deployed, you can query the model in any
supported language, including languages that aren't
included in your training documents.
true
dataset
{DATASET}
For information on how to split your data between a
testing and training set, see Label your utterances in
AI Foundry. Possible values for this field are Train
and Test .
Train
After a successful request, the API response contains an operation-location  header with a URL
that you can use to check the status of the import job. The header is formatted like this
example:
HTTP
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
When you send a successful project import request, the full request URL for checking the
import job's status (including your endpoint, project name, and job ID) is contained in the
response's operation-location  header.
Use the following GET request to query the status of your import job. You can use the URL you
received from the previous step, or replace the placeholder values with your own values.
rest
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/import/jobs/{JOB-ID}?api-version={API-VERSION}
Get import job status
\nPlaceholder
Value
Example
{ENDPOINT}
The endpoint for authenticating
your API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name for your project. This
value is case-sensitive.
myProject
{JOB-ID}
The ID for locating your import
job status.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are
calling.
2023-04-01
Use the following header to authenticate your request.
Key
Description
Value
Ocp-Apim-
Subscription-Key
The key to your resource. Used for authenticating
your API requests.
{YOUR-PRIMARY-
RESOURCE-KEY}
Once you send the request, you'll get the following response. Keep polling this endpoint until
the status parameter changes to "succeeded".
JSON
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/import/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Response body
{
  "jobId": "xxxxx-xxxxx-xxxx-xxxxx",
  "createdDateTime": "2022-04-18T15:17:20Z",
  "lastUpdatedDateTime": "2022-04-18T15:17:22Z",
  "expirationDateTime": "2022-04-25T15:17:20Z",
  "status": "succeeded"
}
\nAfter importing your project, you only have copied the project's assets and metadata and
assets. You still need to train your model, which will incur usage on your account.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Create a POST request using the following URL, headers, and JSON body to submit a training
job.
Use the following URL when creating your API request. Replace the placeholder values with
your own values.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating
your API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name for your project. This
value is case-sensitive.
EmailApp
{API-
VERSION}
The version of the API you are
calling.
2023-04-01
Use the following header to authenticate your request.
Train your model
Submit training job
Request URL
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/:train?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
\nKey
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the following object in your request. The model will be named after the value you use for
the modelLabel  parameter once training is complete.
JSON
Key
Placeholder
Value
Example
modelLabel
{MODEL-
NAME}
Your Model name.
Model1
trainingConfigVersion
{CONFIG-
VERSION}
The training configuration model version. By
default, the latest model version is used.
2022-05-01
trainingMode
{TRAINING-
MODE}
The training mode to be used for training.
Supported modes are Standard training, faster
training, but only available for English and
Advanced training supported for other
languages and multilingual projects, but
involves longer training times. Learn more about
training modes.
standard
kind
percentage
Split methods. Possible Values are percentage
or manual . See how to train a model for more
information.
percentage
trainingSplitPercentage
80
Percentage of your tagged data to be included
in the training set. Recommended value is 80 .
80
Request body
{
  "modelLabel": "{MODEL-NAME}",
  "trainingMode": "{TRAINING-MODE}",
  "trainingConfigVersion": "{CONFIG-VERSION}",
  "evaluationOptions": {
    "kind": "percentage",
    "testingSplitPercentage": 20,
    "trainingSplitPercentage": 80
  }
}
ﾉ
Expand table
\nKey
Placeholder
Value
Example
testingSplitPercentage
20
Percentage of your tagged data to be included
in the testing set. Recommended value is 20 .
20
Once you send your API request, you will receive a 202  response indicating success. In the
response headers, extract the operation-location  value. It will be formatted like this:
rest
You can use this URL to get the training job status.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
When you send a successful training request, the full request URL for checking the job's status
(including your endpoint, project name, and job ID) is contained in the response's operation-
location  header.
Use the following GET request to get the status of your model's training progress. Replace the
placeholder values below with your own values.
rest
７ Note
The trainingSplitPercentage  and testingSplitPercentage  are only required if Kind  is set
to percentage  and the sum of both percentages should be equal to 100.
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/train/jobs/{JOB-ID}?api-version={API-VERSION}
Get Training Status
Request URL
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/train/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
\nPlaceholder
Value
Example
{YOUR-
ENDPOINT}
The endpoint for authenticating
your API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name for your project. This
value is case-sensitive.
EmailApp
{JOB-ID}
The ID for locating your model's
training status.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-VERSION}
The version of the API you are
calling.
2023-04-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Once you send the request, you will get the following response. Keep polling this endpoint
until the status parameter changes to "succeeded".
JSON
Headers
ﾉ
Expand table
Response Body
{
  "result": {
    "modelLabel": "{MODEL-LABEL}",
    "trainingConfigVersion": "{TRAINING-CONFIG-VERSION}",
    "trainingMode": "{TRAINING-MODE}",
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
  "jobId": "xxxxx-xxxxx-xxxx-xxxxx-xxxx",
\nKey
Value
Example
modelLabel
The model name
Model1
trainingConfigVersion
The training configuration version. By default, the
latest version is used.
2022-05-01
trainingMode
Your selected training mode.
standard
startDateTime
The time training started
2022-04-
14T10:23:04.2598544Z
status
The status of the training job
running
estimatedEndDateTime
Estimated time for the training job to finish
2022-04-
14T10:29:38.2598544Z
jobId
Your training job ID
xxxxx-xxxx-xxxx-xxxx-
xxxxxxxxx
createdDateTime
Training job creation date and time
2022-04-14T10:22:42Z
lastUpdatedDateTime
Training job last updated date and time
2022-04-14T10:23:45Z
expirationDateTime
Training job expiration date and time
2022-04-14T10:22:42Z
This is the step where you make your trained model available form consumption via the
runtime prediction API
.
  "createdDateTime": "2022-04-18T15:44:44Z",
  "lastUpdatedDateTime": "2022-04-18T15:45:48Z",
  "expirationDateTime": "2022-04-25T15:44:44Z",
  "status": "running"
}
ﾉ
Expand table
Deploy your model
 Tip
Use the same deployment name as your primary project for easier maintenance and
minimal changes to your system to handle redirecting your traffic.
Submit deployment job
\nReplace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Create a PUT request using the following URL, headers, and JSON body to start deploying a
conversational language understanding model.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating
your API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-NAME}
The name for your project. This
value is case-sensitive.
myProject
{DEPLOYMENT-
NAME}
The name for your deployment.
This value is case-sensitive.
staging
{API-VERSION}
The version of the API you are
calling.
2023-04-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
JSON
Request URL
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/deployments/{DEPLOYMENT-NAME}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Request Body