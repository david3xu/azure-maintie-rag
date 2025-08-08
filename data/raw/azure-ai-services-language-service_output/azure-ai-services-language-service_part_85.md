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
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Once you send your API request, you will receive a 202  response indicating success. In the
response headers, extract the operation-location  value. It will be formatted like this:
rest
JOB-ID  is used to identify your request, since this operation is asynchronous. Use this URL to
get the exported project JSON, using the same authentication method.
Replace the placeholders in the following request with your {PRIMARY-ENDPOINT}  and {PRIMARY-
RESOURCE-KEY}  that you obtained in the first step.
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/:export?stringIndexType=Utf16CodeUnit&api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/jobs/{JOB-ID}?api-version={API-VERSION}
Get export job status
\nUse the following GET request to query the status of your export job. You can use the URL you
received from the previous step, or replace the placeholder values below with your own values.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating your
API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name for your project. This value is
case-sensitive.
myProject
{JOB-ID}
The ID for locating your export job
status. This is in the location  header
value you received in the previous step.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are calling.
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
JSON
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/export/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Response body
{
  "resultUrl": "{Endpoint}/language/authoring/analyze-
conversations/projects/{PROJECT-NAME}/export/jobs/xxxxxx-xxxxx-xxxxx-xx/result?
api-version={API-VERSION}",
  "jobId": "xxxx-xxxxx-xxxxx-xxx",
  "createdDateTime": "2022-04-18T15:23:07Z",
\nUse the URL from the resultUrl  key in the body to view the exported assets from this job.
Submit a GET request using the {RESULT-URL}  you received from the previous step to view the
results of the export job.
Use the following header to authenticate your request.
Key
Description
Value
Ocp-Apim-Subscription-
Key
The key to your resource. Used for authenticating your
API requests.
{PRIMARY-RESOURCE-
KEY}
Copy the response body as you will use it as the body for the next import job.
Now go ahead and import the exported project assets in your new project in the secondary
region so you can replicate it.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Submit a POST request using the following URL, headers, and JSON body to import your
project.
Use the following URL when creating your API request. Replace the placeholder values below
with your own values.
  "lastUpdatedDateTime": "2022-04-18T15:23:08Z",
  "expirationDateTime": "2022-04-25T15:23:07Z",
  "status": "succeeded"
}
Get export results
Headers
ﾉ
Expand table
Import to a new project
Submit import job
Request URL
\nrest
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
myProject
{API-
VERSION}
The version of the API you are
calling.
2023-04-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the following sample JSON as your body.
JSON
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/:import?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Body
７ Note
Each intent should only be of one type only from (CLU,LUIS and qna)
{
  "projectFileVersion": "{API-VERSION}",
  "stringIndexType": "Utf16CodeUnit",
  "metadata": {
    "projectKind": "Orchestration",
    "settings": {
\nKey
Placeholder
Value
Example
api-version
{API-
VERSION}
The version of the API you are calling. The version used
here must be the same API version in the URL.
2022-03-01-
preview
projectName
{PROJECT-
NAME}
The name of your project. This value is case-sensitive.
EmailApp
language
{LANGUAGE-
CODE}
A string specifying the language code for the utterances
used in your project. If your project is a multilingual project,
en-us
      "confidenceThreshold": 0
    },
    "projectName": "{PROJECT-NAME}",
    "description": "Project description",
    "language": "{LANGUAGE-CODE}"
  },
  "assets": {
    "projectKind": "Orchestration",
    "intents": [
      {
        "category": "string",
        "orchestration": {
          "kind": "luis",
          "luisOrchestration": {
            "appId": "00001111-aaaa-2222-bbbb-3333cccc4444",
            "appVersion": "string",
            "slotName": "string"
          },
          "cluOrchestration": {
            "projectName": "string",
            "deploymentName": "string"
          },
          "qnaOrchestration": {
            "projectName": "string"
          }
        }
      }
    ],
    "utterances": [
      {
        "text": "Trying orchestration",
        "language": "{LANGUAGE-CODE}",
        "intent": "string"
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
choose the language code of the majority of the
utterances.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to query the status of your import job. You can use the URL you
received from the previous step, or replace the placeholder values below with your own values.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating your
API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name for your project. This value is
case-sensitive.
myProject
{JOB-ID}
The ID for locating your export job
status. This is in the location  header
value you received in the previous step.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are calling.
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
Get import job status
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/import/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
\nOnce you send the request, you will get the following response. Keep polling this endpoint
until the status parameter changes to "succeeded".
JSON
After importing your project, you only have copied the project's assets and metadata and
assets. You still need to train your model, which will incur usage on your account.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Create a POST request using the following URL, headers, and JSON body to submit a training
job.
Use the following URL when creating your API request. Replace the placeholder values below
with your own values.
rest
Response body
{
  "jobId": "xxxxx-xxxxx-xxxx-xxxxx",
  "createdDateTime": "2022-04-18T15:17:20Z",
  "lastUpdatedDateTime": "2022-04-18T15:17:22Z",
  "expirationDateTime": "2022-04-25T15:17:20Z",
  "status": "succeeded"
}
Train your model
Submit training job
Request URL
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/:train?api-version={API-VERSION}
ﾉ
Expand table
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
EmailApp
{API-
VERSION}
The version of the API you are
calling.
2023-04-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the following object in your request. The model will be named MyModel  once training is
complete.
JSON
Key
Placeholder
Value
Example
modelLabel
{MODEL-NAME}
Your Model name.
Model1
Headers
ﾉ
Expand table
Request body
{
  "modelLabel": "{MODEL-NAME}",
  "trainingMode": "standard",
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
trainingMode
standard
Training mode. Only one mode for training is
available in orchestration, which is standard .
standard
trainingConfigVersion
{CONFIG-
VERSION}
The training configuration model version. By
default, the latest model version is used.
2022-05-01
kind
percentage
Split methods. Possible values are percentage
or manual . See how to train a model for more
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
Once you send your API request, you will receive a 202  response indicating success. In the
response headers, extract the operation-location  value. It will be formatted like this:
rest
You can use this URL to get the training job status.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to get the status of your model's training progress. Replace the
placeholder values below with your own values.
７ Note
The trainingSplitPercentage  and testingSplitPercentage  are only required if Kind  is set
to percentage  and the sum of both percentages should be equal to 100.
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/train/jobs/{JOB-ID}?api-version={API-VERSION}
Get Training Status
Request URL
\nrest
Placeholder
Value
Example
{YOUR-
ENDPOINT}
The endpoint for authenticating your API
request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name for your project. This value is
case-sensitive.
EmailApp
{JOB-ID}
The ID for locating your model's training
status. This is in the location  header
value you received when submitted your
training job.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are calling.
2023-04-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Once you send the request, you will get the following response. Keep polling this endpoint
until the status parameter changes to "succeeded".
JSON
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/train/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Response Body
{
  "result": {
    "modelLabel": "{MODEL-LABEL}",
    "trainingConfigVersion": "{TRAINING-CONFIG-VERSION}",
    "estimatedEndDateTime": "2022-04-18T15:47:58.8190649Z",
    "trainingStatus": {