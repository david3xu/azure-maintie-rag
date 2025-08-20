Key
Value
Example
modelLabel
The model name
Model1
trainingConfigVersion
The training configuration version. By default, the
latest version is used.
2022-05-01
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
      "percentComplete": 3,
      "startDateTime": "2022-04-18T15:45:06.8190649Z",
      "status": "running"
    },
    "evaluationStatus": {
      "percentComplete": 0,
      "status": "notStarted"
    }
  },
  "jobId": "xxxxxx-xxxxx-xxxxxx-xxxxxx",
  "createdDateTime": "2022-04-18T15:44:44Z",
  "lastUpdatedDateTime": "2022-04-18T15:45:48Z",
  "expirationDateTime": "2022-04-25T15:44:44Z",
  "status": "running"
}
ﾉ
Expand table
Deploy your model
 Tip
\nReplace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Create a PUT request using the following URL, headers, and JSON body to start deploying an
orchestration workflow model.
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
Use the same deployment name as your primary project for easier maintenance and
minimal changes to your system to handle redirecting your traffic.
Submit deployment job
Request URL
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/deployments/{DEPLOYMENT-NAME}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
\nJSON
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
Once you send your API request, you will receive a 202  response indicating success. In the
response headers, extract the operation-location  value. It will be formatted like this:
rest
You can use this URL to get the deployment job status.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and
{SECONDARY-RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to get the status of your deployment job. Replace the
placeholder values below with your own values.
rest
Request Body
{
  "trainedModelLabel": "{MODEL-NAME}",
}
ﾉ
Expand table
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/deployments/{DEPLOYMENT-NAME}/jobs/{JOB-ID}?api-version={API-VERSION}
Get the deployment status
Request URL
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-
NAME}/deployments/{DEPLOYMENT-NAME}/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
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
The name for your project. This value is
case-sensitive.
myProject
{DEPLOYMENT-
NAME}
The name for your deployment. This
value is case-sensitive.
staging
{JOB-ID}
The ID for locating your model's training
status. This is in the location  header
value you received from the API in
response to your model deployment
request.
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
Headers
ﾉ
Expand table
Response Body
{
    "jobId":"{JOB-ID}",
    "createdDateTime":"{CREATED-TIME}",
    "lastUpdatedDateTime":"{UPDATED-TIME}",
    "expirationDateTime":"{EXPIRATION-TIME}",
    "status":"running"
}
\nWithin your system, at the step where you call runtime API
 check for the response code
returned from the submit task API. If you observe a consistent failure in submitting the request,
this could indicate an outage in your primary region. Failure once doesn't mean an outage, it
may be transient issue. Retry submitting the job through the secondary resource you have
created. For the second request use your {YOUR-SECONDARY-ENDPOINT}  and secondary key, if you
have followed the steps above, {PROJECT-NAME}  and {DEPLOYMENT-NAME}  would be the same so
no changes are required to the request body.
In case you revert to using your secondary resource you will observe slight increase in latency
because of the difference in regions where your model is deployed.
Maintaining the freshness of both projects is an important part of process. You need to
frequently check if any updates were made to your primary project so that you move them
over to your secondary project. This way if your primary region fail and you move into the
secondary region you should expect similar model performance since it already contains the
latest updates. Setting the frequency of checking if your projects are in sync is an important
choice, we recommend that you do this check daily in order to guarantee the freshness of data
in your secondary model.
Use the following url to get your project details, one of the keys returned in the body indicates
the last modified date of the project. Repeat the following step twice, one for your primary
project and another for your secondary project and compare the timestamp returned for both
of them to check if they are out of sync.
Use the following GET request to get your project details. You can use the URL you received
from the previous step, or replace the placeholder values below with your own values.
rest
Changes in calling the runtime
Check if your projects are out of sync
Get project details
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-NAME}?api-
version={API-VERSION}
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
myProject
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
JSON
Repeat the same steps for your replicated project using {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY} . Compare the returned lastModifiedDateTime  from both projects. If your
primary project was modified sooner than your secondary one, you need to repeat the steps of
exporting, importing, training and deploying your model.
Headers
ﾉ
Expand table
Response body
{
  "createdDateTime": "2022-04-18T13:53:03Z",
  "lastModifiedDateTime": "2022-04-18T13:53:03Z",
  "lastTrainedDateTime": "2022-04-18T14:14:28Z",
  "lastDeployedDateTime": "2022-04-18T14:49:01Z",
  "projectKind": "Orchestration",
  "projectName": "{PROJECT-NAME}",
  "description": "This is a sample Orchestration project.",
  "language": "{LANGUAGE-CODE}"
}
Next steps
\nIn this article, you have learned how to use the export and import APIs to replicate your project
to a secondary Language resource in other region. Next, explore the API reference docs to see
what else you can do with authoring APIs.
Authoring REST API reference
Runtime prediction REST API reference
\nThe "None" intent in orchestration
workflow
06/21/2025
Every project in orchestration workflow includes a default None intent. The None intent is a
required intent and can't be deleted or renamed. The intent is meant to categorize any
utterances that do not belong to any of your other custom intents.
An utterance can be predicted as the None intent if the top scoring intent's score is lower than
the None score threshold. It can also be predicted if the utterance is similar to examples added
to the None intent.
You can go to the project settings of any project and set the None score threshold. The
threshold is a decimal score from 0.0 to 1.0.
For any query and utterance, the highest scoring intent ends up lower than the threshold
score, the top intent will be automatically replaced with the None intent. The scores of all the
other intents remain unchanged.
The score should be set according to your own observations of prediction scores, as they may
vary by project. A higher threshold score forces the utterances to be more similar to the
examples you have in your training data.
When you export a project's JSON file, the None score threshold is defined in the "settings"
parameter of the JSON as the "confidenceThreshold", which accepts a decimal value between
0.0 and 1.0.
The default score for Orchestration Workflow projects is set at 0.5 when creating new project in
Language Studio.
The None intent is also treated like any other intent in your project. If there are utterances that
you want predicted as None, consider adding similar examples to them in your training data.
None score threshold
７ Note
During model evaluation of your test set, the None score threshold is not applied.
Adding examples to the None intent
\nFor example, if you would like to categorize utterances that are not important to your project
as None, then add those utterances to your intent.
Orchestration workflow overview
Next steps
\nData formats accepted by orchestration
workflow
06/21/2025
When data is used by your model for learning, it expects the data to be in a specific format.
When you tag your data in Language Studio, it gets converted to the JSON format described in
this article. You can also manually tag your files.
If you upload a tags file, it should follow this format.
JSON
JSON file format
{
  "projectFileVersion": "{API-VERSION}",
  "stringIndexType": "Utf16CodeUnit",
  "metadata": {
    "projectKind": "Orchestration",
    "projectName": "{PROJECT-NAME}",
    "multilingual": false,
    "description": "This is a description",
    "language": "{LANGUAGE-CODE}"
  },
  "assets": {
    "projectKind": "Orchestration",
    "intents": [
      {
        "category": "{INTENT1}",
        "orchestration": {
          "targetProjectKind": "Luis|Conversation|QuestionAnswering",
          "luisOrchestration": {
            "appId": "{APP-ID}",
            "appVersion": "0.1",
            "slotName": "production"
          },
          "conversationOrchestration": {
            "projectName": "{PROJECT-NAME}",
            "deploymentName": "{DEPLOYMENT-NAME}"
          },
          "questionAnsweringOrchestration": {
            "projectName": "{PROJECT-NAME}"
          }
        }
      }
    ],
    "utterances": [
      {