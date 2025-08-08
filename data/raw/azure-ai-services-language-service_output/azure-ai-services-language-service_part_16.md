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
The name of your project. This value is case-
sensitive.
myProject
{JOB-ID}
The ID for locating your model's training status.
This value is in the location  header value you
received in the previous step.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are calling. The value
referenced here is for the latest version
released. See model lifecycle to learn more
about other available API versions.
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Once you send the request, you’ll get the following response.
JSON
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/train/jobs/{JOB-ID}?
api-version={API-VERSION}
ﾉ
Expand table
Headers
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
\nThis is the step where you make your trained model available form consumption via the runtime
prediction API
.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY}  that you obtained in the first step.
Submit a PUT request using the following URL, headers, and JSON body to submit a deployment
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
The name of your project. This value is case-
sensitive.
myProject
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
Use the same deployment name as your primary project for easier maintenance and minimal
changes to your system to handle redirecting your traffic.
Submit deployment job
{Endpoint}/language/authoring/analyze-
text/projects/{projectName}/deployments/{deploymentName}?api-version={API-VERSION}
ﾉ
Expand table
\nPlaceholder
Value
Example
{DEPLOYMENT-
NAME}
The name of your deployment. This value is
case-sensitive.
staging
{API-
VERSION}
The version of the API you are calling. The
value referenced here is for the latest version
released. Learn more about other available
API versions
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the following JSON in the body of your request. Use the name of the model you to assign to the
deployment.
JSON
Key
Placeholder
Value
Example
trainedModelLabel
{MODEL-
NAME}
The model name that will be assigned to your deployment. You
can only assign successfully trained models. This value is case-
sensitive.
myModel
Once you send your API request, you’ll receive a 202  response indicating that the job was submitted
correctly. In the response headers, extract the operation-location  value. It will be formatted like this:
rest
Headers
ﾉ
Expand table
Request body
{
  "trainedModelLabel": "{MODEL-NAME}"
}
ﾉ
Expand table
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
NAME}/deployments/{DEPLOYMENT-NAME}/jobs/{JOB-ID}?api-version={API-VERSION}
\n{JOB-ID} is used to identify your request, since this operation is asynchronous. You can use this URL
to get the deployment status.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to query the status of the deployment job. You can use the URL you
received from the previous step, or replace the placeholder values below with your own values.
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
The name of your project. This value is case-
sensitive.
myProject
{DEPLOYMENT-
NAME}
The name of your deployment. This value is
case-sensitive.
staging
{JOB-ID}
The ID for locating your model's training
status. This is in the location  header value
you received in the previous step.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-
VERSION}
The version of the API you are calling. The
value referenced here is for the latest version
released. Learn more about other available
API versions
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Get the deployment status
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
NAME}/deployments/{DEPLOYMENT-NAME}/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
\nOnce you send the request, you will get the following response. Keep polling this endpoint until the
status parameter changes to "succeeded". You should get a 200  code to indicate the success of the
request.
JSON
Within your system, at the step where you call runtime prediction API
 check for the response code
returned from the submit task API. If you observe a consistent failure in submitting the request, this
could indicate an outage in your primary region. Failure once doesn't mean an outage, it may be
transient issue. Retry submitting the job through the secondary resource you have created. For the
second request use your {SECONDARY-ENDPOINT}  and {SECONDARY-RESOURCE-KEY} , if you have followed
the steps above, {PROJECT-NAME}  and {DEPLOYMENT-NAME}  would be the same so no changes are
required to the request body.
In case you revert to using your secondary resource you will observe slight increase in latency
because of the difference in regions where your model is deployed.
Maintaining the freshness of both projects is an important part of process. You need to frequently
check if any updates were made to your primary project so that you move them over to your
secondary project. This way if your primary region fail and you move into the secondary region you
should expect similar model performance since it already contains the latest updates. Setting the
frequency of checking if your projects are in sync is an important choice, we recommend that you do
this check daily in order to guarantee the freshness of data in your secondary model.
Use the following url to get your project details, one of the keys returned in the body indicates the
last modified date of the project. Repeat the following step twice, one for your primary project and
another for your secondary project and compare the timestamp returned for both of them to check
if they are out of sync.
Response Body
{
    "jobId":"{JOB-ID}",
    "createdDateTime":"{CREATED-TIME}",
    "lastUpdatedDateTime":"{UPDATED-TIME}",
    "expirationDateTime":"{EXPIRATION-TIME}",
    "status":"running"
}
Changes in calling the runtime
Check if your projects are out of sync
Get project details
\nUse the following GET request to get your project details. Replace the placeholder values below with
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
The name for your project. This value is case-
sensitive.
myProject
{API-
VERSION}
The version of the API you are calling. The
value referenced here is for the latest version
released. Learn more about other available API
versions
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
JSON
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}?api-version={API-
VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Response body
    {
        "createdDateTime": "2021-10-19T23:24:41.572Z",
        "lastModifiedDateTime": "2021-10-19T23:24:41.572Z",
        "lastTrainedDateTime": "2021-10-19T23:24:41.572Z",
        "lastDeployedDateTime": "2021-10-19T23:24:41.572Z",
        "projectKind": "customMultiLabelClassification",
        "storageInputContainerName": "{CONTAINER-NAME}",
        "projectName": "{PROJECT-NAME}",
        "multilingual": false,
        "description": "Project description",
\nOnce you send your API request, you will receive a 200  response indicating success and JSON
response body with your project details.
Repeat the same steps for your replicated project using {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY} . Compare the returned lastModifiedDateTime  from both project. If your primary
project was modified sooner than your secondary one, you need to repeat the steps of exporting,
importing, training and deploying your model.
In this article, you have learned how to use the export and import APIs to replicate your project to a
secondary Language resource in other region. Next, explore the API reference docs to see what else
you can do with authoring APIs.
Authoring REST API reference
Runtime prediction REST API reference
        "language": "{LANGUAGE-CODE}"
    }
Next steps
\nDeploy custom language projects to
multiple regions
Article • 04/29/2025
Custom language service features enable you to deploy your project to more than one region.
This capability makes it much easier to access your project globally while you manage only one
instance of your project in one place. As of November 2024, custom language service features
also enable you to deploy your project to multiple resources within a single region via the API,
so that you can use your custom model wherever you need.
Before you deploy a project, you can assign deployment resources in other regions. Each
deployment resource is a different Language resource from the one that you use to author
your project. You deploy to those resources and then target your prediction requests to that
resource in their respective regions and your queries are served directly from that region.
When you create a deployment, you can select which of your assigned deployment resources
and their corresponding regions you want to deploy to. The model you deploy is then
replicated to each region and accessible with its own endpoint dependent on the deployment
resource's custom subdomain.
Suppose you want to make sure your project, which is used as part of a customer support
chatbot, is accessible by customers across the United States and India. You author a project
with the name ContosoSupport  by using a West US 2 Language resource named MyWestUS2 .
Before deployment, you assign two deployment resources to your project: MyEastUS  and
MyCentralIndia  in East US and Central India, respectively.
When you deploy your project, you select all three regions for deployment: the original West
US 2 region and the assigned ones through East US and Central India.
７ Note
This article applies to the following custom features in Azure AI Language:
Conversational language understanding
Custom text classification
Custom named entity recognition (NER)
Orchestration workflow
Example
\nYou now have three different endpoint URLs to access your project in all three regions:
West US 2: https://mywestus2.cognitiveservices.azure.com/language/:analyze-
conversations
East US: https://myeastus.cognitiveservices.azure.com/language/:analyze-
conversations
Central India: https://mycentralindia.cognitiveservices.azure.com/language/:analyze-
conversations
The same request body to each of those different URLs serves the exact same response directly
from that region.
Assigning deployment resources requires Microsoft Entra authentication. Microsoft Entra ID is
used to confirm that you have access to the resources that you want to assign to your project
for multiregion deployment. In Language Studio, you can automatically enable Microsoft Entra
authentication
 by assigning yourself the Azure Cognitive Services Language Owner role to
your original resource. To programmatically use Microsoft Entra authentication, learn more
from the Azure AI services documentation.
Your project name and resource are used as its main identifiers. A Language resource can only
have a specific project name in each resource. Any other projects with the same name can't be
deployed to that resource.
For example, if a project ContosoSupport  was created by the resource MyWestUS2  in West US 2
and deployed to the resource MyEastUS  in East US, the resource MyEastUS  can't create a
different project called ContosoSupport  and deploy a project to that region. Similarly, your
collaborators can't then create a project ContosoSupport  with the resource MyCentralIndia  in
Central India and deploy it to either MyWestUS2  or MyEastUS .
You can only swap deployments that are available in the exact same regions. Otherwise,
swapping fails.
If you remove an assigned resource from your project, all of the project deployments to that
resource are deleted.
Some regions are only available for deployment and not for authoring projects.
Learn how to deploy models for:
Validations and requirements
Related content
\nConversational language understanding
Custom text classification
Custom NER
Orchestration workflow