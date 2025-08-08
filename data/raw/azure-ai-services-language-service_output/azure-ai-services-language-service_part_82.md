You can deploy your project to multiple regions by assigning different Language resources that
exist in different regions.
To assign deployment resources in other regions in Language Studio
:
1. Make sure you've assigned yourself as a Cognitive Services Language Owner
 to the
resource you used to create the project.
2. Go to the Deploying a model page in Language Studio.
3. Select the Regions tab.
4. Select Add deployment resource.
5. Select a Language resource in another region.
You are now ready to deploy your project to the regions where you have assigned
resources.
When unassigning or removing a deployment resource from a project, you will also delete all
the deployments that have been deployed to that resource's region.
To unassign or remove deployment resources in other regions using Language Studio
:
1. Go to the Regions tab in the Deploy a model page.
2. Select the resource you'd like to unassign.
3. Select the Remove assignment button.
4. In the window that appears, type the name of the resource you want to remove.
Use prediction API to query your model
Assign deployment resources
Language Studio
Unassign deployment resources
Language Studio
Next steps
\nQuery deployment for intent predictions
06/21/2025
After the deployment is added successfully, you can query the deployment for intent and
entities predictions from your utterance based on the model you assigned to the deployment.
You can query the deployment programmatically Prediction API
 or through the Client
libraries (Azure SDK).
You can use Language Studio to submit an utterance, get predictions and visualize the results.
To test your model from Language Studio
1. Select Testing deployments from the left side menu.
2. Select the model you want to test. You can only test models that are assigned to
deployments.
3. From deployment name dropdown, select your deployment name.
4. In the text box, enter an utterance to test.
5. From the top menu, select Run the test.
6. After you run the test, you should see the response of the model in the result. You can
view the results in entities cards view, or view it in JSON format.
Test deployed model
\n1. After the deployment job is completed successfully, select the deployment you want
to use and from the top menu select Get prediction URL.
2. In the window that appears, copy the sample request URL and body into your
command line. Replace <YOUR_QUERY_HERE>  with the actual text you want to send to

Send an orchestration workflow request
Language Studio

\n![Image](images/page813_image1.png)

![Image](images/page813_image2.png)
\nextract intents and entities from.
3. Submit the POST  cURL request in your terminal or command prompt. You'll receive a
202 response with the API results if the request was successful.
Orchestration workflow overview
Next steps
\nBack up and recover your conversational
language understanding models
06/30/2025
When you create a Language resource in the Azure portal, you specify a region for it to be
created in. From then on, your resource and all of the operations related to it take place in the
specified Azure server region. It's rare, but not impossible, to encounter a network issue that
hits an entire region. If your solution needs to always be available, then you should design it to
either fail-over into another region. This requires two Azure AI Language resources in different
regions and the ability to sync your CLU models across regions.
If your app or business depends on the use of a CLU model, we recommend that you create a
replica of your project into another supported region. So that if a regional outage occurs, you
can then access your model in the other fail-over region where you replicated your project.
Replicating a project means that you export your project metadata and assets and import them
into a new project. This only makes a copy of your project settings, intents, entities and
utterances. You still need to train and deploy the models to be available for use with runtime
APIs
.
In this article, you will learn to how to use the export and import APIs to replicate your project
from one resource to another existing in different supported geographical regions, guidance
on keeping your projects in sync and changes needed to your runtime consumption.
Two Azure AI Language resources in different Azure regions, each of them in a different
region.
Use the following steps to get the keys and endpoint of your primary and secondary resources.
These will be used in the following steps.
Go to your resource overview page in the Azure portal
. From the menu on the left side,
select Keys and Endpoint. You will use the endpoint and key for the API requests
Prerequisites
Get your resource keys endpoint
\nStart by exporting the project assets from the project in your primary resource.
Replace the placeholders in the following request with your {PRIMARY-ENDPOINT}  and {PRIMARY-
RESOURCE-KEY}  that you obtained in the first step.
Create a POST  request by using the following URL, headers, and JSON body to export your
project.
Use the following URL when you create your API request. Replace the placeholder values with
your own values.

 Tip
Keep a note of keys and endpoints for both primary and secondary resources. Use these
values to replace the following placeholders: {PRIMARY-ENDPOINT} , {PRIMARY-RESOURCE-
KEY} , {SECONDARY-ENDPOINT}  and {SECONDARY-RESOURCE-KEY} . Also take note of your project
name, your model name and your deployment name. Use these values to replace the
following placeholders: {PROJECT-NAME} , {MODEL-NAME}  and {DEPLOYMENT-NAME} .
Export your primary project assets
Submit export job
Request URL
\n![Image](images/page816_image1.png)
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
value is case sensitive.
EmailApp
{API-
VERSION}
The version of the API that you're
calling.
2023-04-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
After you send your API request, you receive a 202  response that indicates success. In the
response headers, extract the operation-location  value. The value is formatted like this
example:
rest
JOB-ID  is used to identify your request because this operation is asynchronous. Use this URL to
get the exported project JSON by using the same authentication method.
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
\nReplace the placeholders in the following request with your {PRIMARY-ENDPOINT}  and {PRIMARY-
RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to query the status of your export job. You can use the URL you
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
\nUse the url from the resultUrl  key in the body to view the exported assets from this job.
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
Submit a POST  request by using the following URL, headers, and JSON body to import your
project.
conversations/projects/{PROJECT-NAME}/export/jobs/xxxxxx-xxxxx-xxxxx-xx/result?
api-version={API-VERSION}",
  "jobId": "xxxx-xxxxx-xxxxx-xxx",
  "createdDateTime": "2022-04-18T15:23:07Z",
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
\nUse the following URL when you create your API request. Replace the placeholder values with
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