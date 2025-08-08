4. select Deploy to start the deployment job.
5. After deployment is successful, an expiration date will appear next to it. Deployment
expiration is when your deployed model will be unavailable to be used for prediction,
which typically happens twelve months after a training configuration expires.
You can swap deployments after you've tested a model assigned to one deployment, and want
to assign it to another. Swapping deployments involves taking the model assigned to the first
deployment, and assigning it to the second deployment. Then taking the model assigned to
second deployment and assign it to the first deployment. This could be used to swap your
production  and staging  deployments when you want to take the model assigned to staging
and assign it to production .

Swap deployments
Language Studio
\n![Image](images/page131_image1.png)
\nTo swap deployments from within Language Studio
1. In Deploying a model page, select the two deployments you want to swap and select
Swap deployments from the top menu.
2. From the window that appears, select the names of the deployments you want to
swap.
To delete a deployment from within Language Studio
, go to the Deploying a model
page. Select the deployment you want to delete and select Delete deployment from the
top menu.
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
Delete deployment
Language Studio
Assign deployment resources
Language Studio
Unassign deployment resources
\nWhen you unassign or remove a deployment resource from a project, you will also delete all
the deployments that have been deployed to that resource's region.
To unassign or remove deployment resources in other regions using Language Studio
:
1. Go to the Regions tab in the Deploy a model page.
2. Select the resource you'd like to unassign.
3. Select the Remove assignment button.
4. In the window that appears, type the name of the resource you want to remove.
Use prediction API to query your model
Language Studio
Next steps
\nSend text classification requests to your
model
06/30/2025
After you've successfully deployed a model, you can query the deployment to classify text
based on the model you assigned to the deployment. You can query the deployment
programmatically Prediction API or through the client libraries (Azure SDK).
You can use Language Studio to submit the custom text classification task and visualize the
results.
To test your deployed models from within the Language Studio
:
1. Select Testing deployments from the left side menu.
2. Select the deployment you want to test. You can only test models that are assigned to
deployments.
3. For multilingual projects, from the language dropdown, select the language of the text
you are testing.
4. Select the deployment you want to query/test from the dropdown.
5. You can enter the text you want to submit to the request or upload a .txt  file to use.
6. Select Run the test from the top menu.
7. In the Result tab, you can see the extracted entities from your text and their types. You
can also view the JSON response under the JSON tab.
Test deployed model
\n1. After the deployment job is completed successfully, select the deployment you want
to use and from the top menu select Get prediction URL.

Send a text classification request to your model
 Tip
You can test your model in Language Studio by sending sample text to classify it.
Language Studio
\n![Image](images/page135_image1.png)
\n2. In the window that appears, under the Submit pivot, copy the sample request URL
and body. Replace the placeholder values such as YOUR_DOCUMENT_HERE  and
YOUR_DOCUMENT_LANGUAGE_HERE  with the actual text and language you want to process.
3. Submit the POST  cURL request in your terminal or command prompt. You'll receive a
202 response with the API results if the request was successful.
4. In the response header you receive extract {JOB-ID}  from operation-location , which
has the format: {ENDPOINT}/language/analyze-text/jobs/<JOB-ID}>
5. Back to Language Studio; select Retrieve pivot from the same window you got the
example request you got earlier and copy the sample request into a text editor.
6. Add your job ID after /jobs/  to the URL, using the ID you extracted from the
previous step.
7. Submit the GET  cURL request in your terminal or command prompt.
Custom text classification overview

Next steps
\n![Image](images/page136_image1.png)
\nBack up and recover your custom text
classification models
06/30/2025
When you create a Language resource, you specify a region for it to be created in. From then on,
your resource and all of the operations related to it take place in the specified Azure server region.
It's rare, but not impossible, to encounter a network issue that hits an entire region. If your solution
needs to always be available, then you should design it to either fail-over into another region. This
requires two Azure AI Language resources in different regions and the ability to sync custom models
across regions.
If your app or business depends on the use of a custom text classification model, we recommend
that you create a replica of your project into another supported region. So that if a regional outage
occurs, you can then access your model in the other fail-over region where you replicated your
project.
Replicating a project means that you export your project metadata and assets and import them into
a new project. This only makes a copy of your project settings and tagged data. You still need to
train and deploy the models to be available for use with prediction APIs
.
In this article, you will learn to how to use the export and import APIs to replicate your project from
one resource to another existing in different supported geographical regions, guidance on keeping
your projects in sync and changes needed to your runtime consumption.
Two Azure AI Language resources in different Azure regions. Create a Language resource and
connect them to an Azure storage account. It's recommended that you connect both of your
Language resources to the same storage account, though this might introduce slightly higher
latency when importing your project, and training a model.
Use the following steps to get the keys and endpoint of your primary and secondary resources.
These will be used in the following steps.
Go to your resource overview page in the Azure portal
From the menu on the left side, select Keys and Endpoint. You will use the endpoint and key
for the API requests
Prerequisites
Get your resource keys endpoint
\nStart by exporting the project assets from the project in your primary resource.
Replace the placeholders in the following request with your {PRIMARY-ENDPOINT}  and {PRIMARY-
RESOURCE-KEY}  that you obtained in the first step.
Create a POST request using the following URL, headers, and JSON body to export your project.
Use the following URL when creating your API request. Replace the placeholder values below with
your own values.
rest

 Tip
Keep a note of keys and endpoints for both primary and secondary resources. Use these values
to replace the following placeholders: {PRIMARY-ENDPOINT} , {PRIMARY-RESOURCE-KEY} ,
{SECONDARY-ENDPOINT}  and {SECONDARY-RESOURCE-KEY} . Also take note of your project name,
your model name and your deployment name. Use these values to replace the following
placeholders: {PROJECT-NAME} , {MODEL-NAME}  and {DEPLOYMENT-NAME} .
Export your primary project assets
Submit export job
Request URL
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/:export?
\n![Image](images/page138_image1.png)
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
MyProject
{API-
VERSION}
The version of the API you are calling. The
value referenced here is the latest model
version released.
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the following JSON in your request body specifying that you want to export all the assets.
JSON
Once you send your API request, you’ll receive a 202  response indicating that the job was submitted
correctly. In the response headers, extract the operation-location  value. It will be formatted like this:
rest
{JOB-ID}  is used to identify your request, since this operation is asynchronous. You’ll use this URL to
get the export job status.
stringIndexType=Utf16CodeUnit&api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Body
{
  "assetsToExport": ["*"]
}
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/export/jobs/{JOB-
ID}?api-version={API-VERSION}
\nReplace the placeholders in the following request with your {PRIMARY-ENDPOINT}  and {PRIMARY-
RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to get the status of exporting your project assets. Replace the
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
The name of your project. This value is case-
sensitive.
myProject
{JOB-ID}
The ID for locating your model's training
status. This is in the location  header value you
received in the previous step.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
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
Get export job status
Request URL
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/export/jobs/{JOB-
ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Response body