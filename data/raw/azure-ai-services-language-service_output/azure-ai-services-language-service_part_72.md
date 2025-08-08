2. In the window that appears, under the Submit pivot, copy the sample request URL
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
Frequently asked questions
Next steps
\nBack up and recover your custom NER
models
06/30/2025
When you create a Language resource, you specify a region for it to be created in. From then
on, your resource and all of the operations related to it take place in the specified Azure server
region. It's rare, but not impossible, to encounter a network issue that affects an entire region.
If your solution needs to always be available, then you should design it to fail over into another
region. This requires two Azure AI Language resources in different regions and synchronizing
custom models across them.
If your app or business depends on the use of a custom NER model, we recommend that you
create a replica of your project in an additional supported region. If a regional outage occurs,
you can then access your model in the other fail-over region where you replicated your project.
Replicating a project means that you export your project metadata and assets, and import
them into a new project. This only makes a copy of your project settings and tagged data. You
still need to train and deploy the models to be available for use with prediction APIs
.
In this article, you will learn to how to use the export and import APIs to replicate your project
from one resource to another existing in different supported geographical regions, guidance
on keeping your projects in sync and changes needed to your runtime consumption.
Two Azure AI Language resources in different Azure regions. Create your resources and
connect them to an Azure storage account. It's recommended that you connect each of
your Language resources to different storage accounts. Each storage account should be
located in the same respective regions that your separate Language resources are in. You
can follow the quickstart to create an additional Language resource and storage account.
Use the following steps to get the keys and endpoint of your primary and secondary resources.
These will be used in the following steps.
1. Go to your resource overview page in the Azure portal
2. From the menu on the left side, select Keys and Endpoint. You will use the endpoint and
key for the API requests
Prerequisites
Get your resource keys endpoint
\nStart by exporting the project assets from the project in your primary resource.
Replace the placeholders in the following request with your {PRIMARY-ENDPOINT}  and {PRIMARY-
RESOURCE-KEY}  that you obtained in the first step.
Create a POST request using the following URL, headers, and JSON body to export your
project.
Use the following URL when creating your API request. Replace the placeholder values below
with your own values.

 Tip
Keep a note of keys and endpoints for both primary and secondary resources as well as
the primary and secondary container names. Use these values to replace the following
placeholders: {PRIMARY-ENDPOINT} , {PRIMARY-RESOURCE-KEY} , {PRIMARY-CONTAINER-NAME} ,
{SECONDARY-ENDPOINT} , {SECONDARY-RESOURCE-KEY} , and {SECONDARY-CONTAINER-NAME} . Also
take note of your project name, your model name and your deployment name. Use these
values to replace the following placeholders: {PROJECT-NAME} , {MODEL-NAME}  and
{DEPLOYMENT-NAME} .
Export your primary project assets
Submit export job
Request URL
\n![Image](images/page713_image1.png)
\nrest
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
MyProject
{API-
VERSION}
The version of the API you are calling.
The value referenced here is the latest
model version released.
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the following JSON in your request body specifying that you want to export all the assets.
JSON
Once you send your API request, you’ll receive a 202  response indicating that the job was
submitted correctly. In the response headers, extract the operation-location  value. It will be
formatted like this:
rest
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/:export?
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
\n{JOB-ID}  is used to identify your request, since this operation is asynchronous. You’ll use this
URL to get the export job status.
Replace the placeholders in the following request with your {PRIMARY-ENDPOINT}  and {PRIMARY-
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
The name of your project. This value is
case-sensitive.
myProject
{JOB-ID}
The ID for locating your model's training
status. This is in the location  header value
you received in the previous step.
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
NAME}/export/jobs/{JOB-ID}?api-version={API-VERSION}
Get export job status
Request URL
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-
NAME}/export/jobs/{JOB-ID}?api-version={API-VERSION}
ﾉ
Expand table
Headers
\nUse the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
JSON
Use the URL from the resultUrl  key in the body to view the exported assets from this job.
Submit a GET request using the {RESULT-URL}  you received from the previous step to view the
results of the export job.
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Copy the response body as you will use it as the body for the next import job.
ﾉ
Expand table
Response body
{
  "resultUrl": "{RESULT-URL}",
  "jobId": "string",
  "createdDateTime": "2021-10-19T23:24:41.572Z",
  "lastUpdatedDateTime": "2021-10-19T23:24:41.572Z",
  "expirationDateTime": "2021-10-19T23:24:41.572Z",
  "status": "unknown",
  "errors": [
    {
      "code": "unknown",
      "message": "string"
    }
  ]
}
Get export results
Headers
ﾉ
Expand table
\nNow go ahead and import the exported project assets in your new project in the secondary
region so you can replicate it.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT} , {SECONDARY-
RESOURCE-KEY} , and {SECONDARY-CONTAINER-NAME}  that you obtained in the first step.
Submit a POST request using the following URL, headers, and JSON body to import your labels
file. Make sure that your labels file follow the accepted format.
If a project with the same name already exists, the data of that project is replaced.
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
Import to a new project
Submit import job
{Endpoint}/language/authoring/analyze-text/projects/{projectName}/:import?api-
version={API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
\nKey
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Use the following JSON in your request. Replace the placeholder values below with your own
values.
JSON
Body
{
    "projectFileVersion": "{API-VERSION}",
    "stringIndexType": "Utf16CodeUnit",
    "metadata": {
        "projectName": "{PROJECT-NAME}",
        "projectKind": "CustomEntityRecognition",
        "description": "Trying out custom NER",
        "language": "{LANGUAGE-CODE}",
        "multilingual": true,
        "storageInputContainerName": "{CONTAINER-NAME}",
        "settings": {}
    },
    "assets": {
    "projectKind": "CustomEntityRecognition",
        "entities": [
            {
                "category": "Entity1"
            },
            {
                "category": "Entity2"
            }
        ],
        "documents": [
            {
                "location": "{DOCUMENT-NAME}",
                "language": "{LANGUAGE-CODE}",
                "dataset": "{DATASET}",
                "entities": [
                    {
                        "regionOffset": 0,
                        "regionLength": 500,
                        "labels": [
                            {
                                "category": "Entity1",
                                "offset": 25,
                                "length": 10
                            },
                            {
                                "category": "Entity2",
                                "offset": 120,
\nKey
Placeholder
Value
Example
api-version
{API-VERSION}
The version of the
API you are calling.
The version used
here must be the
same API version in
the URL. Learn more
about other
available API
versions
2022-03-01-preview
projectName
{PROJECT-NAME}
The name of your
project. This value is
case-sensitive.
myProject
projectKind
CustomEntityRecognition
Your project kind.
CustomEntityRecognition
language
{LANGUAGE-CODE}
A string specifying
the language code
for the documents
used in your project.
en-us
                                "length": 8
                            }
                        ]
                    }
                ]
            },
            {
                "location": "{DOCUMENT-NAME}",
                "language": "{LANGUAGE-CODE}",
                "dataset": "{DATASET}",
                "entities": [
                    {
                        "regionOffset": 0,
                        "regionLength": 100,
                        "labels": [
                            {
                                "category": "Entity2",
                                "offset": 20,
                                "length": 5
                            }
                        ]
                    }
                ]
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
If your project is a
multilingual project,
choose the
language code of
the majority of the
documents.
multilingual
true
A boolean value
that enables you to
have documents in
multiple languages
in your dataset and
when your model is
deployed you can
query the model in
any supported
language (not
necessarily included
in your training
documents. See
language support
for information on
multilingual
support.
true
storageInputContainerName
{CONTAINER-NAME}
The name of your
Azure storage
container where you
have uploaded your
documents.
myContainer
entities
Array containing all
the entity types you
have in the project.
These are the entity
types that will be
extracted from your
documents into.
documents
Array containing all
the documents in
your project and list
of the entities
labeled within each
document.
[]
location
{DOCUMENT-NAME}
The location of the
documents in the
storage container.
doc1.txt