JSON
Use the url from the resultUrl  key in the body to view the exported assets from this job.
Submit a GET request using the {RESULT-URL}  you received from the previous step to view the
results of the export job.
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Copy the response body as you will use it as the body for the next import job.
Now go ahead and import the exported project assets in your new project in the secondary region
so you can replicate it.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY}  that you obtained in the first step.
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
Import to a new project
Submit import job
\nSubmit a POST request using the following URL, headers, and JSON body to import your labels file.
Make sure that your labels file follow the accepted format.
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
Use the following JSON in your request. Replace the placeholder values below with your own values.
JSON
{Endpoint}/language/authoring/analyze-text/projects/{projectName}/:import?api-version=
{API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Body
Multi label classification
{
  "projectFileVersion": "{API-VERSION}",
  "stringIndexType": "Utf16CodeUnit",
  "metadata": {
\nKey
Placeholder
Value
Example
api-version
{API-VERSION}
The version
of the API
you are
calling. The
2022-05-01
    "projectName": "{PROJECT-NAME}",
    "storageInputContainerName": "{CONTAINER-NAME}",
    "projectKind": "customMultiLabelClassification",
    "description": "Trying out custom multi label text classification",
    "language": "{LANGUAGE-CODE}",
    "multilingual": true,
    "settings": {}
  },
  "assets": {
    "projectKind": "customMultiLabelClassification",
    "classes": [
      {
        "category": "Class1"
      },
      {
        "category": "Class2"
      }
    ],
    "documents": [
      {
        "location": "{DOCUMENT-NAME}",
        "language": "{LANGUAGE-CODE}",
        "dataset": "{DATASET}",
        "classes": [
          {
            "category": "Class1"
          },
          {
            "category": "Class2"
          }
        ]
      },
      {
        "location": "{DOCUMENT-NAME}",
        "language": "{LANGUAGE-CODE}",
        "dataset": "{DATASET}",
        "classes": [
          {
            "category": "Class2"
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
version
used here
must be
the same
API version
in the URL.
Learn more
about
other
available
API
versions
projectName
{PROJECT-NAME}
The name
of your
project.
This value
is case-
sensitive.
myProject
projectKind
customMultiLabelClassification
Your
project
kind.
customMultiLabelClassification
language
{LANGUAGE-CODE}
A string
specifying
the
language
code for
the
documents
used in
your
project. If
your
project is a
multilingual
project,
choose the
language
code of the
majority of
the
documents.
See
language
support to
learn more
about
en-us
\nKey
Placeholder
Value
Example
multilingual
support.
multilingual
true
A boolean
value that
enables
you to have
documents
in multiple
languages
in your
dataset and
when your
model is
deployed
you can
query the
model in
any
supported
language
(not
necessarily
included in
your
training
documents.
See
language
support to
learn more
about
multilingual
support.
true
storageInputContainerName
{CONTAINER-NAME}
The name
of your
Azure
storage
container
where you
have
uploaded
your
documents.
myContainer
classes
[]
Array
containing
all the
classes you
have in the
[]
\nKey
Placeholder
Value
Example
project.
These are
the classes
you want
to classify
your
documents
into.
documents
[]
Array
containing
all the
documents
in your
project and
what the
classes
labeled for
this
document.
[]
location
{DOCUMENT-NAME}
The
location of
the
documents
in the
storage
container.
Since all
the
documents
are in the
root of the
container
this should
be the
document
name.
doc1.txt
dataset
{DATASET}
The test set
to which
this
document
will go to
when split
before
training.
See How to
train a
model for
more
Train
\nKey
Placeholder
Value
Example
information
on data
splitting.
Possible
values for
this field
are Train
and Test .
Once you send your API request, you’ll receive a 202  response indicating that the job was submitted
correctly. In the response headers, extract the operation-location  value. It will be formatted like this:
rest
{JOB-ID}  is used to identify your request, since this operation is asynchronous. You’ll use this URL to
get the import job status.
Possible error scenarios for this request:
The selected resource doesn't have proper permissions for the storage account.
The storageInputContainerName  specified doesn't exist.
Invalid language code is used, or if the language code type isn't string.
multilingual  value is a string and not a boolean.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to get the status of your importing your project. Replace the
placeholder values below with your own values.
rest
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/import/jobs/{JOB-
ID}?api-version={API-VERSION}
Get import job status
Request URL
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/import/jobs/{JOB-
ID}?api-version={API-VERSION}
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
The name of your project. This value is case-
sensitive.
myProject
{JOB-ID}
The ID for locating your model's training
status. This value is in the location  header
value you received in the previous step.
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
After importing your project, you only have copied the project's assets and metadata and assets. You
still need to train your model, which will incur usage on your account.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY}  that you obtained in the first step.
Submit a POST request using the following URL, headers, and JSON body to submit a training job.
Replace the placeholder values below with your own values.
rest
ﾉ
Expand table
Headers
ﾉ
Expand table
Train your model
Submit training job
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/:train?api-version=
{API-VERSION}
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
The name of your project. This value is case-
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
Use the following JSON in your request body. The model will be given the {MODEL-NAME}  once
training is complete. Only successful training jobs will produce models.
JSON
Key
Placeholder
Value
Example
modelLabel
{MODEL-NAME}
The model name that will be assigned to your model
once trained successfully.
myModel
ﾉ
Expand table
Headers
ﾉ
Expand table
Request body
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
\nKey
Placeholder
Value
Example
trainingConfigVersion
{CONFIG-
VERSION}
This is the model version that will be used to train the
model.
2022-05-01
evaluationOptions
Option to split your data across training and testing
sets.
{}
kind
percentage
Split methods. Possible values are percentage  or
manual . See How to train a model for more
information.
percentage
trainingSplitPercentage
80
Percentage of your tagged data to be included in the
training set. Recommended value is 80 .
80
testingSplitPercentage
20
Percentage of your tagged data to be included in the
testing set. Recommended value is 20 .
20
Once you send your API request, you’ll receive a 202  response indicating that the job was submitted
correctly. In the response headers, extract the location  value. It will be formatted like this:
rest
{JOB-ID} is used to identify your request, since this operation is asynchronous. You can use this URL
to get the training status.
Replace the placeholders in the following request with your {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY}  that you obtained in the first step.
Use the following GET request to get the status of your model's training progress. Replace the
placeholder values below with your own values.
rest
７ Note
The trainingSplitPercentage  and testingSplitPercentage  are only required if Kind  is set to
percentage  and the sum of both percentages should be equal to 100.
{ENDPOINT}/language/authoring/analyze-text/projects/{PROJECT-NAME}/train/jobs/{JOB-ID}?
api-version={API-VERSION}
Get Training Status
Request URL