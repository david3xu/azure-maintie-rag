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
When you send a successful deployment request, the full request URL for checking the job's
status (including your endpoint, project name, and job ID) is contained in the response's
operation-location  header.
Use the following GET request to get the status of your deployment job. Replace the
placeholder values with your own values.
rest
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
\nPlaceholder
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
Once you send the request, you'll get the following response. Keep polling this endpoint until
the status parameter changes to "succeeded".
JSON
ﾉ
Expand table
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
Changes in calling the runtime
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
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating
your API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
Check if your projects are out of sync
Get project details
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-NAME}?api-
version={API-VERSION}
ﾉ
Expand table
\nPlaceholder
Value
Example
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
In this article, you have learned how to use the export and import APIs to replicate your project
to a secondary Language resource in other region. Next, explore the API reference docs to see
Headers
ﾉ
Expand table
Response body
{
  "createdDateTime": "2022-04-18T13:53:03Z",
  "lastModifiedDateTime": "2022-04-18T13:53:03Z",
  "lastTrainedDateTime": "2022-04-18T14:14:28Z",
  "lastDeployedDateTime": "2022-04-18T14:49:01Z",
  "projectKind": "Conversation",
  "projectName": "{PROJECT-NAME}",
  "multilingual": true,
  "description": "This is a sample conversation project.",
  "language": "{LANGUAGE-CODE}"
}
Next steps
\nwhat else you can do with authoring APIs.
Authoring REST API reference
Runtime prediction REST API reference
\nMigrate from Language Understanding
(LUIS) to conversational language
understanding (CLU)
05/23/2025
Conversational language understanding (CLU) is a cloud-based AI offering in Azure AI
Language. It's the newest generation of Language Understanding (LUIS) and offers backwards
compatibility with previously created LUIS applications. CLU employs state-of-the-art machine
learning intelligence to allow users to build a custom natural language understanding model
for predicting intents and entities in conversational utterances.
CLU offers the following advantages over LUIS:
Improved accuracy with state-of-the-art machine learning models for better intent
classification and entity extraction. LUIS required more examples to generalize certain
concepts in intents and entities, while CLU's more advanced machine learning reduces the
burden on customers by requiring less data.
Multilingual support for model learning and training. Train projects in one language and
immediately predict intents and entities across 96 languages.
Ease of integration with different CLU and custom question answering projects using
orchestration workflow.
The ability to add testing data within the experience using Language Studio and APIs for
model performance evaluation prior to deployment.
To get started, you can use CLU directly or migrate your LUIS application.
The following table presents a side-by-side comparison between the features of LUIS and CLU.
It also highlights the changes to your LUIS application after migrating to CLU. Select the linked
concept to learn more about the changes.
LUIS features
CLU features
Post migration
Machine-learned and
Structured ML entities
Learned entity
components
Machine-learned entities without subentities are
transferred as CLU entities. Structured ML entities only
transfer leaf nodes (lowest level subentities that don't
have their own subentities) as entities in CLU. The
Comparison between LUIS and CLU
ﾉ
Expand table
\nLUIS features
CLU features
Post migration
name of the entity in CLU is the name of the subentity
concatenated with the parent. For example, Order.Size
List, regex, and prebuilt
entities
List, regex, and
prebuilt entity
components
List, regex, and prebuilt entities are transferred as
entities in CLU with a populated entity component
based on the entity type.
Pattern.Any  entities
Not currently
available
Pattern.Any  entities are removed.
Single culture for each
application
Multilingual models
enable multiple
languages for each
project.
The primary language of your project is set as your
LUIS application culture. Your project can be trained to
extend to different languages.
Entity roles
Roles are no longer
needed.
Entity roles are transferred as entities.
Settings for: normalize
punctuation, normalize
diacritics, normalize
word form, use all
training data
Settings are no
longer needed.
Settings aren't transferred.
Patterns and phrase list
features
Patterns and Phrase
list features are no
longer needed.
Patterns and phrase list features aren't transferred.
Entity features
Entity components
List or prebuilt entities added as features to an entity
are transferred as added components to that entity.
Entity features aren't transferred for intents.
Intents and utterances
Intents and
utterances
All intents and utterances are transferred. Utterances
are labeled with their transferred entities.
Application GUIDs
Project names
A project is created for each migrating application with
the application name. Any special characters in the
application names are removed in CLU.
Versioning
Every time you train,
a model is created
and acts as a version
of your project.
A project is created for the selected application
version.
Evaluation using batch
testing
Evaluation using
testing sets
Adding your testing dataset is required.
Role-Based Access
Control (RBAC) for LUIS
resources
Role-Based Access
Control (RBAC)
Language resource RBAC must be manually added
after migration.
\nLUIS features
CLU features
Post migration
available for
Language resources
Single training mode
Standard and
advanced training
modes
Training is required after application migration.
Two publishing slots
and version publishing
Ten deployment slots
with custom naming
Deployment is required after the application’s
migration and training.
LUIS authoring APIs and
SDK support in .NET,
Python, Java, and
Node.js
CLU Authoring REST
APIs
.
For more information, see the quickstart article for
information on the CLU authoring APIs. Refactoring is
necessary to use the CLU authoring APIs.
LUIS Runtime APIs and
SDK support in .NET,
Python, Java, and
Node.js
CLU Runtime APIs
.
CLU Runtime SDK
support for .NET and
Python.
See how to call the API for more information.
Refactoring is necessary to use the CLU runtime API
response.
Use the following steps to migrate your LUIS application using either the LUIS portal or REST
API.
Follow these steps to begin migration using the LUIS Portal
:
1. After logging into the LUIS portal, click the button on the banner at the top of the
screen to launch the migration wizard. The migration copies your selected LUIS
applications to CLU.
The migration overview tab provides a brief explanation of conversational language
understanding and its benefits. Press Next to proceed.
Migrate your LUIS applications
LUIS portal
Migrate your LUIS applications using the LUIS
portal

\n2. Determine the Language resource that you wish to migrate your LUIS application to.
If you have already created your Language resource, select your Azure subscription
followed by your Language resource, and then select Next. If you don't have a
Language resource, click the link to create a new Language resource. Afterwards,
select the resource and select Next.

\n![Image](images/page309_image1.png)
\n3. Select all your LUIS applications that you want to migrate, and specify each of their
versions. Select Next. After selecting your application and version, you're prompted
with a message informing you of any features that won't be carried over from your
LUIS application.

７ Note
Special characters aren't supported by conversational language understanding.
Any special characters in your selected LUIS application names are removed in
\n![Image](images/page310_image1.png)