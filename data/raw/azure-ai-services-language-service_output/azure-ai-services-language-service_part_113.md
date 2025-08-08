We offer quickstarts in most popular programming languages, each designed to teach you
basic design patterns, and have you running code in less than 10 minutes.
Get started with the custom question answering client library
Custom question answering provides everything you need to build, manage, and deploy your
custom project.
Complete a quickstart
Next steps
\nCustom question answering project
lifecycle
06/21/2025
Custom question answering learns best in an iterative cycle of model changes, utterance
examples, deployment, and gathering data from endpoint queries.
Custom question answering projects provide a best-match answer to a user query based on
the content of the project. Creating a project is a one-time action to setting up a content
repository of questions, answers, and associated metadata. A project can be created by
crawling pre-existing content such the following sources:
FAQ pages
Product manuals
Q-A pairs
Learn how to create a project.
The project is ready for testing once it is populated with content, either editorially or through
automatic extraction. Interactive testing can be done in Language Studio, in the custom
question answering menu through the Test panel. You enter common user queries. Then you
verify that the responses returned with both the correct response and a sufficient confidence
score.
To fix low confidence scores: add alternate questions.
When a query incorrectly returns the default response: add new answers to the correct
question.
This tight loop of test-update continues until you are satisfied with the results.
Once you are done testing the project, you can deploy it to production. Deployment pushes
the latest version of the tested project to a dedicated Azure AI Search index representing the
published project. It also creates an endpoint that can be called in your application or chat bot.
Creating a project
Testing and updating your project
Deploy your project
\nDue to the deployment action, any further changes made to the test version of the project
leave the published version unaffected. The published version can be live in a production
application.
Each of these projects can be targeted for testing separately.
To be able to log the chat logs of your service and get additional analytics, you would need to
enable Azure Monitor Diagnostic Logs after you create your language resource.
Based on what you learn from your analytics, make appropriate updates to your project.
Version control for data is provided through the import/export features on the project page in
the custom question answering section of Language Studio.
You can back up a project by exporting the project, in either .tsv  or .xls  format. Once
exported, include this file as part of your regular source control check.
When you need to go back to a specific version, you need to import that file from your local
system. An exported must only be used via import on the project page. It can't be used as a file
or URL document data source. This will replace questions and answers currently in the project
with the contents of the imported file.
A project is the repository of questions and answer sets created, maintained, and used through
custom question answering. Each language resource can hold multiple projects.
A project has two states: test and published.
The test project is the version currently edited and saved. The test version has been tested for
accuracy, and for completeness of responses. Changes made to the test project don't affect the
end user of your application or chat bot. The test project is known as test  in the HTTP request.
The test  knowledge is available with Language Studio's interactive Test pane.
Monitor usage
Version control for data in your project
Test and production project
Test project
Production project
\nThe published project is the version that's used in your chat bot or application. Publishing a
project puts the content of its test version into its published version. The published project is
the version that the application uses through the endpoint. Make sure that the content is
correct and well tested. The published project is known as prod  in the HTTP request.
Next steps
\nAzure resources for custom question
answering
06/30/2025
Custom question answering uses several Azure sources, each with a different purpose.
Understanding how they are used individually allows you to plan for and select the correct
pricing tier or know when to change your pricing tier. Understanding how resources are used
in combination allows you to find and fix problems when they occur.
When you first develop a project, in the prototype phase, it is common to have a single
resource for both testing and production.
When you move into the development phase of the project, you should consider:
How many languages will your project hold?
How many regions you need your project to be available in?
How many documents will your system hold in each domain?
Typically there are three parameters you need to consider:
The throughput you need:
The throughput for custom question answering is currently capped at 10 text records
per second for both management APIs and prediction APIs.
This should also influence your Azure AI Search selection, see more details here.
Additionally, you might need to adjust Azure AI Search capacity with replicas.
Size and the number of projects: Choose the appropriate Azure search SKU
 for your
scenario. Typically, you decide the number of projects you need based on number of
Resource planning
 Tip
"Knowledge base" and "project" are equivalent terms in custom question answering and
can be used interchangeably.
Pricing tier considerations
\ndifferent subject domains. One subject domain (for a single language) should be in one
project.
With custom question answering, you have a choice to set up your language resource in a
single language or multiple languages. You can make this selection when you create your
first project in the Language Studio
.
For example, if your tier has 15 allowed indexes, you can publish 14 projects of the same
language (one index per published project). The 15th index is used for all the projects for
authoring and testing. If you choose to have projects in different languages, then you can
only publish seven projects.
Number of documents as sources: There are no limits to the number of documents you
can add as sources in custom question answering.
The following table gives you some high-level guidelines.
Azure AI Search
Limitations
Experimentation
Free Tier
Publish Up to 2 KBs, 50 MB size
Dev/Test Environment
Basic
Publish Up to 14 KBs, 2 GB size
Production Environment
Standard
Publish Up to 49 KBs, 25 GB size
The throughput for custom question answering is currently capped at 10 text records per
second for both management APIs and prediction APIs. To target 10 text records per second
for your service, we recommend the S1 (one instance) tier of Azure AI Search.
） Important
You can publish N-1 projects of a single language or N/2 projects of different
languages in a particular tier, where N is the maximum indexes allowed in the tier.
Also check the maximum size and the number of documents allowed per tier.
ﾉ
Expand table
Recommended settings
Keys in custom question answering
\nYour custom question answering feature deals with two kinds of keys: authoring keys and
Azure AI Search keys used to access the service in the customer’s subscription.
Use these keys when making requests to the service through APIs.
Name
Location
Purpose
Authoring/Subscription
key
Azure
portal
These keys are used to access the Language service APIs). These
APIs let you edit the questions and answers in your project, and
publish your project. These keys are created when you create a
new resource.
Find these keys on the Azure AI services resource on the Keys
and Endpoint page.
Azure AI Search Admin
Key
Azure
portal
These keys are used to communicate with the Azure AI Search
service deployed in the user’s Azure subscription. When you
associate an Azure AI Search resource with the custom question
answering feature, the admin key is automatically passed to
custom question answering.
You can find these keys on the Azure AI Search resource on the
Keys page.
You can view and reset your authoring keys from the Azure portal, where you added the
custom question answering feature in your language resource.
1. Go to the language resource in the Azure portal and select the resource that has the
Azure AI services type:
ﾉ
Expand table
Find authoring keys in the Azure portal
\n2. Go to Keys and Endpoint:
In custom question answering, both the management and the prediction services are colocated
in the same region.
Each Azure resource created with custom question answering feature has a specific purpose:
Language resource (Also referred to as a Text Analytics resource depending on the
context of where you are evaluating the resource.)
Management service region
Resource purposes
\n![Image](images/page1128_image1.png)

![Image](images/page1128_image2.png)
\nAzure AI Search resource
The language resource with custom question answering feature provides access to the
authoring and publishing APIs, hosts the ranking runtime as well as provides telemetry.
The Azure AI Search resource is used to:
Store the question and answer pairs
Provide the initial ranking (ranker #1) of the question and answer pairs at runtime
You can publish N-1 projects of a single language or N/2 projects of different languages in a
particular tier, where N is the maximum number of indexes allowed in the Azure AI Search tier.
Also check the maximum size and the number of documents allowed per tier.
For example, if your tier has 15 allowed indexes, you can publish 14 projects of the same
language (one index per published project). The 15th index is used for all the projects for
authoring and testing. If you choose to have projects in different languages, then you can only
publish seven projects.
With custom question answering, you have a choice to set up your service for projects in a
single language or multiple languages. You make this choice during the creation of the first
project in your language resource.
Learn about the custom question answering projects
Language resource
Azure AI Search resource
Index usage
Language usage
Next steps
\nPlan your custom question answering app
06/21/2025
To plan your custom question answering app, you need to understand how custom question
answering works and interacts with other Azure services. You should also have a solid grasp of
project concepts.
Each Azure resource created with custom question answering has a specific purpose. Each
resource has its own purpose, limits, and pricing tier. It's important to understand the function
of these resources so that you can use that knowledge into your planning process.
Resource
Purpose
Language resource resource
Authoring, query prediction endpoint and telemetry
Azure AI Search resource
Data storage and search
Custom question answering throughput is currently capped at 10 text records per second for
both management APIs and prediction APIs. To target 10 text records per second for your
service, we recommend the S1 (one instance) SKU of Azure AI Search.
A single language resource with the custom question answering feature enabled can host more
than one project. The number of projects is determined by the Azure AI Search pricing tier's
quantity of supported indexes. Learn more about the relationship of indexes to projects.
When you build a real app, plan sufficient resources for the size of your project and for your
expected query prediction requests.
A project size is controlled by the:
Azure AI Search resource pricing tier limits
Azure resources
ﾉ
Expand table
Resource planning
Language resource
Project size and throughput