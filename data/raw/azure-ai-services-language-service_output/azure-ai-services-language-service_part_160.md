Note that regional endpoints do not support AAD authentication. Instead, create a custom
domain name for your resource to use AAD authentication.
The QuestionAnsweringClient
 is the primary interface for asking questions using a
knowledge base with your own information, or text input using pre-trained models. It provides
both synchronous and asynchronous APIs to ask questions.
The QuestionAnsweringAuthoringClient
 provides an interface for managing Question
Answering projects. Examples of the available operations include creating and deploying
projects, updating your knowledge sources, and updating question and answer pairs. It
provides both synchronous and asynchronous APIs.
We guarantee that all client instance methods are thread-safe and independent of each other
(guideline
). This ensures that the recommendation of reusing client instances is always safe,
even across threads.
Client options
 | Accessing the response
 | Long-running operations
 | Handling failures
 |
Diagnostics
 | Mocking
 | Client lifetime
The Azure.AI.Language.QuestionAnswering client library provides both synchronous and
asynchronous APIs.
QuestionAnsweringClient client = new QuestionAnsweringClient(endpoint, 
credential);
Key concepts
QuestionAnsweringClient
QuestionAnsweringAuthoringClient
Thread safety
Additional concepts
Examples
QuestionAnsweringClient
\nThe following examples show common scenarios using the client  created above.
The only input required to a ask a question using an existing knowledge base is just the
question itself:
C#
You can set additional properties on QuestionAnsweringClientOptions  to limit the number of
answers, specify a minimum confidence score, and more.
If your knowledge base is configured for chit-chat, you can ask a follow-up question provided
the previous question-answering ID and, optionally, the exact question the user asked:
C#
Ask a question
string projectName = "{ProjectName}";
string deploymentName = "{DeploymentName}";
QuestionAnsweringProject project = new QuestionAnsweringProject(projectName, 
deploymentName);
Response<AnswersResult> response = client.GetAnswers("How long should my Surface 
battery last?", project);
foreach (KnowledgeBaseAnswer answer in response.Value.Answers)
{
    Console.WriteLine($"({answer.Confidence:P2}) {answer.Answer}");
    Console.WriteLine($"Source: {answer.Source}");
    Console.WriteLine();
}
Ask a follow-up question
string projectName = "{ProjectName}";
string deploymentName = "{DeploymentName}";
// Answers are ordered by their ConfidenceScore so assume the user choose the 
first answer below:
KnowledgeBaseAnswer previousAnswer = answers.Answers.First();
QuestionAnsweringProject project = new QuestionAnsweringProject(projectName, 
deploymentName);
AnswersOptions options = new AnswersOptions
{
    AnswerContext = new KnowledgeBaseAnswerContext(previousAnswer.QnaId.Value)
};
Response<AnswersResult> response = client.GetAnswers("How long should charging 
take?", project, options);
\nThe following examples show common scenarios using the QuestionAnsweringAuthoringClient
instance created in this section.
To create a new project, you must specify the project's name and a create a RequestContent
instance with the parameters needed to set up the project.
C#
foreach (KnowledgeBaseAnswer answer in response.Value.Answers)
{
    Console.WriteLine($"({answer.Confidence:P2}) {answer.Answer}");
    Console.WriteLine($"Source: {answer.Source}");
    Console.WriteLine();
}
QuestionAnsweringAuthoringClient
Create a new project
// Set project name and request content parameters
string newProjectName = "{ProjectName}";
RequestContent creationRequestContent = RequestContent.Create(
    new {
        description = "This is the description for a test project",
        language = "en",
        multilingualResource = false,
        settings = new {
            defaultAnswer = "No answer found for your question."
            }
        }
    );
Response creationResponse = client.CreateProject(newProjectName, 
creationRequestContent);
// Projects can be retrieved as follows
Pageable<BinaryData> projects = client.GetProjects();
Console.WriteLine("Projects: ");
foreach (BinaryData project in projects)
{
    Console.WriteLine(project);
}
Deploy your project
\nYour projects can be deployed using the DeployProjectAsync  or the synchronous
DeployProject . All you need to specify is the project's name and the deployment name that
you wish to use. Please note that the service will not allow you to deploy empty projects.
C#
One way to add content to your project is to add a knowledge source. The following example
shows how you can set up a RequestContent  instance to add a new knowledge source of the
type "url".
C#
// Set deployment name and start operation
string newDeploymentName = "{DeploymentName}";
Operation<BinaryData> deploymentOperation = 
client.DeployProject(WaitUntil.Completed, newProjectName, newDeploymentName);
// Deployments can be retrieved as follows
Pageable<BinaryData> deployments = client.GetDeployments(newProjectName);
Console.WriteLine("Deployments: ");
foreach (BinaryData deployment in deployments)
{
    Console.WriteLine(deployment);
}
Add a knowledge source
// Set request content parameters for updating our new project's sources
string sourceUri = "{KnowledgeSourceUri}";
RequestContent updateSourcesRequestContent = RequestContent.Create(
    new[] {
        new {
                op = "add",
                value = new
                {
                    displayName = "MicrosoftFAQ",
                    source = sourceUri,
                    sourceUri = sourceUri,
                    sourceKind = "url",
                    contentStructureKind = "unstructured",
                    refresh = false
                }
            }
    });
Operation<Pageable<BinaryData>> updateSourcesOperation = 
client.UpdateSources(WaitUntil.Completed, newProjectName, 
updateSourcesRequestContent);
\nWhen you interact with the Cognitive Language Services Question Answering client library
using the .NET SDK, errors returned by the service correspond to the same HTTP status codes
returned for REST API requests.
For example, if you submit a question to a non-existant knowledge base, a 400  error is
returned indicating "Bad Request".
C#
You will notice that additional information is logged, like the client request ID of the operation.
text
// Knowledge Sources can be retrieved as follows
Pageable<BinaryData> sources = updateSourcesOperation.Value;
Console.WriteLine("Sources: ");
foreach (BinaryData source in sources)
{
    Console.WriteLine(source);
}
Troubleshooting
General
try
{
    QuestionAnsweringProject project = new QuestionAnsweringProject("invalid-
knowledgebase", "test");
    Response<AnswersResult> response = client.GetAnswers("Does this knowledge base 
exist?", project);
}
catch (RequestFailedException ex)
{
    Console.WriteLine(ex.ToString());
}
Azure.RequestFailedException: Please verify azure search service is up, restart 
the WebApp and try again
Status: 400 (Bad Request)
ErrorCode: BadArgument
Content:
{
    "error": {
\nThe simplest way to see the logs is to enable console logging. To create an Azure SDK log
listener that outputs messages to the console use the
AzureEventSourceListener.CreateConsoleLogger  method.
C#
To learn more about other logging mechanisms see here
.
View our samples
.
Read about the different features
 of the Question Answering service.
Try our service demos
.
See the CONTRIBUTING.md
 for details on building, testing, and contributing to this library.
This project welcomes contributions and suggestions. Most contributions require you to agree
to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do,
grant us the rights to use your contribution. For details, visit cla.microsoft.com
.
    "code": "BadArgument",
    "message": "Please verify azure search service is up, restart the WebApp and 
try again"
    }
}
Headers:
x-envoy-upstream-service-time: 23
apim-request-id: 76a83876-22d1-4977-a0b1-559a674f3605
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Content-Type-Options: nosniff
Date: Wed, 30 Jun 2021 00:32:07 GMT
Content-Length: 139
Content-Type: application/json; charset=utf-8
Setting up console logging
// Setup a listener to monitor logged events.
using AzureEventSourceListener listener = 
AzureEventSourceListener.CreateConsoleLogger();
Next steps
Contributing
\nWhen you submit a pull request, a CLA-bot will automatically determine whether you need to
provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repos using our
CLA.
This project has adopted the Microsoft Open Source Code of Conduct
. For more information
see the Code of Conduct FAQ
 or contact opencode@microsoft.com with any additional
questions or comments.
\nAzure Text Analytics client library for
Python - version 5.3.0
Article • 06/21/2023
The Azure Cognitive Service for Language is a cloud-based service that provides Natural
Language Processing (NLP) features for understanding and analyzing text, and includes the
following main features:
Sentiment Analysis
Named Entity Recognition
Language Detection
Key Phrase Extraction
Entity Linking
Multiple Analysis
Personally Identifiable Information (PII) Detection
Text Analytics for Health
Custom Named Entity Recognition
Custom Text Classification
Extractive Text Summarization
Abstractive Text Summarization
Source code
 | Package (PyPI)
 | Package (Conda)
 | API reference documentation
 |
Product documentation | Samples
Python 3.7 later is required to use this package.
You must have an Azure subscription
 and a Cognitive Services or Language service
resource to use this package.
The Language service supports both multi-service and single-service access. Create a Cognitive
Services resource if you plan to access multiple cognitive services under a single endpoint/key.
For Language service access only, create a Language service resource. You can create the
resource using the Azure Portal
 or Azure CLI following the steps in this document.
Getting started
Prerequisites
Create a Cognitive Services or Language service resource
\nInteraction with the service using the client library begins with a client. To create a client object,
you will need the Cognitive Services or Language service endpoint  to your resource and a
credential  that allows you access:
Python
Note that for some Cognitive Services resources the endpoint might look different from the
above code snippet. For example, https://<region>.api.cognitive.microsoft.com/ .
Install the Azure Text Analytics client library for Python with pip
:
Bash
Python
Note that 5.2.X  and newer targets the Azure Cognitive Service for Language APIs. These
APIs include the text analysis and natural language processing features found in the
previous versions of the Text Analytics client library. In addition, the service API has
changed from semantic to date-based versioning. This version of the client library defaults
to the latest supported API version, which currently is 2023-04-01 .
This table shows the relationship between SDK versions and supported API versions of the
service
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
credential = AzureKeyCredential("<api_key>")
text_analytics_client = TextAnalyticsClient(endpoint="https://<resource-
name>.cognitiveservices.azure.com/", credential=credential)
Install the package
pip install azure-ai-textanalytics
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(endpoint, AzureKeyCredential(key))
\nSDK version
Supported API version of service
5.3.X - Latest stable release
3.0, 3.1, 2022-05-01, 2023-04-01 (default)
5.2.X
3.0, 3.1, 2022-05-01 (default)
5.1.0
3.0, 3.1 (default)
5.0.0
3.0
API version can be selected by passing the api_version
 keyword argument into the client. For
the latest Language service features, consider selecting the most recent beta API version. For
production scenarios, the latest stable version is recommended. Setting to an older version
may result in reduced feature compatibility.
You can find the endpoint for your Language service resource using the Azure Portal or Azure
CLI:
Bash
You can get the API key from the Cognitive Services or Language service resource in the Azure
Portal. Alternatively, you can use Azure CLI snippet below to get the API key of your resource.
az cognitiveservices account keys list --name "resource-name" --resource-group "resource-
group-name"
Once you have the value for the API key, you can pass it as a string into an instance of
AzureKeyCredential
. Use the key as the credential parameter to authenticate the client:
ﾉ
Expand table
Authenticate the client
Get the endpoint
# Get the endpoint for the Language service resource
az cognitiveservices account show --name "resource-name" --resource-group 
"resource-group-name" --query "properties.endpoint"
Get the API Key
Create a TextAnalyticsClient with an API Key Credential