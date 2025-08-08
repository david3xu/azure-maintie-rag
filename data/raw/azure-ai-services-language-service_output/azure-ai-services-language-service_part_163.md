You can set additional keyword options to limit the number of answers, specify a minimum
confidence score, and more.
If your knowledge base is configured for chit-chat, the answers from the knowledge base may
include suggested prompts for follow-up questions
 to initiate a conversation. You can ask a
follow-up question by providing the ID of your chosen answer as the context for the continued
conversation:
Python
Python
for candidate in output.answers:
    print("({}) {}".format(candidate.confidence, candidate.answer))
    print("Source: {}".format(candidate.source))
Ask a follow-up question
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.ai.language.questionanswering import models
endpoint = os.environ["AZURE_QUESTIONANSWERING_ENDPOINT"]
key = os.environ["AZURE_QUESTIONANSWERING_KEY"]
client = QuestionAnsweringClient(endpoint, AzureKeyCredential(key))
output = client.get_answers(
    question="How long should charging take?",
    answer_context=models.KnowledgeBaseAnswerContext(
        previous_qna_id=previous_answer.qna_id
    ),
    project_name="FAQ",
    deployment_name="live"
)
for candidate in output.answers:
    print("({}) {}".format(candidate.confidence, candidate.answer))
    print("Source: {}".format(candidate.source))
Create a new project
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering.authoring import AuthoringClient
# get service secrets
\nPython
endpoint = os.environ["AZURE_QUESTIONANSWERING_ENDPOINT"]
key = os.environ["AZURE_QUESTIONANSWERING_KEY"]
# create client
client = AuthoringClient(endpoint, AzureKeyCredential(key))
with client:
    # create project
    project_name = "IssacNewton"
    project = client.create_project(
        project_name=project_name,
        options={
            "description": "biography of Sir Issac Newton",
            "language": "en",
            "multilingualResource": True,
            "settings": {
                "defaultAnswer": "no answer"
            }
        })
    print("view created project info:")
    print("\tname: {}".format(project["projectName"]))
    print("\tlanguage: {}".format(project["language"]))
    print("\tdescription: {}".format(project["description"]))
Add a knowledge source
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering.authoring import AuthoringClient
# get service secrets
endpoint = os.environ["AZURE_QUESTIONANSWERING_ENDPOINT"]
key = os.environ["AZURE_QUESTIONANSWERING_KEY"]
# create client
client = AuthoringClient(endpoint, AzureKeyCredential(key))
project_name = "IssacNewton"
update_sources_poller = client.begin_update_sources(
    project_name=project_name,
    sources=[
        {
            "op": "add",
            "value": {
                "displayName": "Issac Newton Bio",
                "sourceUri": "https://wikipedia.org/wiki/Isaac_Newton",
                "sourceKind": "url"
            }
        }
\nPython
    ]
)
update_sources_poller.result()
# list sources
print("list project sources")
sources = client.list_sources(
    project_name=project_name
)
for source in sources:
    print("project: {}".format(source["displayName"]))
    print("\tsource: {}".format(source["source"]))
    print("\tsource Uri: {}".format(source["sourceUri"]))
    print("\tsource kind: {}".format(source["sourceKind"]))
Deploy your project
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering.authoring import AuthoringClient
# get service secrets
endpoint = os.environ["AZURE_QUESTIONANSWERING_ENDPOINT"]
key = os.environ["AZURE_QUESTIONANSWERING_KEY"]
# create client
client = AuthoringClient(endpoint, AzureKeyCredential(key))
project_name = "IssacNewton"
# deploy project
deployment_poller = client.begin_deploy_project(
    project_name=project_name,
    deployment_name="production"
)
deployment_poller.result()
# list all deployments
deployments = client.list_deployments(
    project_name=project_name
)
print("view project deployments")
for d in deployments:
    print(d)
Asynchronous operations
\nThe above examples can also be run asynchronously using the clients in the aio  namespace:
Python
Optional keyword arguments can be passed in at the client and per-operation level. The azure-
core reference documentation
 describes available configurations for retries, logging,
transport protocols, and more.
Azure Question Answering clients raise exceptions defined in Azure Core
. When you interact
with the Cognitive Language Service Question Answering client library using the Python SDK,
errors returned by the service correspond to the same HTTP status codes returned for REST API
requests.
For example, if you submit a question to a non-existent knowledge base, a 400  error is
returned indicating "Bad Request".
Python
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering.aio import QuestionAnsweringClient
endpoint = os.environ["AZURE_QUESTIONANSWERING_ENDPOINT"]
key = os.environ["AZURE_QUESTIONANSWERING_KEY"]
client = QuestionAnsweringClient(endpoint, AzureKeyCredential(key))
output = await client.get_answers(
    question="How long should my Surface battery last?",
    project_name="FAQ",
    deployment_name="production"
)
Optional Configuration
Troubleshooting
General
from azure.core.exceptions import HttpResponseError
try:
    client.get_answers(
        question="Why?",
        project_name="invalid-knowledge-base",
\nThis library uses the standard logging
 library for logging. Basic information about HTTP
sessions (URLs, headers, etc.) is logged at INFO level.
Detailed DEBUG level logging, including request/response bodies and unredacted headers, can
be enabled on a client with the logging_enable  argument.
See full SDK logging documentation with examples here.
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
When you submit a pull request, a CLA-bot will automatically determine whether you need to
provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repos using our
CLA.
This project has adopted the Microsoft Open Source Code of Conduct
. For more information
see the Code of Conduct FAQ
 or contact opencode@microsoft.com with any additional
questions or comments.
        deployment_name="test"
    )
except HttpResponseError as error:
    print("Query failed: {}".format(error.message))
Logging
Next steps
Contributing
\nArticle • 06/14/2023
Azure Pipelines
Azure Pipelines
set up now
set up now
Azure Conversational Language
Understanding client library for Python -
version 1.1.0
Conversational Language Understanding - aka CLU for short - is a cloud-based conversational
AI service which provides many language understanding capabilities like:
Conversation App: It's used in extracting intents and entities in conversations
Workflow app: Acts like an orchestrator to select the best candidate to analyze
conversations to get best response from apps like Qna, Luis, and Conversation App
Conversational Summarization: Used to analyze conversations in the form of
issues/resolution, chapter title, and narrative summarizations
Source code
 | Package (PyPI)
 | Package (Conda)
 | API reference documentation
 |
Samples
 | Product documentation | REST API documentation
Python 3.7 or later is required to use this package.
An Azure subscription
A Language service resource
Install the Azure Conversations client library for Python with pip
:
Bash
Note: This version of the client library defaults to the 2023-04-01 version of the service
Getting started
Prerequisites
Install the package
pip install azure-ai-language-conversations
Authenticate the client
\nIn order to interact with the CLU service, you'll need to create an instance of the
ConversationAnalysisClient
 class, or ConversationAuthoringClient
 class. You will need an
endpoint, and an API key to instantiate a client object. For more information regarding
authenticating with Cognitive Services, see Authenticate requests to Azure Cognitive Services.
You can get the endpoint and an API key from the Cognitive Services resource in the Azure
Portal
.
Alternatively, use the Azure CLI command shown below to get the API key from the Cognitive
Service resource.
PowerShell
Once you've determined your endpoint and API key you can instantiate a
ConversationAnalysisClient :
Python
Once you've determined your endpoint and API key you can instantiate a
ConversationAuthoringClient :
Python
Get an API key
az cognitiveservices account keys list --resource-group <resource-group-name> --
name <resource-name>
Create ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
endpoint = "https://<my-custom-subdomain>.cognitiveservices.azure.com/"
credential = AzureKeyCredential("<api-key>")
client = ConversationAnalysisClient(endpoint, credential)
Create ConversationAuthoringClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient
endpoint = "https://<my-custom-subdomain>.cognitiveservices.azure.com/"
\nTo use an Azure Active Directory (AAD) token credential, provide an instance of the desired
credential type obtained from the azure-identity
 library. Note that regional endpoints do not
support AAD authentication. Create a custom subdomain name for your resource in order to
use this type of authentication.
Authentication with AAD requires some initial setup:
Install azure-identity
Register a new AAD application
Grant access to the Language service by assigning the "Cognitive Services Language
Reader" role to your service principal.
After setup, you can choose which type of credential
 from azure.identity to use. As an
example, DefaultAzureCredential
 can be used to authenticate the client:
Set the values of the client ID, tenant ID, and client secret of the AAD application as
environment variables: AZURE_CLIENT_ID , AZURE_TENANT_ID , AZURE_CLIENT_SECRET
Use the returned token credential to authenticate the client:
Python
The ConversationAnalysisClient
 is the primary interface for making predictions using your
deployed Conversations models. For asynchronous operations, an async
ConversationAnalysisClient  is in the azure.ai.language.conversation.aio  namespace.
credential = AzureKeyCredential("<api-key>")
client = ConversationAuthoringClient(endpoint, credential)
Create a client with an Azure Active Directory Credential
from azure.ai.language.conversations import ConversationAnalysisClient
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
client = ConversationAnalysisClient(endpoint="https://<my-custom-
subdomain>.cognitiveservices.azure.com/", credential=credential)
Key concepts
ConversationAnalysisClient
\nYou can use the ConversationAuthoringClient
 to interface with the Azure Language Portal
to carry out authoring operations on your language resource/project. For example, you can use
it to create a project, populate with training data, train, test, and deploy. For asynchronous
operations, an async ConversationAuthoringClient  is in the
azure.ai.language.conversation.authoring.aio  namespace.
The azure-ai-language-conversation  client library provides both synchronous and
asynchronous APIs.
The following examples show common scenarios using the client  created above.
If you would like to extract custom intents and entities from a user utterance, you can call the
client.analyze_conversation()  method with your conversation's project name as follows:
Python
ConversationAuthoringClient
Examples
Analyze Text with a Conversation App
# import libraries
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
# get secrets
clu_endpoint = os.environ["AZURE_CONVERSATIONS_ENDPOINT"]
clu_key = os.environ["AZURE_CONVERSATIONS_KEY"]
project_name = os.environ["AZURE_CONVERSATIONS_PROJECT_NAME"]
deployment_name = os.environ["AZURE_CONVERSATIONS_DEPLOYMENT_NAME"]
# analyze quey
client = ConversationAnalysisClient(clu_endpoint, AzureKeyCredential(clu_key))
with client:
    query = "Send an email to Carol about the tomorrow's demo"
    result = client.analyze_conversation(
        task={
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "participantId": "1",
                    "id": "1",
                    "modality": "text",
                    "language": "en",
                    "text": query
\nIf you would like to pass the user utterance to your orchestrator (worflow) app, you can call the
client.analyze_conversation()  method with your orchestration's project name. The
orchestrator project simply orchestrates the submitted user utterance between your language
apps (Luis, Conversation, and Question Answering) to get the best response according to the
user intent. See the next example:
Python
                },
                "isLoggingEnabled": False
            },
            "parameters": {
                "projectName": project_name,
                "deploymentName": deployment_name,
                "verbose": True
            }
        }
    )
# view result
print("query: {}".format(result["result"]["query"]))
print("project kind: {}\n".format(result["result"]["prediction"]["projectKind"]))
print("top intent: {}".format(result["result"]["prediction"]["topIntent"]))
print("category: {}".format(result["result"]["prediction"]["intents"][0]
["category"]))
print("confidence score: {}\n".format(result["result"]["prediction"]["intents"][0]
["confidenceScore"]))
print("entities:")
for entity in result["result"]["prediction"]["entities"]:
    print("\ncategory: {}".format(entity["category"]))
    print("text: {}".format(entity["text"]))
    print("confidence score: {}".format(entity["confidenceScore"]))
    if "resolutions" in entity:
        print("resolutions")
        for resolution in entity["resolutions"]:
            print("kind: {}".format(resolution["resolutionKind"]))
            print("value: {}".format(resolution["value"]))
    if "extraInformation" in entity:
        print("extra info")
        for data in entity["extraInformation"]:
            print("kind: {}".format(data["extraInformationKind"]))
            if data["extraInformationKind"] == "ListKey":
                print("key: {}".format(data["key"]))
            if data["extraInformationKind"] == "EntitySubtype":
                print("value: {}".format(data["value"]))
Analyze Text with an Orchestration App