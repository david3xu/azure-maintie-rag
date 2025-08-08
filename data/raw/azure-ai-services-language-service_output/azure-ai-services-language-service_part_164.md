# import libraries
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
# get secrets
clu_endpoint = os.environ["AZURE_CONVERSATIONS_ENDPOINT"]
clu_key = os.environ["AZURE_CONVERSATIONS_KEY"]
project_name = os.environ["AZURE_CONVERSATIONS_WORKFLOW_PROJECT_NAME"]
deployment_name = os.environ["AZURE_CONVERSATIONS_WORKFLOW_DEPLOYMENT_NAME"]
# analyze query
client = ConversationAnalysisClient(clu_endpoint, AzureKeyCredential(clu_key))
with client:
    query = "Reserve a table for 2 at the Italian restaurant"
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
# top intent
top_intent = result["result"]["prediction"]["topIntent"]
print("top intent: {}".format(top_intent))
top_intent_object = result["result"]["prediction"]["intents"][top_intent]
print("confidence score: {}".format(top_intent_object["confidenceScore"]))
print("project kind: {}".format(top_intent_object["targetProjectKind"]))
if top_intent_object["targetProjectKind"] == "Luis":
    print("\nluis response:")
    luis_response = top_intent_object["result"]["prediction"]
    print("top intent: {}".format(luis_response["topIntent"]))
    print("\nentities:")
\nYou can use this sample if you need to summarize a conversation in the form of an issue, and
final resolution. For example, a dialog from tech support:
Python
    for entity in luis_response["entities"]:
        print("\n{}".format(entity))
Conversational Summarization
# import libraries
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
# get secrets
endpoint = os.environ["AZURE_CONVERSATIONS_ENDPOINT"]
key = os.environ["AZURE_CONVERSATIONS_KEY"]
# analyze query
client = ConversationAnalysisClient(endpoint, AzureKeyCredential(key))
with client:
    poller = client.begin_conversation_analysis(
        task={
            "displayName": "Analyze conversations from xxx",
            "analysisInput": {
                "conversations": [
                    {
                        "conversationItems": [
                            {
                                "text": "Hello, how can I help you?",
                                "modality": "text",
                                "id": "1",
                                "participantId": "Agent"
                            },
                            {
                                "text": "How to upgrade Office? I am getting error 
messages the whole day.",
                                "modality": "text",
                                "id": "2",
                                "participantId": "Customer"
                            },
                            {
                                "text": "Press the upgrade button please. Then 
sign in and follow the instructions.",
                                "modality": "text",
                                "id": "3",
                                "participantId": "Agent"
                            }
                        ],
                        "modality": "text",
                        "id": "conversation1",
                        "language": "en"
                    },
\nThis sample shows a common scenario for the authoring part of the SDK
Python
                ]
            },
            "tasks": [
                {
                    "taskName": "Issue task",
                    "kind": "ConversationalSummarizationTask",
                    "parameters": {
                        "summaryAspects": ["issue"]
                    }
                },
                {
                    "taskName": "Resolution task",
                    "kind": "ConversationalSummarizationTask",
                    "parameters": {
                        "summaryAspects": ["resolution"]
                    }
                },
            ]
        }
    )
    # view result
    result = poller.result()
    task_results = result["tasks"]["items"]
    for task in task_results:
        print(f"\n{task['taskName']} status: {task['status']}")
        task_result = task["results"]
        if task_result["errors"]:
            print("... errors occurred ...")
            for error in task_result["errors"]:
                print(error)
        else:
            conversation_result = task_result["conversations"][0]
            if conversation_result["warnings"]:
                print("... view warnings ...")
                for warning in conversation_result["warnings"]:
                    print(warning)
            else:
                summaries = conversation_result["summaries"]
                print("... view task result ...")
                for summary in summaries:
                    print(f"{summary['aspect']}: {summary['text']}")
Import a Conversation Project
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient
\nclu_endpoint = os.environ["AZURE_CONVERSATIONS_ENDPOINT"]
clu_key = os.environ["AZURE_CONVERSATIONS_KEY"]
project_name = "test_project"
exported_project_assets = {
    "projectKind": "Conversation",
    "intents": [{"category": "Read"}, {"category": "Delete"}],
    "entities": [{"category": "Sender"}],
    "utterances": [
        {
            "text": "Open Blake's email",
            "dataset": "Train",
            "intent": "Read",
            "entities": [{"category": "Sender", "offset": 5, "length": 5}],
        },
        {
            "text": "Delete last email",
            "language": "en-gb",
            "dataset": "Test",
            "intent": "Delete",
            "entities": [],
        },
    ],
}
client = ConversationAuthoringClient(
    clu_endpoint, AzureKeyCredential(clu_key)
)
poller = client.begin_import_project(
    project_name=project_name,
    project={
        "assets": exported_project_assets,
        "metadata": {
            "projectKind": "Conversation",
            "settings": {"confidenceThreshold": 0.7},
            "projectName": "EmailApp",
            "multilingual": True,
            "description": "Trying out CLU",
            "language": "en-us",
        },
        "projectFileVersion": "2022-05-01",
    },
)
response = poller.result()
print(response)
Optional Configuration
\nOptional keyword arguments can be passed in at the client and per-operation level. The azure-
core reference documentation
 describes available configurations for retries, logging,
transport protocols, and more.
The Conversations client will raise exceptions defined in Azure Core
.
This library uses the standard logging
 library for logging. Basic information about HTTP
sessions (URLs, headers, etc.) is logged at INFO level.
Detailed DEBUG level logging, including request/response bodies and unredacted headers, can
be enabled on a client with the logging_enable  argument.
See full SDK logging documentation with examples here.
Python
Similarly, logging_enable  can enable detailed logging for a single operation, even when it isn't
enabled for the client:
Troubleshooting
General
Logging
import sys
import logging
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
# Create a logger for the 'azure' SDK
logger = logging.getLogger('azure')
logger.setLevel(logging.DEBUG)
# Configure a console output
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
endpoint = "https://<my-custom-subdomain>.cognitiveservices.azure.com/"
credential = AzureKeyCredential("<my-api-key>")
# This client will log detailed information about its HTTP sessions, at DEBUG 
level
client = ConversationAnalysisClient(endpoint, credential, logging_enable=True)
result = client.analyze_conversation(...)
\nPython
See the Sample README
 for several code snippets illustrating common patterns used in the
CLU Python API.
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
result = client.analyze_conversation(..., logging_enable=True)
Next steps
More sample code
Contributing
\nAzure Text Analytics client library for Java -
version 5.5.7
06/21/2025
The Azure Cognitive Service for Language is a cloud-based service that provides Natural
Language Processing (NLP) features for understanding and analyzing text, and includes the
following main features:
Sentiment Analysis
Entity Recognition (Named, Linked, and Personally Identifiable Information (PII) entities)
Language Detection
Key Phrase Extraction
Multiple Actions Analysis Per Document
Healthcare Entities Analysis
Abstractive Text Summarization
Extractive Text Summarization
Custom Named Entity Recognition
Custom Text Classification
Source code
 | Package (Maven)
 | API reference documentation
 | Product Documentation
| Samples
A Java Development Kit (JDK), version 8 or later.
Here are details about Java 8 client compatibility with Azure Certificate Authority.
Azure Subscription
Cognitive Services or Language service account to use this package.
Please include the azure-sdk-bom to your project to take dependency on GA version of the
library. In the following snippet, replace the {bom_version_to_target} placeholder with the
version number. To learn more about the BOM, see the AZURE SDK BOM README
.
Getting started
Prerequisites
Include the Package
Include the BOM file
\nXML
and then include the direct dependency in the dependencies section without the version tag.
XML
If you want to take dependency on a particular version of the library that is not present in the
BOM, add the direct dependency to your project as follows.
XML
Note: This version of the client library defaults to the 2023-04-01  version of the service. It is a
newer version than 3_0 , 3_1  and 2022-05-01 .
This table shows the relationship between SDK services and supported API versions of the
service:
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>com.azure</groupId>
            <artifactId>azure-sdk-bom</artifactId>
            <version>{bom_version_to_target}</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
<dependencies>
  <dependency>
    <groupId>com.azure</groupId>
    <artifactId>azure-ai-textanalytics</artifactId>
  </dependency>
</dependencies>
Include direct dependency
<dependency>
    <groupId>com.azure</groupId>
    <artifactId>azure-ai-textanalytics</artifactId>
    <version>5.5.7</version>
</dependency>
ﾉ
Expand table
\nSDK version
Supported API version of service
5.3.x
3.0, 3.1, 2022-05-01, 2023-04-01 (default)
5.2.x
3.0, 3.1, 2022-05-01
5.1.x
3.0, 3.1
5.0.x
3.0
The Language service supports both multi-service and single-service access. Create a Cognitive
Services resource if you plan to access multiple cognitive services under a single endpoint/key.
For Language service access only, create a Language service resource.
You can create the resource using the Azure Portal or Azure CLI following the steps in this
document.
In order to interact with the Language service, you will need to create an instance of the Text
Analytics client, both the asynchronous and synchronous clients can be created by using
TextAnalyticsClientBuilder  invoking buildClient()  creates a synchronous client while
buildAsyncClient()  creates its asynchronous counterpart.
You will need an endpoint and either a key or AAD TokenCredential to instantiate a client
object.
You can find the endpoint for your Language service resource in the Azure Portal
 under the
"Keys and Endpoint", or Azure CLI.
Bash
Create a Cognitive Services or Language Service resource
Authenticate the client
Looking up the endpoint
# Get the endpoint for the Language service resource
az cognitiveservices account show --name "resource-name" --resource-group 
"resource-group-name" --query "endpoint"
Create a Text Analytics client with key credential
\nOnce you have the value for the key, provide it as a string to the AzureKeyCredential
. This
can be found in the Azure Portal
 under the "Keys and Endpoint" section in your created
Language service resource or by running the following Azure CLI command:
Bash
Use the key as the credential parameter to authenticate the client:
Java
The Azure Text Analytics client library provides a way to rotate the existing key.
Java
Azure SDK for Java supports an Azure Identity package, making it easy to get credentials from
Microsoft identity platform.
Authentication with AAD requires some initial setup:
Add the Azure Identity package
XML
az cognitiveservices account keys list --resource-group <your-resource-group-name> 
--name <your-resource-name>
TextAnalyticsClient textAnalyticsClient = new TextAnalyticsClientBuilder()
    .credential(new AzureKeyCredential("{key}"))
    .endpoint("{endpoint}")
    .buildClient();
AzureKeyCredential credential = new AzureKeyCredential("{key}");
TextAnalyticsClient textAnalyticsClient = new TextAnalyticsClientBuilder()
    .credential(credential)
    .endpoint("{endpoint}")
    .buildClient();
credential.update("{new_key}");
Create a Text Analytics client with Azure Active Directory credential
<dependency>
    <groupId>com.azure</groupId>
    <artifactId>azure-identity</artifactId>
    <version>1.13.1</version>
</dependency>