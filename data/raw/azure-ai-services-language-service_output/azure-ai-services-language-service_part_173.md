If you would like to pass the user utterance to your orchestrator (worflow) app, you can
call the client.analyze_conversation() method with your orchestration's project name.
The orchestrator project simply orchestrates the submitted user utterance between your
language apps (Luis, Conversation, and Question Answering) to get the best response
according to the user intent. See the next example:
Python
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
client = ConversationAnalysisClient(clu_endpoint, 
AzureKeyCredential(clu_key))
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
print("project kind: {}\n".format(result["result"]["prediction"]
["projectKind"]))
# top intent
top_intent = result["result"]["prediction"]["topIntent"]
\nYou can use this sample if you need to summarize a conversation in the form of an issue,
and final resolution. For example, a dialog from tech support:
Python
print("top intent: {}".format(top_intent))
top_intent_object = result["result"]["prediction"]["intents"][top_intent]
print("confidence score: {}".format(top_intent_object["confidenceScore"]))
print("project kind: {}".format(top_intent_object["targetProjectKind"]))
if top_intent_object["targetProjectKind"] == "Luis":
    print("\nluis response:")
    luis_response = top_intent_object["result"]["prediction"]
    print("top intent: {}".format(luis_response["topIntent"]))
    print("\nentities:")
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
                                "text": "How to upgrade Office? I am getting 
error messages the whole day.",
                                "modality": "text",
                                "id": "2",
                                "participantId": "Customer"
                            },
                            {
                                "text": "Press the upgrade button please. 
\nThen sign in and follow the instructions.",
                                "modality": "text",
                                "id": "3",
                                "participantId": "Agent"
                            }
                        ],
                        "modality": "text",
                        "id": "conversation1",
                        "language": "en"
                    },
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
\nYou can use this sample if you need to extract and redact pii info from/in conversations
Python
Conversational PII
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
            "displayName": "Analyze PII in conversation",
            "analysisInput": {
                "conversations": [
                    {
                        "conversationItems": [
                            {
                                "id": "1",
                                "participantId": "0",
                                "modality": "transcript",
                                "text": "It is john doe.",
                                "lexical": "It is john doe",
                                "itn": "It is john doe",
                                "maskedItn": "It is john doe"
                            },
                            {
                                "id": "2",
                                "participantId": "1",
                                "modality": "transcript",
                                "text": "Yes, 633-27-8199 is my phone",
                                "lexical": "yes six three three two seven 
eight one nine nine is my phone",
                                "itn": "yes 633278199 is my phone",
                                "maskedItn": "yes 633278199 is my phone",
                            },
                            {
                                "id": "3",
                                "participantId": "1",
                                "modality": "transcript",
                                "text": "j.doe@yahoo.com is my email",
                                "lexical": "j dot doe at yahoo dot com is my 
email",
                                "maskedItn": "j.doe@yahoo.com is my email",
                                "itn": "j.doe@yahoo.com is my email",
                            }
                        ],
\nAnalyze sentiment in conversations.
                        "modality": "transcript",
                        "id": "1",
                        "language": "en"
                    }
                ]
            },
            "tasks": [
                {
                    "kind": "ConversationalPIITask",
                    "parameters": {
                        "redactionSource": "lexical",
                        "piiCategories": [
                            "all"
                        ]
                    }
                }
            ]
        }
    )
    # view result
    result = poller.result()
    task_result = result["tasks"]["items"][0]
    print("... view task status ...")
    print("status: {}".format(task_result["status"]))
    conv_pii_result = task_result["results"]
    if conv_pii_result["errors"]:
        print("... errors occurred ...")
        for error in conv_pii_result["errors"]:
            print(error)
    else:
        conversation_result = conv_pii_result["conversations"][0]
        if conversation_result["warnings"]:
            print("... view warnings ...")
            for warning in conversation_result["warnings"]:
                print(warning)
        else:
            print("... view task result ...")
            for conversation in conversation_result["conversationItems"]:
                print("conversation id: {}".format(conversation["id"]))
                print("... entities ...")
                for entity in conversation["entities"]:
                    print("text: {}".format(entity["text"]))
                    print("category: {}".format(entity["category"]))
                    print("confidence: 
{}".format(entity["confidenceScore"]))
                    print("offset: {}".format(entity["offset"]))
                    print("length: {}".format(entity["length"]))
Conversational Sentiment Analysis
\nPython
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
          "displayName": "Sentiment Analysis from a call center 
conversation",
          "analysisInput": {
            "conversations": [
              {
                "id": "1",
                "language": "en",
                "modality": "transcript",
                "conversationItems": [
                  {
                    "participantId": "1",
                    "id": "1",
                    "text": "I like the service. I do not like the food",
                    "lexical": "i like the service i do not like the food",
                  }
                ]
              }
            ]
          },
          "tasks": [
            {
              "taskName": "Conversation Sentiment Analysis",
              "kind": "ConversationalSentimentTask",
              "parameters": {
                "modelVersion": "latest",
                "predictionSource": "text"
              }
            }
          ]
        }
    )
    result = poller.result()
    task_result = result["tasks"]["items"][0]
    print("... view task status ...")
    print(f"status: {task_result['status']}")
    conv_sentiment_result = task_result["results"]
    if conv_sentiment_result["errors"]:
        print("... errors occurred ...")
\nThis sample shows a common scenario for the authoring part of the SDK
Python
        for error in conv_sentiment_result["errors"]:
            print(error)
    else:
        conversation_result = conv_sentiment_result["conversations"][0]
        if conversation_result["warnings"]:
            print("... view warnings ...")
            for warning in conversation_result["warnings"]:
                print(warning)
        else:
            print("... view task result ...")
            for conversation in conversation_result["conversationItems"]:
                print(f"Participant ID: {conversation['participantId']}")
                print(f"Sentiment: {conversation['sentiment']}")
                print(f"confidenceScores: 
{conversation['confidenceScores']}")
Import a Conversation Project
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import 
ConversationAuthoringClient
clu_endpoint = os.environ["AZURE_CONVERSATIONS_ENDPOINT"]
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
\nOptional keyword arguments can be passed in at the client and per-operation level. The
azure-core reference documentation
 describes available configurations for retries,
logging, transport protocols, and more.
The Conversations client will raise exceptions defined in Azure Core
.
This library uses the standard
logging
 library for logging.
Basic information about
HTTP sessions (URLs, headers, etc.) is logged at INFO
level.
Detailed DEBUG level logging, including request/response bodies and unredacted
headers, can be enabled on a client with the logging_enable argument.
See full SDK logging documentation with examples here.
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
Troubleshooting
General
Logging
\nPython
Similarly, logging_enable can enable detailed logging for a single operation, even when
it isn't enabled for the client:
Python
See the Sample README
 for several code snippets illustrating common patterns used
in the CLU Python API.
See the CONTRIBUTING.md
 for details on building, testing, and contributing to this
library.
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
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
# This client will log detailed information about its HTTP sessions, at 
DEBUG level
client = ConversationAnalysisClient(endpoint, credential, 
logging_enable=True)
result = client.analyze_conversation(...)
result = client.analyze_conversation(..., logging_enable=True)
Next steps
More sample code
Contributing
\nand actually do, grant us the rights to use your contribution. For details, visit
cla.microsoft.com
.
When you submit a pull request, a CLA-bot will automatically determine whether you
need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply
follow the instructions provided by the bot. You will only need to do this once across all
repos using our CLA.
This project has adopted the Microsoft Open Source Code of Conduct
. For more
information see the Code of Conduct FAQ
 or contact opencode@microsoft.com with
any additional questions or comments.
\n![Image](images/page1730_image1.png)