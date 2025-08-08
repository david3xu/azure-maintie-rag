Note: Healthcare Entities Analysis is only available with API version v3.1 and newer.
Long-running operation begin_analyze_actions
 performs multiple analyses over one
set of documents in a single request. Currently it is supported using any combination of
the following Language APIs in a single request:
Entities Recognition
PII Entities Recognition
Linked Entity Recognition
Key Phrase Extraction
Sentiment Analysis
Custom Entity Recognition (API version 2022-05-01 and newer)
Custom Single Label Classification (API version 2022-05-01 and newer)
Custom Multi Label Classification (API version 2022-05-01 and newer)
Healthcare Entities Analysis (API version 2022-05-01 and newer)
Extractive Summarization (API version 2022-10-01-preview and newer)
Abstractive Summarization (API version 2022-10-01-preview and newer)
Python
            print(f"...Role '{role.name}' with entity '{role.entity.text}'")
    print("------------------------------------------")
print("Now, let's get all of medication dosage relations from the 
documents")
dosage_of_medication_relations = [
    entity_relation
    for doc in docs
    for entity_relation in doc.entity_relations if 
entity_relation.relation_type == 
HealthcareEntityRelation.DOSAGE_OF_MEDICATION
]
Multiple Analysis
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import (
    TextAnalyticsClient,
    RecognizeEntitiesAction,
    RecognizeLinkedEntitiesAction,
    RecognizePiiEntitiesAction,
    ExtractKeyPhrasesAction,
    AnalyzeSentimentAction,
)
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
\nkey = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
)
documents = [
    'We went to Contoso Steakhouse located at midtown NYC last week for a 
dinner party, and we adore the spot! '
    'They provide marvelous food and they have a great menu. The chief cook 
happens to be the owner (I think his name is John Doe) '
    'and he is super nice, coming out of the kitchen and greeted us all.'
    ,
    'We enjoyed very much dining in the place! '
    'The Sirloin steak I ordered was tender and juicy, and the place was 
impeccably clean. You can even pre-order from their '
    'online menu at www.contososteakhouse.com, call 312-555-0176 or send 
email to order@contososteakhouse.com! '
    'The only complaint I have is the food didn\'t come fast enough. Overall 
I highly recommend it!'
]
poller = text_analytics_client.begin_analyze_actions(
    documents,
    display_name="Sample Text Analysis",
    actions=[
        RecognizeEntitiesAction(),
        RecognizePiiEntitiesAction(),
        ExtractKeyPhrasesAction(),
        RecognizeLinkedEntitiesAction(),
        AnalyzeSentimentAction(),
    ],
)
document_results = poller.result()
for doc, action_results in zip(documents, document_results):
    print(f"\nDocument text: {doc}")
    for result in action_results:
        if result.kind == "EntityRecognition":
            print("...Results of Recognize Entities Action:")
            for entity in result.entities:
                print(f"......Entity: {entity.text}")
                print(f".........Category: {entity.category}")
                print(f".........Confidence Score: 
{entity.confidence_score}")
                print(f".........Offset: {entity.offset}")
        elif result.kind == "PiiEntityRecognition":
            print("...Results of Recognize PII Entities action:")
            for pii_entity in result.entities:
                print(f"......Entity: {pii_entity.text}")
                print(f".........Category: {pii_entity.category}")
                print(f".........Confidence Score: 
\nThe returned response is an object encapsulating multiple iterables, each representing
results of individual analyses.
Note: Multiple analysis is available in API version v3.1 and newer.
{pii_entity.confidence_score}")
        elif result.kind == "KeyPhraseExtraction":
            print("...Results of Extract Key Phrases action:")
            print(f"......Key Phrases: {result.key_phrases}")
        elif result.kind == "EntityLinking":
            print("...Results of Recognize Linked Entities action:")
            for linked_entity in result.entities:
                print(f"......Entity name: {linked_entity.name}")
                print(f".........Data source: {linked_entity.data_source}")
                print(f".........Data source language: 
{linked_entity.language}")
                print(
                    f".........Data source entity ID: 
{linked_entity.data_source_entity_id}"
                )
                print(f".........Data source URL: {linked_entity.url}")
                print(".........Document matches:")
                for match in linked_entity.matches:
                    print(f"............Match text: {match.text}")
                    print(f"............Confidence Score: 
{match.confidence_score}")
                    print(f"............Offset: {match.offset}")
                    print(f"............Length: {match.length}")
        elif result.kind == "SentimentAnalysis":
            print("...Results of Analyze Sentiment action:")
            print(f"......Overall sentiment: {result.sentiment}")
            print(
                f"......Scores: positive=
{result.confidence_scores.positive}; \
                neutral={result.confidence_scores.neutral}; \
                negative={result.confidence_scores.negative} \n"
            )
        elif result.is_error is True:
            print(
                f"...Is an error with code '{result.error.code}' and message 
'{result.error.message}'"
            )
    print("------------------------------------------")
Optional Configuration
\nOptional keyword arguments can be passed in at the client and per-operation level.
The
azure-core reference documentation

describes available configurations for retries,
logging, transport protocols, and more.
The Text Analytics client will raise exceptions defined in Azure Core
.
This library uses the standard
logging
 library for logging.
Basic information about
HTTP sessions (URLs, headers, etc.) is logged at INFO
level.
Detailed DEBUG level logging, including request/response bodies and unredacted
headers, can be enabled on a client with the logging_enable keyword argument:
Python
Similarly, logging_enable can enable detailed logging for a single operation,
even when
it isn't enabled for the client:
Troubleshooting
General
Logging
import sys
import logging
from azure.identity import DefaultAzureCredential
from azure.ai.textanalytics import TextAnalyticsClient
# Create a logger for the 'azure' SDK
logger = logging.getLogger('azure')
logger.setLevel(logging.DEBUG)
# Configure a console output
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
endpoint = "https://<resource-name>.cognitiveservices.azure.com/"
credential = DefaultAzureCredential()
# This client will log detailed information about its HTTP sessions, at 
DEBUG level
text_analytics_client = TextAnalyticsClient(endpoint, credential, 
logging_enable=True)
result = text_analytics_client.analyze_sentiment(["I did not like the 
restaurant. The food was too spicy."])
\nPython
These code samples show common scenario operations with the Azure Text Analytics
client library.
Authenticate the client with a Cognitive Services/Language service API key or a token
credential from azure-identity
:
sample_authentication.py
 (async version
)
Common scenarios
Analyze sentiment: sample_analyze_sentiment.py
 (async version
)
Recognize entities: sample_recognize_entities.py
 (async version
)
Recognize personally identifiable information: sample_recognize_pii_entities.py
(async version
)
Recognize linked entities: sample_recognize_linked_entities.py
 (async version
)
Extract key phrases: sample_extract_key_phrases.py
 (async version
)
Detect language: sample_detect_language.py
 (async version
)
Healthcare Entities Analysis: sample_analyze_healthcare_entities.py
 (async
version
)
Multiple Analysis: sample_analyze_actions.py
 (async version
)
Custom Entity Recognition: sample_recognize_custom_entities.py
(async_version
)
Custom Single Label Classification: sample_single_label_classify.py
(async_version
)
Custom Multi Label Classification: sample_multi_label_classify.py
(async_version
)
Extractive text summarization: sample_extract_summary.py
 (async version
)
Abstractive text summarization: sample_abstractive_summary.py
 (async
version
)
Dynamic Classification: sample_dynamic_classification.py
 (async_version
)
Advanced scenarios
result = text_analytics_client.analyze_sentiment(documents, 
logging_enable=True)
Next steps
More sample code
\nOpinion Mining: sample_analyze_sentiment_with_opinion_mining.py
(async_version
)
NER resolutions: sample_recognize_entity_resolutions.py
 (async_version
)
For more extensive documentation on Azure Cognitive Service for Language, see the
Language Service documentation on docs.microsoft.com.
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
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
Additional documentation
Contributing
\nArticle • 07/20/2022
Azure Pipelines
Azure Pipelines
set up now
set up now
Azure Cognitive Language Services
Question Answering client library for
Python - version 1.1.0b2
Question Answering is a cloud-based API service that lets you create a conversational
question-and-answer layer over your existing data. Use it to build a knowledge base by
extracting questions and answers from your semi-structured content, including FAQ,
manuals, and documents. Answer users’ questions with the best answers from the QnAs
in your knowledge base—automatically. Your knowledge base gets smarter, too, as it
continually learns from users' behavior.
Source code
 | Package (PyPI)
 | API reference documentation
 | Product
documentation
 | Samples
Azure SDK Python packages support for Python 2.7 ended 01 January 2022. For more
information and questions, please refer to https://github.com/Azure/azure-sdk-for-
python/issues/20691
Python 3.6 or later is required to use this package.
An Azure subscription
A Language Service resource
Install the Azure QuestionAnswering client library for Python with pip
:
Bash
Disclaimer
Getting started
Prerequisites
Install the package
pip install azure-ai-language-questionanswering --pre
\nIn order to interact with the Question Answering service, you'll need to create an
instance of the QuestionAnsweringClient
 class or an instance of the
QuestionAnsweringProjectsClient
 for managing projects within your resource. You will
need an endpoint, and an API key to instantiate a client object. For more information
regarding authenticating with Cognitive Services, see Authenticate requests to Azure
Cognitive Services.
You can get the endpoint and an API key from the Cognitive Services resource or
Question Answering resource in the Azure Portal
.
Alternatively, use the Azure CLI command shown below to get the API key from the
Question Answering resource.
PowerShell
Once you've determined your endpoint and API key you can instantiate a
QuestionAnsweringClient :
Python
With your endpoint and API key, you can instantiate a
QuestionAnsweringProjectsClient
:
Python
Authenticate the client
Get an API key
az cognitiveservices account keys list --resource-group <resource-group-
name> --name <resource-name>
Create QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient
endpoint = "https://{myaccount}.api.cognitive.microsoft.com"
credential = AzureKeyCredential("{api-key}")
client = QuestionAnsweringClient(endpoint, credential)
Create QuestionAnsweringProjectsClient
\nTo use an Azure Active Directory (AAD) token credential, provide an instance of the
desired credential type obtained from the azure-identity
 library. Note that regional
endpoints do not support AAD authentication. Create a custom subdomain name for
your resource in order to use this type of authentication.
Authentication with AAD requires some initial setup:
Install azure-identity
Register a new AAD application
Grant access to the Language service by assigning the "Cognitive Services
Language Reader" role to your service principal.
After setup, you can choose which type of credential
 from azure.identity to use. As an
example, DefaultAzureCredential
 can be used to authenticate the client:
Set the values of the client ID, tenant ID, and client secret of the AAD application as
environment variables: AZURE_CLIENT_ID , AZURE_TENANT_ID , AZURE_CLIENT_SECRET
Use the returned token credential to authenticate the client:
Python
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering.projects import 
QuestionAnsweringProjectsClient
endpoint = "https://{myaccount}.api.cognitive.microsoft.com"
credential = AzureKeyCredential("{api-key}")
client = QuestionAnsweringProjectsClient(endpoint, credential)
Create a client with an Azure Active Directory Credential
from azure.ai.textanalytics import QuestionAnsweringClient
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
client = QuestionAnsweringClient(endpoint="https://<my-custom-
subdomain>.cognitiveservices.azure.com/", credential=credential)
Key concepts
QuestionAnsweringClient
\nThe QuestionAnsweringClient
 is the primary interface for asking questions using a
knowledge base with your own information, or text input using pre-trained models. For
asynchronous operations, an async QuestionAnsweringClient  is in the
azure.ai.language.questionanswering.aio  namespace.
The QuestionAnsweringProjectsClient
 provides an interface for managing Question
Answering projects. Examples of the available operations include creating and deploying
projects, updating your knowledge sources, and updating question and answer pairs. It
provides both synchronous and asynchronous APIs.
The azure-ai-language-questionanswering  client library provides both synchronous and
asynchronous APIs.
The following examples show common scenarios using the client  created above.
Ask a question
Ask a follow-up question
Asynchronous operations
The only input required to ask a question using a knowledge base is just the question
itself:
Python
QuestionAnsweringProjectsClient
Examples
QuestionAnsweringClient
Ask a question
output = client.get_answers(
    question="How long should my Surface battery last?",
    project_name="FAQ",
    deployment_name="test"
)
for candidate in output.answers:
    print("({}) {}".format(candidate.confidence, candidate.answer))
    print("Source: {}".format(candidate.source))