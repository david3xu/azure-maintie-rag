Feedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Try the Ingestion Client
Yes
No
\nIngestion Client with Azure AI services
Article • 03/10/2025
The Ingestion Client is a tool released by Microsoft on GitHub that helps you quickly
deploy a call center transcription solution to Azure with a no-code approach.
Ingestion Client uses the Azure AI Language, Azure AI Speech, Azure storage
, and
Azure Functions
.
An Azure account and a multi-service Azure AI services resource are needed to run the
Ingestion Client.
Azure subscription - Create one for free
Create an Azure AI services resource
 in the Azure portal.
Get the resource key and region. After your resource is deployed, select Go to
resource to view and manage keys. For more information about Azure AI services
resources, see this quickstart.
The Ingestion Client works by connecting a dedicated Azure storage
 account to
custom Azure Functions
 in a serverless fashion to pass transcription requests to the
service. The transcribed audio files land in the dedicated Azure Storage container
.
 Tip
You can use the tool and resulting solution in production to process a high volume
of audio.
Get started with the Ingestion Client
Ingestion Client Features
） Important
Pricing varies depending on the mode of operation (batch vs real-time) as well as
the Azure Function SKU selected. By default the tool will create a Premium Azure
Function SKU to handle large volume. Visit the Pricing
 page for more
information.
\nInternally, the tool uses Speech and Language services, and follows best practices to
handle scale-up, retries and failover. The following schematic describes the resources
and connections.
The following Speech service feature is used by the Ingestion Client:
Batch speech to text: Transcribe large amounts of audio files asynchronously
including speaker diarization and is typically used in post-call analytics scenarios.
Diarization is the process of recognizing and separating speakers in mono channel
audio data.
Here are some Language service features that are used by the Ingestion Client:
Personally Identifiable Information (PII) extraction and redaction: Identify,
categorize, and redact sensitive information in conversation transcription.
Sentiment analysis and opinion mining: Analyze transcriptions and associate
positive, neutral, or negative sentiment at the utterance and conversation-level.
Besides Azure AI services, these Azure products are used to complete the solution:
Azure storage
: Used for storing telephony data and the transcripts that batch
transcription API returns. This storage account should use notifications, specifically
for when new files are added. These notifications are used to trigger the
transcription process.
Azure Functions
: Used for creating the shared access signature (SAS) URI for
each recording, and triggering the HTTP POST request to start a transcription.
Additionally, you use Azure Functions to create requests to retrieve and delete
transcriptions by using the Batch Transcription API.
\n![Image](images/page1563_image1.png)
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
The tool is built to show customers results quickly. You can customize the tool to your
preferred SKUs and setup. The SKUs can be edited from the Azure portal
 and the code
itself is available on GitHub
.
Learn more about Azure AI services features for call center
Explore the Language service features
Explore the Speech service features
Tool customization
７ Note
We suggest creating the resources in the same dedicated resource group to
understand and track costs more easily.
Next steps
Yes
No
\nAzure AI Language REST API reference
06/10/2025
The API reference for authoring and runtime APIs for Azure AI Language.
Service
Description
API Reference
(Latest GA
version)
API Reference
(Latest Preview
version)
Text - runtime
Runtime prediction calls to the
following features:
* entity linking
* key phrase extraction
* language detection
* Named Entity Recognition
* language detection
* Personally Identifiable Information
(PII)
* sentiment analysis
* text analytics for health
* summarization
* custom text classification
* custom Named Entity Recognition
(NER)
2024-11-01
2025-05-15-
preview
Documents - runtime
* Personally Identifiable Information
(PII)
* summarization
Not available
2025-05-15-
preview
Conversation - runtime
* Personally Identifiable Information
(PII)
* summarization
2024-11-01
2025-05-15-
preview
Text - authoring
Authoring API calls to create, build,
train, and deploy your projects Custom
Text Classification or Custom Named
Entity Recognition projects.
2023-04-01
2025-05-15-
preview
Conversational Language
Understanding &
Orchestration workflow -
authoring
Authoring API calls to create, build,
manage, train, and deploy your CLU
projects
2023-04-01
2025-05-15-
preview
Links to services
ﾉ
Expand table
\nService
Description
API Reference
(Latest GA
version)
API Reference
(Latest Preview
version)
Conversational Language
Understanding &
Orchestration workflow -
runtime
Runtime prediction calls to query your
deployed CLU project
2024-11-01
2025-05-15-
preview
Custom Question
Answering - authoring
Authoring API calls to create, build,
and deploy your projects
2023-04-01
2023-04-15-
preview
Custom Question
Answering - runtime
Runtime prediction calls to query
custom question answering models.
2023-04-01
2023-04-15-
preview
Question Answering -
runtime
Runtime prediction calls to query
question answering models.
2023-04-01
2023-04-15-
preview
\nAzure Cognitive Services Text Analytics
client library for .NET - version 5.3.0
Article • 06/20/2023
Text Analytics is part of the Azure Cognitive Service for Language, a cloud-based service that
provides Natural Language Processing (NLP) features for understanding and analyzing text.
This client library offers the following features:
Language detection
Sentiment analysis
Key phrase extraction
Named entity recognition (NER)
Personally identifiable information (PII) entity recognition
Entity linking
Text analytics for health
Custom named entity recognition (Custom NER)
Custom text classification
Extractive text summarization
Abstractive text summarization
Source code
 | Package (NuGet)
 | API reference documentation
 | Product documentation
| Samples
Install the Azure Text Analytics client library for .NET with NuGet
:
.NET CLI
This table shows the relationship between SDK versions and supported API versions of the
service:
Note that 5.2.0  is the first stable version of the client library that targets the Azure
Cognitive Service for Language APIs which includes the existing text analysis and natural
language processing features found in the Text Analytics client library. In addition, the
service API has changed from semantic to date-based versioning.
Getting started
Install the package
dotnet add package Azure.AI.TextAnalytics
\nSDK version
Supported API version of service
5.3.X
3.0, 3.1, 2022-05-01, 2023-04-01 (default)
5.2.X
3.0, 3.1, 2022-05-01 (default)
5.1.X
3.0, 3.1 (default)
5.0.X
3.0
1.0.X
3.0
An Azure subscription
.
An existing Cognitive Services or Language service resource.
Azure Cognitive Service for Language supports both multi-service and single-service access.
Create a Cognitive Services resource if you plan to access multiple cognitive services under a
single endpoint and API key. To access the features of the Language service only, create a
Language service resource instead.
You can create either resource via the Azure portal or, alternatively, you can follow the steps in
this document to create it using the Azure CLI.
Interaction with the service using the client library begins with creating an instance of the
TextAnalyticsClient
 class. You will need an endpoint, and either an API key or
TokenCredential  to instantiate a client object. For more information regarding authenticating
with cognitive services, see Authenticate requests to Azure Cognitive Services.
You can get the endpoint  and API key  from the Cognitive Services resource or Language
service resource information in the Azure Portal
.
Alternatively, use the Azure CLI snippet below to get the API key from the Language service
resource.
ﾉ
Expand table
Prerequisites
Create a Cognitive Services resource or a Language service resource
Authenticate the client
Get an API key
\nPowerShell
Once you have the value for the API key, create an AzureKeyCredential . This will allow you to
update the API key without creating a new client.
With the value of the endpoint and an AzureKeyCredential , you can create the
TextAnalyticsClient
:
C#
Client API key authentication is used in most of the examples in this getting started guide, but
you can also authenticate with Azure Active Directory using the Azure Identity library
. Note
that regional endpoints do not support AAD authentication. Create a custom subdomain for
your resource in order to use this type of authentication.
To use the DefaultAzureCredential
 provider shown below, or other credential providers
provided with the Azure SDK, please install the Azure.Identity package:
.NET CLI
You will also need to register a new AAD application and grant access to the Language service
by assigning the "Cognitive Services User"  role to your service principal.
Set the values of the client ID, tenant ID, and client secret of the AAD application as
environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET.
C#
az cognitiveservices account keys list --resource-group <your-resource-group-name> 
--name <your-resource-name>
Create a TextAnalyticsClient  using an API key credential
Uri endpoint = new("<endpoint>");
AzureKeyCredential credential = new("<apiKey>");
TextAnalyticsClient client = new(endpoint, credential);
Create a TextAnalyticsClient  with an Azure Active Directory credential
dotnet add package Azure.Identity
Uri endpoint = new("<endpoint>");
TextAnalyticsClient client = new(endpoint, new DefaultAzureCredential());
\nA TextAnalyticsClient  is the primary interface for developers using the Text Analytics client
library. It provides both synchronous and asynchronous operations to access a specific use of
text analysis, such as language detection or key phrase extraction.
A document, is a single unit of input to be analyzed by the predictive models in the Language
service. Operations on TextAnalyticsClient  may take a single document or a collection of
documents to be analyzed as a batch. For document length limits, maximum batch size, and
supported text encoding see here
.
For each supported operation, TextAnalyticsClient  provides a method that accepts a batch of
documents as strings, or a batch of either TextDocumentInput  or DetectLanguageInput  objects.
This methods allow callers to give each document a unique ID, indicate that the documents in
the batch are written in different languages, or provide a country hint about the language of
the document.
Note: It is recommended to use the batch methods when working on production environments
as they allow you to send one request with multiple documents. This is more performant than
sending a request per each document.
Return values, such as AnalyzeSentimentResult , is the result of a Text Analytics operation,
containing a prediction or predictions about a single document. An operation's return value
also may optionally include information about the document and how it was processed.
A Return value collection, such as AnalyzeSentimentResultCollection , is a collection of
operation results, where each corresponds to one of the documents provided in the input
batch. A document and its result will have the same index in the input and result collections.
The return value also contains a HasError  property that allows to identify if an operation
Key concepts
TextAnalyticsClient
Input
Operation on multiple documents
Return value
Return value Collection