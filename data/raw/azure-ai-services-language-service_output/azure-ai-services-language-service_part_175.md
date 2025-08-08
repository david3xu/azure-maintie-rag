you to build custom models for text classification tasks.
Single label classification
Multi label classification
For more information see How to use: Custom Text Classification.
The Analyze  functionality allows choosing which of the supported Language service features to
execute in the same set of documents. Currently, the supported features are:
Named Entities Recognition
PII Entities Recognition
Linked Entity Recognition
Key Phrase Extraction
Sentiment Analysis
Healthcare Analysis
Custom Entity Recognition (API version 2022-05-01 and newer)
Custom Single-Label Classification (API version 2022-05-01 and newer)
Custom Multi-Label Classification (API version 2022-05-01 and newer)
Abstractive Text Summarization (API version 2023-04-01 and newer)
Extractive Text Summarization (API version 2023-04-01 and newer)
Sample: Multiple action analysis
For more examples, such as asynchronous samples, refer to here
.
Text Analytics clients raise exceptions. For example, if you try to detect the languages of a
batch of text with same document IDs, 400  error is return that indicating bad request. In the
following code snippet, the error is handled gracefully by catching the exception and display
the additional information about the error.
Java
Analyze multiple actions
Troubleshooting
General
List<DetectLanguageInput> documents = Arrays.asList(
    new DetectLanguageInput("1", "This is written in English.", "us"),
    new DetectLanguageInput("1", "Este es un documento  escrito en Español.", 
\nYou can set the AZURE_LOG_LEVEL  environment variable to view logging statements made in the
client library. For example, setting AZURE_LOG_LEVEL=2  would show all informational, warning,
and error log messages. The log levels can be found here: log levels
.
All client libraries by default use the Netty HTTP client. Adding the above dependency will
automatically configure the client library to use the Netty HTTP client. Configuring or changing
the HTTP client is detailed in the HTTP clients wiki.
All client libraries, by default, use the Tomcat-native Boring SSL library to enable native-level
performance for SSL operations. The Boring SSL library is an uber jar containing native libraries
for Linux / macOS / Windows, and provides better performance compared to the default SSL
implementation within the JDK. For more information, including how to reduce the
dependency size, refer to the performance tuning
 section of the wiki.
Samples are explained in detail here
.
This project welcomes contributions and suggestions. Most contributions require you to agree
to a Contributor License Agreement (CLA)
 declaring that you have the right to, and actually
do, grant us the rights to use your contribution.
"es")
);
try {
    textAnalyticsClient.detectLanguageBatchWithResponse(documents, null, 
Context.NONE);
} catch (HttpResponseException e) {
    System.out.println(e.getMessage());
}
Enable client logging
Default HTTP Client
Default SSL library
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
Impressions
\n![Image](images/page1743_image1.png)
\nAzure Text Analysis client library for
JavaScript - version 1.1.0-beta.2
Article • 03/07/2023
Azure Cognitive Service for Language
 is a cloud-based service that provides advanced
natural language processing over raw text, and includes the following main features:
Note: This SDK targets Azure Cognitive Service for Language API version 2022-10-01-
preview.
Language Detection
Sentiment Analysis
Key Phrase Extraction
Named Entity Recognition
Recognition of Personally Identifiable Information
Entity Linking
Healthcare Analysis
Extractive Summarization
Abstractive Summarization
Custom Entity Recognition
Custom Document Classification
Dynamic Classification
Support Multiple Actions Per Document
Use the client library to:
Detect what language input text is written in.
Determine what customers think of your brand or topic by analyzing raw text for
clues about positive or negative sentiment.
Automatically extract key phrases to quickly identify the main points.
Identify and categorize entities in your text as people, places, organizations,
date/time, quantities, percentages, currencies, healthcare specific, and more.
Perform multiple of the above tasks at once.
Key links:
Source code
Package (NPM)
API reference documentation
Product documentation
Samples⚠️
\nPlease see the Migration Guide
 for detailed instructions on how to update application
code from version 5.x of the AI Text Analytics client library to the new AI Language Text
client library.
Abstractive Summarization
Dynamic Classification
Script Detection
Automatic Language Detection
Entity Resolutions
Specifying healthcare document type for better FHIR results
LTS versions of Node.js
Latest versions of Safari, Chrome, Edge, and Firefox.
See our support policy
 for more details.
An Azure subscription
.
An existing Cognitive Services or Language resource. If you need to create the
resource, you can use the Azure Portal
 or Azure CLI following the steps in this
document.
If you use the Azure CLI, replace <your-resource-group-name> and <your-resource-name>
with your own unique names:
PowerShell
Migrating from @azure/ai-text-analytics advisory ⚠️
What's New
Getting started
Currently supported environments
Prerequisites
az cognitiveservices account create --kind TextAnalytics --resource-group 
<your-resource-group-name> --name <your-resource-name> --sku <your-sku-name> 
--location <your-location>
\nInstall the Azure Text Analysis client library for JavaScript with npm:
Bash
To create a client object to access the Language API, you will need the endpoint of your
Language resource and a credential. The Text Analysis client can use either Azure
Active Directory credentials or an API key credential to authenticate.
You can find the endpoint for your Language resource either in the Azure Portal
 or by
using the Azure CLI snippet below:
Bash
Use the Azure Portal
 to browse to your Language resource and retrieve an API key, or
use the Azure CLI snippet below:
Note: Sometimes the API key is referred to as a "subscription key" or "subscription API
key."
PowerShell
Once you have an API key and endpoint, you can use the AzureKeyCredential class to
authenticate the client as follows:
JavaScript
Install the @azure/ai-language-text  package
npm install @azure/ai-language-text
Create and authenticate a TextAnalysisClient
az cognitiveservices account show --name <your-resource-name> --resource-
group <your-resource-group-name> --query "properties.endpoint"
Using an API Key
az cognitiveservices account keys list --resource-group <your-resource-
group-name> --name <your-resource-name>
const { TextAnalysisClient, AzureKeyCredential } = require("@azure/ai-
language-text");
\nClient API key authentication is used in most of the examples, but you can also
authenticate with Azure Active Directory using the Azure Identity library
. To use the
DefaultAzureCredential
 provider shown below,
or other credential providers provided
with the Azure SDK, please install the @azure/identity package:
Bash
You will also need to register a new AAD application and grant access to Language by
assigning the "Cognitive Services User" role to your service principal (note: other roles
such as "Owner" will not grant the necessary permissions, only "Cognitive Services
User" will suffice to run the examples and the sample code).
Set the values of the client ID, tenant ID, and client secret of the AAD application as
environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET.
JavaScript
TextAnalysisClient is the primary interface for developers using the Text Analysis client
library. Explore the methods on this client object to understand the different features of
the Language service that you can access.
const client = new TextAnalysisClient("<endpoint>", new AzureKeyCredential("
<API key>"));
Using an Azure Active Directory Credential
npm install @azure/identity
const { TextAnalysisClient } = require("@azure/ai-language-text");
const { DefaultAzureCredential } = require("@azure/identity");
const client = new TextAnalysisClient("<endpoint>", new 
DefaultAzureCredential());
Key concepts
TextAnalysisClient
Input
\nA document represents a single unit of input to be analyzed by the predictive models in
the Language service. Operations on TextAnalysisClient take a collection of inputs to
be analyzed as a batch. The operation methods have overloads that allow the inputs to
be represented as strings, or as objects with attached metadata.
For example, each document can be passed as a string in an array, e.g.
TypeScript
or, if you wish to pass in a per-item document id or language/countryHint, they can be
given as a list of TextDocumentInput or DetectLanguageInput depending on the
operation;
JavaScript
See service limitations for the input, including document length limits, maximum batch
size, and supported text encodings.
The return value corresponding to a single document is either a successful result or an
error object. Each TextAnalysisClient method returns a heterogeneous array of results
and errors that correspond to the inputs by index. A text input and its result will have
the same index in the input and result collections.
An result, such as SentimentAnalysisResult, is the result of a Language operation,
containing a prediction or predictions about a single text input. An operation's result
type also may optionally include information about the input document and how it was
processed.
const documents = [
  "I hated the movie. It was so slow!",
  "The movie made it into my top ten favorites.",
  "What a great movie!",
];
const textDocumentInputs = [
  { id: "1", language: "en", text: "I hated the movie. It was so slow!" },
  { id: "2", language: "en", text: "The movie made it into my top ten 
favorites." },
  { id: "3", language: "en", text: "What a great movie!" },
];
Return Value
\nThe error object, TextAnalysisErrorResult, indicates that the service encountered an
error while processing the document and contains information about the error.
In the collection returned by an operation, errors are distinguished from successful
responses by the presence of the error property, which contains the inner
TextAnalysisError object if an error was encountered. For successful result objects, this
property is always undefined.
For example, to filter out all errors, you could use the following filter:
JavaScript
Note: TypeScript users can benefit from better type-checking of result and error objects
if compilerOptions.strictNullChecks is set to true in the tsconfig.json configuration.
For example:
TypeScript
Actions Batching
Choose Model Version
Paging
Rehydrate Polling
Document Error Handling
const results = await client.analyze("SentimentAnalysis", documents);
const onlySuccessful = results.filter((result) => result.error === 
undefined);
const [result] = await client.analyze("SentimentAnalysis", ["Hello 
world!"]);
if (result.error !== undefined) {
  // In this if block, TypeScript will be sure that the type of `result` is
  // `TextAnalysisError` if compilerOptions.strictNullChecks is enabled in
  // the tsconfig.json
  console.log(result.error);
}
Samples
Client Usage
\nGet Statistics
Abstractive Summarization
Dynamic Classification
Language Detection
Entity Linking
Entity Regconition
Extractive Summarization
Healthcare Analysis
Key Phrase Extraction
Language Detection
Opinion Mining
PII Entity Recognition
Sentiment Analysis
Custom Entity Recognition
Custom Single-lable Classfication
Custom Multi-lable Classfication
Enabling logging may help uncover useful information about failures. In order to see a
log of HTTP requests and responses, set the AZURE_LOG_LEVEL environment variable to
info. Alternatively, logging can be enabled at runtime by calling setLogLevel in the
@azure/logger:
JavaScript
For more detailed instructions on how to enable logs, you can look at the
@azure/logger package docs
.
Prebuilt Tasks
Custom Tasks
Troubleshooting
Logging
const { setLogLevel } = require("@azure/logger");
setLogLevel("info");