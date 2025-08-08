Please see the Migration Guide
 for detailed instructions on how to update application code
from version 5.x of the AI Text Analytics client library to the new AI Language Text client library.
Abstractive Summarization
Healthcare Analysis
LTS versions of Node.js
Latest versions of Safari, Chrome, Edge, and Firefox.
See our support policy
 for more details.
An Azure subscription
.
An existing Cognitive Services or Language resource. If you need to create the resource,
you can use the Azure Portal
 or Azure CLI following the steps in this document.
If you use the Azure CLI, replace <your-resource-group-name>  and <your-resource-name>  with
your own unique names:
PowerShell
Install the Azure Text Analysis client library for JavaScript with npm :
Bash
What's New
Getting started
Currently supported environments
Prerequisites
az cognitiveservices account create --kind TextAnalytics --resource-group <your-
resource-group-name> --name <your-resource-name> --sku <your-sku-name> --location 
<your-location>
Install the @azure/ai-language-text  package
npm install @azure/ai-language-text
\nTo create a client object to access the Language API, you will need the endpoint  of your
Language resource and a credential . The Text Analysis client can use either Azure Active
Directory credentials or an API key credential to authenticate.
You can find the endpoint for your Language resource either in the Azure Portal
 or by using
the Azure CLI snippet below:
Bash
Use the Azure Portal
 to browse to your Language resource and retrieve an API key, or use
the Azure CLI snippet below:
Note: Sometimes the API key is referred to as a "subscription key" or "subscription API key."
PowerShell
Once you have an API key and endpoint, you can use the AzureKeyCredential  class to
authenticate the client as follows:
JavaScript
Client API key authentication is used in most of the examples, but you can also authenticate
with Azure Active Directory using the Azure Identity library
. To use the
DefaultAzureCredential
 provider shown below, or other credential providers provided with
the Azure SDK, please install the @azure/identity  package:
Create and authenticate a TextAnalysisClient
az cognitiveservices account show --name <your-resource-name> --resource-group 
<your-resource-group-name> --query "properties.endpoint"
Using an API Key
az cognitiveservices account keys list --resource-group <your-resource-group-name> 
--name <your-resource-name>
const { TextAnalysisClient, AzureKeyCredential } = require("@azure/ai-language-
text");
const client = new TextAnalysisClient("<endpoint>", new AzureKeyCredential("<API 
key>"));
Using an Azure Active Directory Credential
\nBash
You will also need to register a new AAD application and grant access to Language by
assigning the "Cognitive Services User"  role to your service principal (note: other roles such
as "Owner"  will not grant the necessary permissions, only "Cognitive Services User"  will
suffice to run the examples and the sample code).
Set the values of the client ID, tenant ID, and client secret of the AAD application as
environment variables: AZURE_CLIENT_ID , AZURE_TENANT_ID , AZURE_CLIENT_SECRET .
JavaScript
TextAnalysisClient  is the primary interface for developers using the Text Analysis client library.
Explore the methods on this client object to understand the different features of the Language
service that you can access.
A document represents a single unit of input to be analyzed by the predictive models in the
Language service. Operations on TextAnalysisClient  take a collection of inputs to be analyzed
as a batch. The operation methods have overloads that allow the inputs to be represented as
strings, or as objects with attached metadata.
For example, each document can be passed as a string in an array, e.g.
TypeScript
npm install @azure/identity
const { TextAnalysisClient } = require("@azure/ai-language-text");
const { DefaultAzureCredential } = require("@azure/identity");
const client = new TextAnalysisClient("<endpoint>", new DefaultAzureCredential());
Key concepts
TextAnalysisClient
Input
const documents = [
  "I hated the movie. It was so slow!",
  "The movie made it into my top ten favorites.",
\nor, if you wish to pass in a per-item document id  or language / countryHint , they can be given
as a list of TextDocumentInput  or DetectLanguageInput  depending on the operation;
JavaScript
See service limitations for the input, including document length limits, maximum batch size,
and supported text encodings.
The return value corresponding to a single document is either a successful result or an error
object. Each TextAnalysisClient  method returns a heterogeneous array of results and errors
that correspond to the inputs by index. A text input and its result will have the same index in
the input and result collections.
An result, such as SentimentAnalysisResult , is the result of a Language operation, containing a
prediction or predictions about a single text input. An operation's result type also may
optionally include information about the input document and how it was processed.
The error object, TextAnalysisErrorResult , indicates that the service encountered an error
while processing the document and contains information about the error.
In the collection returned by an operation, errors are distinguished from successful responses
by the presence of the error  property, which contains the inner TextAnalysisError  object if an
error was encountered. For successful result objects, this property is always undefined .
For example, to filter out all errors, you could use the following filter :
JavaScript
  "What a great movie!",
];
const textDocumentInputs = [
  { id: "1", language: "en", text: "I hated the movie. It was so slow!" },
  { id: "2", language: "en", text: "The movie made it into my top ten favorites." 
},
  { id: "3", language: "en", text: "What a great movie!" },
];
Return Value
Document Error Handling
const results = await client.analyze("SentimentAnalysis", documents);
\nNote: TypeScript users can benefit from better type-checking of result and error objects if
compilerOptions.strictNullChecks  is set to true  in the tsconfig.json  configuration. For
example:
TypeScript
Actions Batching
Choose Model Version
Paging
Rehydrate Polling
Get Statistics
Abstractive Summarization
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
const onlySuccessful = results.filter((result) => result.error === undefined);
const [result] = await client.analyze("SentimentAnalysis", ["Hello world!"]);
if (result.error !== undefined) {
  // In this if block, TypeScript will be sure that the type of `result` is
  // `TextAnalysisError` if compilerOptions.strictNullChecks is enabled in
  // the tsconfig.json
  console.log(result.error);
}
Samples
Client Usage
Prebuilt Tasks
\nCustom Entity Recognition
Custom Single-lable Classfication
Custom Multi-lable Classfication
Enabling logging may help uncover useful information about failures. In order to see a log of
HTTP requests and responses, set the AZURE_LOG_LEVEL  environment variable to info .
Alternatively, logging can be enabled at runtime by calling setLogLevel  in the @azure/logger :
JavaScript
For more detailed instructions on how to enable logs, you can look at the @azure/logger
package docs
.
Please take a look at the samples
 directory for detailed examples on how to use this library.
If you'd like to contribute to this library, please read the contributing guide
 to learn more
about how to build and test the code.
Microsoft Azure SDK for JavaScript
Custom Tasks
Troubleshooting
Logging
const { setLogLevel } = require("@azure/logger");
setLogLevel("info");
Next steps
Contributing
Related projects
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