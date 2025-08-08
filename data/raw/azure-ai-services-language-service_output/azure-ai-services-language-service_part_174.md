Azure Text Analytics client library for Java -
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
\nRegister a new Azure Active Directory application
Grant access to Language service by assigning the "Cognitive Services User"  role to
your service principal.
After setup, you can choose which type of credential
 from azure.identity to use. As an
example, DefaultAzureCredential can be used to authenticate the client: Set the values of the
client ID, tenant ID, and client secret of the AAD application as environment variables:
AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET.
Authorization is easiest using DefaultAzureCredential. It finds the best credential to use in its
running environment. For more information about using Azure Active Directory authorization
with Language service, please refer to the associated documentation.
Java
The Text Analytics client library provides a TextAnalyticsClient
 and TextAnalyticsAsyncClient
to do analysis on batches of documents. It provides both synchronous and asynchronous
operations to access a specific use of Language service, such as language detection or key
phrase extraction.
A text input, also called a document, is a single unit of document to be analyzed by the
predictive models in the Language service. Operations on a Text Analytics client may take a
single document or a collection of documents to be analyzed as a batch. See service limitations
for the document, including document length limits, maximum batch size, and supported text
encoding.
TokenCredential defaultCredential = new DefaultAzureCredentialBuilder().build();
TextAnalyticsAsyncClient textAnalyticsAsyncClient = new 
TextAnalyticsClientBuilder()
    .endpoint("{endpoint}")
    .credential(defaultCredential)
    .buildAsyncClient();
Key concepts
Text Analytics client
Input
Operation on multiple documents
\nFor each supported operation, the Text Analytics client provides method overloads to take a
single document, a batch of documents as strings, or a batch of either TextDocumentInput  or
DetectLanguageInput  objects. The overload taking the TextDocumentInput  or
DetectLanguageInput  batch allows callers to give each document a unique ID, indicate that the
documents in the batch are written in different languages, or provide a country hint about the
language of the document.
An operation result, such as AnalyzeSentimentResult , is the result of a Language service
operation, containing a prediction or predictions about a single document and a list of
warnings inside of it. An operation's result type also may optionally include information about
the input document and how it was processed. An operation result contains a isError
property that allows to identify if an operation executed was successful or unsuccessful for the
given document. When the operation results an error, you can simply call getError()  to get
TextAnalyticsError  which contains the reason why it is unsuccessful. If you are interested in
how many characters are in your document, or the number of operation transactions that have
gone through, simply call getStatistics()  to get the TextDocumentStatistics  which contains
both information.
An operation result collection, such as AnalyzeSentimentResultCollection , which is the
collection of the result of analyzing sentiment operation. It also includes the model version of
the operation and statistics of the batch documents.
Note: It is recommended to use the batch methods when working on production environments
as they allow you to send one request with multiple documents. This is more performant than
sending a request per each document.
The following sections provide several code snippets covering some of the most common
Language service tasks, including:
Analyze Sentiment
Detect Language
Extract Key Phrases
Recognize Named Entities
Recognize Personally Identifiable Information Entities
Return value
Return value collection
Examples
\nRecognize Linked Entities
Analyze Healthcare Entities
Analyze Multiple Actions
Custom Entities Recognition
Custom Text Classification
Abstractive Text Summarization
Extractive Text Summarization
Language service supports both synchronous and asynchronous client creation by using
TextAnalyticsClientBuilder ,
Java
or
Java
Run a predictive model to identify the positive, negative, neutral or mixed sentiment contained
in the provided document or batch of documents.
Java
Text Analytics Client
TextAnalyticsClient textAnalyticsClient = new TextAnalyticsClientBuilder()
    .credential(new AzureKeyCredential("{key}"))
    .endpoint("{endpoint}")
    .buildClient();
TextAnalyticsAsyncClient textAnalyticsAsyncClient = new 
TextAnalyticsClientBuilder()
    .credential(new AzureKeyCredential("{key}"))
    .endpoint("{endpoint}")
    .buildAsyncClient();
Analyze sentiment
String document = "The hotel was dark and unclean. I like microsoft.";
DocumentSentiment documentSentiment = 
textAnalyticsClient.analyzeSentiment(document);
System.out.printf("Analyzed document sentiment: %s.%n", 
documentSentiment.getSentiment());
documentSentiment.getSentences().forEach(sentenceSentiment ->
\nFor samples on using the production recommended option AnalyzeSentimentBatch  see here
.
To get more granular information about the opinions related to aspects of a product/service,
also knows as Aspect-based Sentiment Analysis in Natural Language Processing (NLP), see
sample on sentiment analysis with opinion mining see here
.
Please refer to the service documentation for a conceptual discussion of sentiment analysis.
Run a predictive model to determine the language that the provided document or batch of
documents are written in.
Java
For samples on using the production recommended option DetectLanguageBatch  see here
.
Please refer to the service documentation for a conceptual discussion of language detection.
Run a model to identify a collection of significant phrases found in the provided document or
batch of documents.
Java
For samples on using the production recommended option ExtractKeyPhrasesBatch  see
here
. Please refer to the service documentation for a conceptual discussion of key phrase
extraction.
    System.out.printf("Analyzed sentence sentiment: %s.%n", 
sentenceSentiment.getSentiment()));
Detect language
String document = "Bonjour tout le monde";
DetectedLanguage detectedLanguage = textAnalyticsClient.detectLanguage(document);
System.out.printf("Detected language name: %s, ISO 6391 name: %s, confidence 
score: %f.%n",
    detectedLanguage.getName(), detectedLanguage.getIso6391Name(), 
detectedLanguage.getConfidenceScore());
Extract key phrases
String document = "My cat might need to see a veterinarian.";
System.out.println("Extracted phrases:");
textAnalyticsClient.extractKeyPhrases(document).forEach(keyPhrase -> 
System.out.printf("%s.%n", keyPhrase));
\nRun a predictive model to identify a collection of named entities in the provided document or
batch of documents and categorize those entities into categories such as person, location, or
organization. For more information on available categories, see Named Entity Categories.
Java
For samples on using the production recommended option RecognizeEntitiesBatch  see
here
. Please refer to the service documentation for a conceptual discussion of named entity
recognition.
Run a predictive model to identify a collection of Personally Identifiable Information(PII)
entities in the provided document. It recognizes and categorizes PII entities in its input text,
such as Social Security Numbers, bank account information, credit card numbers, and more.
This endpoint is only supported for API versions v3.1-preview.1 and above.
Java
For samples on using the production recommended option RecognizePiiEntitiesBatch  see
here
. Please refer to the service documentation for supported PII entity types.
Recognize named entities
String document = "Satya Nadella is the CEO of Microsoft";
textAnalyticsClient.recognizeEntities(document).forEach(entity ->
    System.out.printf("Recognized entity: %s, category: %s, subcategory: %s, 
confidence score: %f.%n",
        entity.getText(), entity.getCategory(), entity.getSubcategory(), 
entity.getConfidenceScore()));
Recognize Personally Identifiable Information entities
String document = "My SSN is 859-98-0987";
PiiEntityCollection piiEntityCollection = 
textAnalyticsClient.recognizePiiEntities(document);
System.out.printf("Redacted Text: %s%n", piiEntityCollection.getRedactedText());
piiEntityCollection.forEach(entity -> System.out.printf(
    "Recognized Personally Identifiable Information entity: %s, entity category: 
%s, entity subcategory: %s,"
        + " confidence score: %f.%n",
    entity.getText(), entity.getCategory(), entity.getSubcategory(), 
entity.getConfidenceScore()));
Recognize linked entities
\nRun a predictive model to identify a collection of entities found in the provided document or
batch of documents, and include information linking the entities to their corresponding entries
in a well-known knowledge base.
Java
For samples on using the production recommended option RecognizeLinkedEntitiesBatch  see
here
. Please refer to the service documentation for a conceptual discussion of entity linking.
Text Analytics for health is a containerized service that extracts and labels relevant medical
information from unstructured texts such as doctor's notes, discharge summaries, clinical
documents, and electronic health records.
Healthcare entities recognition
For more information see How to: Use Text Analytics for health.
Custom NER is one of the custom features offered by Azure Cognitive Service for Language. It
is a cloud-based API service that applies machine-learning intelligence to enable you to build
custom models for custom named entity recognition tasks.
Custom entities recognition
For more information see How to use: Custom Entities Recognition.
Custom text classification is one of the custom features offered by Azure Cognitive Service for
Language. It is a cloud-based API service that applies machine-learning intelligence to enable
String document = "Old Faithful is a geyser at Yellowstone Park.";
textAnalyticsClient.recognizeLinkedEntities(document).forEach(linkedEntity -> {
    System.out.println("Linked Entities:");
    System.out.printf("Name: %s, entity ID in data source: %s, URL: %s, data 
source: %s.%n",
        linkedEntity.getName(), linkedEntity.getDataSourceEntityId(), 
linkedEntity.getUrl(), linkedEntity.getDataSource());
    linkedEntity.getMatches().forEach(match ->
        System.out.printf("Text: %s, confidence score: %f.%n", match.getText(), 
match.getConfidenceScore()));
});
Analyze healthcare entities
Custom entities recognition
Custom text classification