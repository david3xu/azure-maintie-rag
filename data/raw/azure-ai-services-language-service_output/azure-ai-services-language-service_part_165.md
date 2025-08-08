Register a new Azure Active Directory application
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
\nyou to build custom models for text classification tasks.
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
\n![Image](images/page1649_image1.png)
\nAzure Text Analysis client library for
JavaScript - version 1.1.0
Article • 06/20/2023
Azure Cognitive Service for Language
 is a cloud-based service that provides advanced
natural language processing over raw text, and includes the following main features:
Note: This SDK targets Azure Cognitive Service for Language API version 2023-04-01.
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
Support Multiple Actions Per Document
Use the client library to:
Detect what language input text is written in.
Determine what customers think of your brand or topic by analyzing raw text for clues
about positive or negative sentiment.
Automatically extract key phrases to quickly identify the main points.
Identify and categorize entities in your text as people, places, organizations, date/time,
quantities, percentages, currencies, healthcare specific, and more.
Perform multiple of the above tasks at once.
Key links:
Source code
Package (NPM)
API reference documentation
Product documentation
Samples
Migrating from @azure/ai-text-analytics advisory ⚠️