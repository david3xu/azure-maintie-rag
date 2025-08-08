Console.WriteLine($"  Links:");
            // View the entity data sources.
            foreach (EntityDataSource entityDataSource in entity.DataSources)
            {
                Console.WriteLine($"    Entity ID in Data Source: 
{entityDataSource.EntityId}");
                Console.WriteLine($"    DataSource: {entityDataSource.Name}");
            }
            // View the entity assertions.
            if (entity.Assertion is not null)
            {
                Console.WriteLine($"  Assertions:");
                if (entity.Assertion?.Association is not null)
                {
                    Console.WriteLine($"    Association: 
{entity.Assertion?.Association}");
                }
                if (entity.Assertion?.Certainty is not null)
                {
                    Console.WriteLine($"    Certainty: 
{entity.Assertion?.Certainty}");
                }
                if (entity.Assertion?.Conditionality is not null)
                {
                    Console.WriteLine($"    Conditionality: 
{entity.Assertion?.Conditionality}");
                }
            }
        }
        Console.WriteLine($"  We found {documentResult.EntityRelations.Count} 
relations in the current document:");
        Console.WriteLine();
        // View the healthcare entity relations that were recognized.
        foreach (HealthcareEntityRelation relation in 
documentResult.EntityRelations)
        {
            Console.WriteLine($"    Relation: {relation.RelationType}");
            if (relation.ConfidenceScore is not null)
            {
                Console.WriteLine($"    ConfidenceScore: 
{relation.ConfidenceScore}");
            }
            Console.WriteLine($"    For this relation there are 
{relation.Roles.Count} roles");
            // View the relation roles.
            foreach (HealthcareEntityRelationRole role in relation.Roles)
            {
\nThis functionality allows running multiple actions in one or more documents. Actions include:
Named Entities Recognition
PII Entities Recognition
Linked Entity Recognition
Key Phrase Extraction
Sentiment Analysis
Healthcare Entities Recognition (see sample here
)
Custom Named Entities Recognition (see sample here
)
Custom Single Label Classification (see sample here
)
Custom Multi Label Classification (see sample here
)
C#
                Console.WriteLine($"      Role Name: {role.Name}");
                Console.WriteLine($"      Associated Entity Text: 
{role.Entity.Text}");
                Console.WriteLine($"      Associated Entity Category: 
{role.Entity.Category}");
                Console.WriteLine();
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }
}
Run multiple actions Asynchronously
    string documentA =
        "We love this trail and make the trip every year. The views are 
breathtaking and well worth the hike!"
        + " Yesterday was foggy though, so we missed the spectacular views. We 
tried again today and it was"
        + " amazing. Everyone in my family liked the trail although it was too 
challenging for the less"
        + " athletic among us.";
    string documentB =
        "Last week we stayed at Hotel Foo to celebrate our anniversary. The staff 
knew about our anniversary"
        + " so they helped me organize a little surprise for my partner. The room 
was clean and with the"
        + " decoration I requested. It was perfect!";
    // Prepare the input of the text analysis operation. You can add multiple 
\ndocuments to this list and
    // perform the same operation on all of them simultaneously.
    List<string> batchedDocuments = new()
    {
        documentA,
        documentB
    };
    TextAnalyticsActions actions = new()
    {
        ExtractKeyPhrasesActions = new List<ExtractKeyPhrasesAction>() { new 
ExtractKeyPhrasesAction() { ActionName = "ExtractKeyPhrasesSample" } },
        RecognizeEntitiesActions = new List<RecognizeEntitiesAction>() { new 
RecognizeEntitiesAction() { ActionName = "RecognizeEntitiesSample" } },
        DisplayName = "AnalyzeOperationSample"
    };
    // Perform the text analysis operation.
    AnalyzeActionsOperation operation = await 
client.AnalyzeActionsAsync(WaitUntil.Completed, batchedDocuments, actions);
    // View the operation status.
    Console.WriteLine($"Created On   : {operation.CreatedOn}");
    Console.WriteLine($"Expires On   : {operation.ExpiresOn}");
    Console.WriteLine($"Id           : {operation.Id}");
    Console.WriteLine($"Status       : {operation.Status}");
    Console.WriteLine($"Last Modified: {operation.LastModified}");
    Console.WriteLine();
    if (!string.IsNullOrEmpty(operation.DisplayName))
    {
        Console.WriteLine($"Display name: {operation.DisplayName}");
        Console.WriteLine();
    }
    Console.WriteLine($"Total actions: {operation.ActionsTotal}");
    Console.WriteLine($"  Succeeded actions: {operation.ActionsSucceeded}");
    Console.WriteLine($"  Failed actions: {operation.ActionsFailed}");
    Console.WriteLine($"  In progress actions: {operation.ActionsInProgress}");
    Console.WriteLine();
    await foreach (AnalyzeActionsResult documentsInPage in operation.Value)
    {
        IReadOnlyCollection<ExtractKeyPhrasesActionResult> keyPhrasesResults = 
documentsInPage.ExtractKeyPhrasesResults;
        IReadOnlyCollection<RecognizeEntitiesActionResult> entitiesResults = 
documentsInPage.RecognizeEntitiesResults;
        Console.WriteLine("Recognized Entities");
        int docNumber = 1;
        foreach (RecognizeEntitiesActionResult entitiesActionResults in 
entitiesResults)
        {
            Console.WriteLine($" Action name: 
{entitiesActionResults.ActionName}");
\n            Console.WriteLine();
            foreach (RecognizeEntitiesResult documentResult in 
entitiesActionResults.DocumentsResults)
            {
                Console.WriteLine($" Document #{docNumber++}");
                Console.WriteLine($"  Recognized {documentResult.Entities.Count} 
entities:");
                foreach (CategorizedEntity entity in documentResult.Entities)
                {
                    Console.WriteLine();
                    Console.WriteLine($"    Entity: {entity.Text}");
                    Console.WriteLine($"    Category: {entity.Category}");
                    Console.WriteLine($"    Offset: {entity.Offset}");
                    Console.WriteLine($"    Length: {entity.Length}");
                    Console.WriteLine($"    ConfidenceScore: 
{entity.ConfidenceScore}");
                    Console.WriteLine($"    SubCategory: {entity.SubCategory}");
                }
                Console.WriteLine();
            }
        }
        Console.WriteLine("Extracted Key Phrases");
        docNumber = 1;
        foreach (ExtractKeyPhrasesActionResult keyPhrasesActionResult in 
keyPhrasesResults)
        {
            Console.WriteLine($" Action name: 
{keyPhrasesActionResult.ActionName}");
            Console.WriteLine();
            foreach (ExtractKeyPhrasesResult documentResults in 
keyPhrasesActionResult.DocumentsResults)
            {
                Console.WriteLine($" Document #{docNumber++}");
                Console.WriteLine($"  Recognized the following 
{documentResults.KeyPhrases.Count} Keyphrases:");
                foreach (string keyphrase in documentResults.KeyPhrases)
                {
                    Console.WriteLine($"    {keyphrase}");
                }
                Console.WriteLine();
            }
        }
    }
}
Troubleshooting
General
\nWhen you interact with the Cognitive Services for Language using the .NET Text Analytics SDK,
errors returned by the Language service correspond to the same HTTP status codes returned
for REST API requests.
For example, if you submit a batch of text document inputs containing duplicate document ids,
a 400  error is returned, indicating "Bad Request".
C#
You will notice that additional information is logged, like the client request ID of the operation.
text
The simplest way to see the logs is to enable the console logging. To create an Azure SDK log
listener that outputs messages to console use AzureEventSourceListener.CreateConsoleLogger
method.
C#
try
{
    DetectedLanguage result = client.DetectLanguage(document);
}
catch (RequestFailedException e)
{
    Console.WriteLine(e.ToString());
}
Message:
    Azure.RequestFailedException:
    Status: 400 (Bad Request)
Content:
    {"error":{"code":"InvalidRequest","innerError":
{"code":"InvalidDocument","message":"Request contains duplicated Ids. Make sure 
each document has a unique Id."},"message":"Invalid document in request."}}
Headers:
    Transfer-Encoding: chunked
    x-aml-ta-request-id: 146ca04a-af54-43d4-9872-01a004bee5f8
    X-Content-Type-Options: nosniff
    x-envoy-upstream-service-time: 6
    apim-request-id: c650acda-2b59-4ff7-b96a-e316442ea01b
    Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
    Date: Wed, 18 Dec 2019 16:24:52 GMT
    Content-Type: application/json; charset=utf-8
Setting up console logging
\nTo learn more about other logging mechanisms see here
.
Samples showing how to use this client library are available in this GitHub repository. Samples
are provided for each main functional area, and for each area, samples are provided for
analyzing a single document, and a collection of documents in both sync and async mode.
Detect Language
Analyze Sentiment
Extract Key Phrases
Recognize Named Entities
Recognize PII Entities
Recognize Linked Entities
Recognize Healthcare Entities
Custom Named Entities Recognition
Custom Single Label Classification
Custom Multi Label Classification
Extractive Summarization
Abstractive Summarization
Understand how to work with long-running operations
Running multiple actions in one or more documents
Analyze Sentiment with Opinion Mining
Mock a client for testing
 using the Moq
 library
See the CONTRIBUTING.md
 for details on building, testing, and contributing to this library.
This project welcomes contributions and suggestions. Most contributions require you to agree
to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do,
grant us the rights to use your contribution. For details, visit cla.microsoft.com
.
// Setup a listener to monitor logged events.
using AzureEventSourceListener listener = 
AzureEventSourceListener.CreateConsoleLogger();
Next steps
Advanced samples
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
\nAzure Cognitive Language Services
Question Answering client library for .NET -
version 1.1.0
Article • 10/18/2022
The Question Answering service is a cloud-based API service that lets you create a
conversational question-and-answer layer over your existing data. Use it to build a knowledge
base by extracting questions and answers from your semi-structured content, including FAQ,
manuals, and documents. Answer users’ questions with the best answers from the QnAs in your
knowledge base—automatically. Your knowledge base gets smarter, too, as it continually learns
from user behavior.
Source code
 | Package (NuGet)
 | API reference documentation | Product documentation |
Samples
 | Migration guide
Install the Azure Cognitive Language Services Question Answering client library for .NET with
NuGet
:
.NET CLI
An Azure subscription
An existing Question Answering resource
Though you can use this SDK to create and import conversation projects, Language Studio
 is
the preferred method for creating projects.
In order to interact with the Question Answering service, you'll need to either create an
instance of the QuestionAnsweringClient
 class for querying existing projects or an instance
of the QuestionAnsweringAuthoringClient
 for managing projects within your resource. You
Getting started
Install the package
dotnet add package Azure.AI.Language.QuestionAnswering
Prerequisites
Authenticate the client
\nwill need an endpoint, and an API key to instantiate a client object. For more information
regarding authenticating with Cognitive Services, see Authenticate requests to Azure Cognitive
Services.
You can get the endpoint and an API key from the Cognitive Services resource or Question
Answering resource in the Azure Portal
.
Alternatively, use the Azure CLI command shown below to get the API key from the Question
Answering resource.
PowerShell
To use the QuestionAnsweringClient , make sure you use the right namespaces:
C#
With your endpoint and API key you can instantiate a QuestionAnsweringClient :
C#
To use the QuestionAnsweringAuthoringClient , use the following namespace in addition to
those above, if needed.
C#
Get an API key
az cognitiveservices account keys list --resource-group <resource-group-name> --
name <resource-name>
Create a QuestionAnsweringClient
using Azure.Core;
using Azure.AI.Language.QuestionAnswering;
Uri endpoint = new Uri("https://myaccount.cognitiveservices.azure.com/");
AzureKeyCredential credential = new AzureKeyCredential("{api-key}");
QuestionAnsweringClient client = new QuestionAnsweringClient(endpoint, 
credential);
Create a QuestionAnsweringAuthoringClient
\nWith your endpoint and API key, you can instantiate a QuestionAnsweringAuthoringClient :
C#
You can also create a QuestionAnsweringClient  or QuestionAnsweringAuthoringClient  using
Azure Active Directory (AAD) authentication. Your user or service principal must be assigned
the "Cognitive Services Language Reader" role. Using the DefaultAzureCredential
 you can
authenticate a service using Managed Identity or a service principal, authenticate as a
developer working on an application, and more all without changing code.
Before you can use the DefaultAzureCredential , or any credential type from Azure.Identity
,
you'll first need to install the Azure.Identity package
.
To use DefaultAzureCredential  with a client ID and secret, you'll need to set the
AZURE_TENANT_ID , AZURE_CLIENT_ID , and AZURE_CLIENT_SECRET  environment variables;
alternatively, you can pass those values to the ClientSecretCredential  also in Azure.Identity.
Make sure you use the right namespace for DefaultAzureCredential  at the top of your source
file:
C#
Then you can create an instance of DefaultAzureCredential  and pass it to a new instance of
your client:
C#
using Azure.AI.Language.QuestionAnswering.Authoring;
Uri endpoint = new Uri("https://myaccount.cognitiveservices.azure.com/");
AzureKeyCredential credential = new AzureKeyCredential("{api-key}");
QuestionAnsweringAuthoringClient client = new 
QuestionAnsweringAuthoringClient(endpoint, credential);
Create a client using Azure Active Directory authentication
using Azure.Identity;
Uri endpoint = new Uri("https://myaccount.cognitiveservices.azure.com");
DefaultAzureCredential credential = new DefaultAzureCredential();