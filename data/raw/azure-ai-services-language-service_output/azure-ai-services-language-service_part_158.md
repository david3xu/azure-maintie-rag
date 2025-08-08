executed was successful or unsuccessful for the given document. It may optionally include
information about the document batch and how it was processed.
For large documents which take a long time to execute, these operations are implemented as
long-running operations
. Long-running operations consist of an initial request sent to the
service to start an operation, followed by polling the service at intervals to determine whether
the operation has completed or failed, and if it has succeeded, to get the result.
For long running operations in the Azure SDK, the client exposes a Start<operation-name>
method that returns an Operation<T>  or a PageableOperation<T> . You can use the extension
method WaitForCompletionAsync()  to wait for the operation to complete and obtain its result.
A sample code snippet is provided to illustrate using long-running operations below.
We guarantee that all client instance methods are thread-safe and independent of each other
(guideline
). This ensures that the recommendation of reusing client instances is always safe,
even across threads.
Client options
 | Accessing the response
 | Handling failures
 | Diagnostics
 | Mocking
 |
Client lifetime
The following section provides several code snippets using the client  created above, and
covers the main features present in this client library. Although most of the snippets below
make use of synchronous service calls, keep in mind that the Azure.AI.TextAnalytics  package
supports both synchronous and asynchronous APIs.
Detect Language
Analyze Sentiment
Extract Key Phrases
Recognize Named Entities
Recognize PII Entities
Long-Running Operations
Thread safety
Additional concepts
Examples
Sync examples
\nRecognize Linked Entities
Detect Language Asynchronously
Recognize Named Entities Asynchronously
Analyze Healthcare Entities Asynchronously
Run multiple actions Asynchronously
Run a predictive model to determine the language that the passed-in document or batch of
documents are written in.
C#
For samples on using the production recommended option DetectLanguageBatch  see here
.
Please refer to the service documentation for a conceptual discussion of language detection.
Async examples
Detect Language
string document =
    "Este documento está escrito en un lenguaje diferente al inglés. Su objectivo 
es demostrar cómo"
    + " invocar el método de Detección de Lenguaje del servicio de Text Analytics 
en Microsoft Azure."
    + " También muestra cómo acceder a la información retornada por el servicio. 
Esta funcionalidad es"
    + " útil para los sistemas de contenido que recopilan texto arbitrario, donde 
el lenguaje no se conoce"
    + " de antemano. Puede usarse para detectar una amplia gama de lenguajes, 
variantes, dialectos y"
    + " algunos idiomas regionales o culturales.";
try
{
    Response<DetectedLanguage> response = client.DetectLanguage(document);
    DetectedLanguage language = response.Value;
    Console.WriteLine($"Detected language is {language.Name} with a confidence 
score of {language.ConfidenceScore}.");
}
catch (RequestFailedException exception)
{
    Console.WriteLine($"Error Code: {exception.ErrorCode}");
    Console.WriteLine($"Message: {exception.Message}");
}
\nRun a predictive model to determine the positive, negative, neutral or mixed sentiment
contained in the passed-in document or batch of documents.
C#
For samples on using the production recommended option AnalyzeSentimentBatch  see here
.
To get more granular information about the opinions related to targets of a product/service,
also known as Aspect-based Sentiment Analysis in Natural Language Processing (NLP), see a
sample on sentiment analysis with opinion mining here
.
Please refer to the service documentation for a conceptual discussion of sentiment analysis.
Run a model to identify a collection of significant phrases found in the passed-in document or
batch of documents.
C#
Analyze Sentiment
string document =
    "I had the best day of my life. I decided to go sky-diving and it made me 
appreciate my whole life so"
    + "much more. I developed a deep-connection with my instructor as well, and I 
feel as if I've made a"
    + "life-long friend in her.";
try
{
    Response<DocumentSentiment> response = client.AnalyzeSentiment(document);
    DocumentSentiment docSentiment = response.Value;
    Console.WriteLine($"Document sentiment is {docSentiment.Sentiment} with: ");
    Console.WriteLine($"  Positive confidence score: 
{docSentiment.ConfidenceScores.Positive}");
    Console.WriteLine($"  Neutral confidence score: 
{docSentiment.ConfidenceScores.Neutral}");
    Console.WriteLine($"  Negative confidence score: 
{docSentiment.ConfidenceScores.Negative}");
}
catch (RequestFailedException exception)
{
    Console.WriteLine($"Error Code: {exception.ErrorCode}");
    Console.WriteLine($"Message: {exception.Message}");
}
Extract Key Phrases
\nFor samples on using the production recommended option ExtractKeyPhrasesBatch  see
here
.
Please refer to the service documentation for a conceptual discussion of key phrase extraction.
Run a predictive model to identify a collection of named entities in the passed-in document or
batch of documents and categorize those entities into categories such as person, location, or
organization. For more information on available categories, see Text Analytics Named Entity
Categories.
C#
string document =
    "My cat might need to see a veterinarian. It has been sneezing more than 
normal, and although my"
    + " little sister thinks it is funny, I am worried it has the cold that I got 
last week. We are going"
    + " to call tomorrow and try to schedule an appointment for this week. 
Hopefully it will be covered by"
    + " the cat's insurance. It might be good to not let it sleep in my room for a 
while.";
try
{
    Response<KeyPhraseCollection> response = client.ExtractKeyPhrases(document);
    KeyPhraseCollection keyPhrases = response.Value;
    Console.WriteLine($"Extracted {keyPhrases.Count} key phrases:");
    foreach (string keyPhrase in keyPhrases)
    {
        Console.WriteLine($"  {keyPhrase}");
    }
}
catch (RequestFailedException exception)
{
    Console.WriteLine($"Error Code: {exception.ErrorCode}");
    Console.WriteLine($"Message: {exception.Message}");
}
Recognize Named Entities
string document =
    "We love this trail and make the trip every year. The views are breathtaking 
and well worth the hike!"
    + " Yesterday was foggy though, so we missed the spectacular views. We tried 
again today and it was"
    + " amazing. Everyone in my family liked the trail although it was too 
challenging for the less"
    + " athletic among us. Not necessarily recommended for small children. A hotel 
\nFor samples on using the production recommended option RecognizeEntitiesBatch  see
here
.
Please refer to the service documentation for a conceptual discussion of named entity
recognition.
Run a predictive model to identify a collection of entities containing Personally Identifiable
Information found in the passed-in document or batch of documents, and categorize those
entities into categories such as US social security number, drivers license number, or credit card
number.
C#
close to the trail"
    + " offers services for childcare in case you want that.";
try
{
    Response<CategorizedEntityCollection> response = 
client.RecognizeEntities(document);
    CategorizedEntityCollection entitiesInDocument = response.Value;
    Console.WriteLine($"Recognized {entitiesInDocument.Count} entities:");
    foreach (CategorizedEntity entity in entitiesInDocument)
    {
        Console.WriteLine($"  Text: {entity.Text}");
        Console.WriteLine($"  Offset: {entity.Offset}");
        Console.WriteLine($"  Length: {entity.Length}");
        Console.WriteLine($"  Category: {entity.Category}");
        if (!string.IsNullOrEmpty(entity.SubCategory))
            Console.WriteLine($"  SubCategory: {entity.SubCategory}");
        Console.WriteLine($"  Confidence score: {entity.ConfidenceScore}");
        Console.WriteLine();
    }
}
catch (RequestFailedException exception)
{
    Console.WriteLine($"Error Code: {exception.ErrorCode}");
    Console.WriteLine($"Message: {exception.Message}");
}
Recognize PII Entities
string document =
    "Parker Doe has repaid all of their loans as of 2020-04-25. Their SSN is 859-
98-0987. To contact them,"
    + " use their phone number 800-102-1100. They are originally from Brazil and 
have document ID number"
    + " 998.214.865-68.";
\nFor samples on using the production recommended option RecognizePiiEntitiesBatch  see
here
.
Please refer to the service documentation for supported PII entity types.
Run a predictive model to identify a collection of entities found in the passed-in document or
batch of documents, and include information linking the entities to their corresponding entries
in a well-known knowledge base.
C#
try
{
    Response<PiiEntityCollection> response = 
client.RecognizePiiEntities(document);
    PiiEntityCollection entities = response.Value;
    Console.WriteLine($"Redacted Text: {entities.RedactedText}");
    Console.WriteLine();
    Console.WriteLine($"Recognized {entities.Count} PII entities:");
    foreach (PiiEntity entity in entities)
    {
        Console.WriteLine($"  Text: {entity.Text}");
        Console.WriteLine($"  Category: {entity.Category}");
        if (!string.IsNullOrEmpty(entity.SubCategory))
            Console.WriteLine($"  SubCategory: {entity.SubCategory}");
        Console.WriteLine($"  Confidence score: {entity.ConfidenceScore}");
        Console.WriteLine();
    }
}
catch (RequestFailedException exception)
{
    Console.WriteLine($"Error Code: {exception.ErrorCode}");
    Console.WriteLine($"Message: {exception.Message}");
}
Recognize Linked Entities
string document =
    "Microsoft was founded by Bill Gates with some friends he met at Harvard. One 
of his friends, Steve"
    + " Ballmer, eventually became CEO after Bill Gates as well. Steve Ballmer 
eventually stepped down as"
    + " CEO of Microsoft, and was succeeded by Satya Nadella. Microsoft originally 
moved its headquarters"
    + " to Bellevue, Washington in Januaray 1979, but is now headquartered in 
Redmond.";
try
{
\nFor samples on using the production recommended option RecognizeLinkedEntitiesBatch  see
here
.
Please refer to the service documentation for a conceptual discussion of entity linking.
Run a predictive model to determine the language that the passed-in document or batch of
documents are written in.
C#
    Response<LinkedEntityCollection> response = 
client.RecognizeLinkedEntities(document);
    LinkedEntityCollection linkedEntities = response.Value;
    Console.WriteLine($"Recognized {linkedEntities.Count} entities:");
    foreach (LinkedEntity linkedEntity in linkedEntities)
    {
        Console.WriteLine($"  Name: {linkedEntity.Name}");
        Console.WriteLine($"  Language: {linkedEntity.Language}");
        Console.WriteLine($"  Data Source: {linkedEntity.DataSource}");
        Console.WriteLine($"  URL: {linkedEntity.Url}");
        Console.WriteLine($"  Entity Id in Data Source: 
{linkedEntity.DataSourceEntityId}");
        foreach (LinkedEntityMatch match in linkedEntity.Matches)
        {
            Console.WriteLine($"    Match Text: {match.Text}");
            Console.WriteLine($"    Offset: {match.Offset}");
            Console.WriteLine($"    Length: {match.Length}");
            Console.WriteLine($"    Confidence score: {match.ConfidenceScore}");
        }
        Console.WriteLine();
    }
}
catch (RequestFailedException exception)
{
    Console.WriteLine($"Error Code: {exception.ErrorCode}");
    Console.WriteLine($"Message: {exception.Message}");
}
Detect Language Asynchronously
string document =
    "Este documento está escrito en un lenguaje diferente al inglés. Su objectivo 
es demostrar cómo"
    + " invocar el método de Detección de Lenguaje del servicio de Text Analytics 
en Microsoft Azure."
    + " También muestra cómo acceder a la información retornada por el servicio. 
Esta funcionalidad es"
    + " útil para los sistemas de contenido que recopilan texto arbitrario, donde 
el lenguaje no se conoce"
    + " de antemano. Puede usarse para detectar una amplia gama de lenguajes, 
\nRun a predictive model to identify a collection of named entities in the passed-in document or
batch of documents and categorize those entities into categories such as person, location, or
organization. For more information on available categories, see Text Analytics Named Entity
Categories.
C#
variantes, dialectos y"
    + " algunos idiomas regionales o culturales.";
try
{
    Response<DetectedLanguage> response = await 
client.DetectLanguageAsync(document);
    DetectedLanguage language = response.Value;
    Console.WriteLine($"Detected language is {language.Name} with a confidence 
score of {language.ConfidenceScore}.");
}
catch (RequestFailedException exception)
{
    Console.WriteLine($"Error Code: {exception.ErrorCode}");
    Console.WriteLine($"Message: {exception.Message}");
}
Recognize Named Entities Asynchronously
string document =
    "We love this trail and make the trip every year. The views are breathtaking 
and well worth the hike!"
    + " Yesterday was foggy though, so we missed the spectacular views. We tried 
again today and it was"
    + " amazing. Everyone in my family liked the trail although it was too 
challenging for the less"
    + " athletic among us. Not necessarily recommended for small children. A hotel 
close to the trail"
    + " offers services for childcare in case you want that.";
try
{
    Response<CategorizedEntityCollection> response = await 
client.RecognizeEntitiesAsync(document);
    CategorizedEntityCollection entitiesInDocument = response.Value;
    Console.WriteLine($"Recognized {entitiesInDocument.Count} entities:");
    foreach (CategorizedEntity entity in entitiesInDocument)
    {
        Console.WriteLine($"  Text: {entity.Text}");
        Console.WriteLine($"  Offset: {entity.Offset}");
        Console.WriteLine($"  Length: {entity.Length}");
        Console.WriteLine($"  Category: {entity.Category}");
\nText Analytics for health is a containerized service that extracts and labels relevant medical
information from unstructured texts such as doctor's notes, discharge summaries, clinical
documents, and electronic health records. For more information see How to: Use Text Analytics
for health.
C#
        if (!string.IsNullOrEmpty(entity.SubCategory))
            Console.WriteLine($"  SubCategory: {entity.SubCategory}");
        Console.WriteLine($"  Confidence score: {entity.ConfidenceScore}");
        Console.WriteLine();
    }
}
catch (RequestFailedException exception)
{
    Console.WriteLine($"Error Code: {exception.ErrorCode}");
    Console.WriteLine($"Message: {exception.Message}");
}
Analyze Healthcare Entities Asynchronously
string documentA =
    "RECORD #333582770390100 | MH | 85986313 | | 054351 | 2/14/2001 12:00:00 AM |"
    + " CORONARY ARTERY DISEASE | Signed | DIS |"
    + Environment.NewLine
    + " Admission Date: 5/22/2001 Report Status: Signed Discharge Date: 4/24/2001"
    + " ADMISSION DIAGNOSIS: CORONARY ARTERY DISEASE."
    + Environment.NewLine
    + " HISTORY OF PRESENT ILLNESS: The patient is a 54-year-old gentleman with a 
history of progressive"
    + " angina over the past several months. The patient had a cardiac 
catheterization in July of this"
    + " year revealing total occlusion of the RCA and 50% left main disease, with 
a strong family history"
    + " of coronary artery disease with a brother dying at the age of 52 from a 
myocardial infarction and"
    + " another brother who is status post coronary artery bypass grafting. The 
patient had a stress"
    + " echocardiogram done on July, 2001, which showed no wall motion 
abnormalities, but this was a"
    + " difficult study due to body habitus. The patient went for six minutes with 
minimal ST depressions"
    + " in the anterior lateral leads, thought due to fatigue and wrist pain, his 
anginal equivalent. Due"
    + " to the patient'sincreased symptoms and family history and history left 
main disease with total"
    + " occasional of his RCA was referred for revascularization with open heart 
surgery.";
string documentB = "Prescribed 100mg ibuprofen, taken twice daily.";
\n// Prepare the input of the text analysis operation. You can add multiple 
documents to this list and
// perform the same operation on all of them simultaneously.
List<string> batchedDocuments = new()
{
    documentA,
    documentB
};
// Perform the text analysis operation.
AnalyzeHealthcareEntitiesOperation operation = await 
client.AnalyzeHealthcareEntitiesAsync(WaitUntil.Completed, batchedDocuments);
Console.WriteLine($"The operation has completed.");
Console.WriteLine();
// View the operation status.
Console.WriteLine($"Created On   : {operation.CreatedOn}");
Console.WriteLine($"Expires On   : {operation.ExpiresOn}");
Console.WriteLine($"Id           : {operation.Id}");
Console.WriteLine($"Status       : {operation.Status}");
Console.WriteLine($"Last Modified: {operation.LastModified}");
Console.WriteLine();
// View the operation results.
await foreach (AnalyzeHealthcareEntitiesResultCollection documentsInPage in 
operation.Value)
{
    Console.WriteLine($"Analyze Healthcare Entities, model version: \"
{documentsInPage.ModelVersion}\"");
    Console.WriteLine();
    foreach (AnalyzeHealthcareEntitiesResult documentResult in documentsInPage)
    {
        if (documentResult.HasError)
        {
            Console.WriteLine($"  Error!");
            Console.WriteLine($"  Document error code: 
{documentResult.Error.ErrorCode}");
            Console.WriteLine($"  Message: {documentResult.Error.Message}");
            continue;
        }
        Console.WriteLine($"  Recognized the following 
{documentResult.Entities.Count} healthcare entities:");
        Console.WriteLine();
        // View the healthcare entities that were recognized.
        foreach (HealthcareEntity entity in documentResult.Entities)
        {
            Console.WriteLine($"  Entity: {entity.Text}");
            Console.WriteLine($"  Category: {entity.Category}");
            Console.WriteLine($"  Offset: {entity.Offset}");
            Console.WriteLine($"  Length: {entity.Length}");
            Console.WriteLine($"  NormalizedText: {entity.NormalizedText}");