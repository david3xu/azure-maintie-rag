Please take a look at the samples
 directory for detailed examples on how to use this
library.
If you'd like to contribute to this library, please read the contributing guide
 to learn
more about how to build and test the code.
Microsoft Azure SDK for JavaScript
Next steps
Contributing
Related projects
\n![Image](images/page1751_image1.png)
\nAzure AI services support and help options
Article • 05/02/2025
Here are the options for getting support, staying up to date, giving feedback, and reporting
bugs for Azure AI services.
In the Azure portal, you can find answers to common AI service issues.
1. Go to your Azure AI services resource in the Azure portal. You can find it on the list on
this page: Azure AI services
. If you're a United States government customer, use the
Azure portal for the United States government
.
2. In the left pane, under Help, select Support + Troubleshooting.
3. Describe your issue in the text box, and answer the remaining questions in the form.
4. You'll find Learn articles and other resources that might help you resolve your issue.
Explore the range of Azure support options and choose the plan
 that best fits, whether
you're a developer just starting your cloud journey or a large organization deploying business-
critical, strategic applications. Azure customers can create and manage support requests in the
Azure portal.
To submit a support request for Azure AI services, follow the instructions on the New support
request
 page in the Azure portal. After choosing your Issue type, select Cognitive Services in
the Service type dropdown field.
For quick and reliable answers on your technical product questions from Microsoft Engineers,
Azure Most Valuable Professionals (MVPs), or our expert community, engage with us on
Microsoft Q&A, Azure's preferred destination for community support.
If you can't find an answer to your problem using search, submit a new question to Microsoft
Q&A. Use one of the following tags when you ask your question:
Azure AI services
Azure OpenAI
Get solutions to common issues
Create an Azure support request
Post a question on Microsoft Q&A
\nAzure OpenAI
Vision
Azure AI Vision
Azure AI Custom Vision
Azure Face
Azure AI Document Intelligence
Video Indexer
Language
Azure AI Immersive Reader
Language Understanding (LUIS)
Azure QnA Maker
Azure AI Language
Azure Translator
Speech
Azure AI Speech
Decision
Azure AI Anomaly Detector
Content Moderator
Azure AI Metrics Advisor
Azure AI Personalizer
For answers to your developer questions from the largest community developer ecosystem, ask
your question on Stack Overflow.
If you submit a new question to Stack Overflow, use one or more of the following tags when
you create the question:
Azure AI services
Azure OpenAI
Azure OpenAI
Vision
Post a question to Stack Overflow
\nAzure AI Vision
Azure AI Custom Vision
Azure Face
Azure AI Document Intelligence
Video Indexer
Language
Azure AI Immersive Reader
Language Understanding (LUIS)
Azure QnA Maker
Azure AI Language service
Azure Translator
Speech
Azure AI Speech service
Decision
Azure AI Anomaly Detector
Content Moderator
Azure AI Metrics Advisor
Azure AI Personalizer
To request new features, post them on https://feedback.azure.com . Share your ideas for
making Azure AI services and its APIs work better for the applications you develop.
Azure AI services
Vision
Azure AI Vision
Azure AI Custom Vision
Azure Face
Azure AI Document Intelligence
Video Indexer
Language
Azure AI Immersive Reader
Language Understanding (LUIS)
Submit feedback
\nAzure QnA Maker
Azure AI Language
Azure Translator
Speech
Azure AI Speech service
Decision
Azure AI Anomaly Detector
Content Moderator
Azure AI Metrics Advisor
Azure AI Personalizer
You can learn about the features in a new release or get the latest news on the Azure blog.
Staying informed can help you find the difference between a programming error, a service bug,
or a feature not yet available in Azure AI services.
Learn more about product updates, roadmap, and announcements in Azure Updates
.
News about Azure AI services is shared in the Azure AI blog
.
Join the conversation on Reddit
 about Azure AI services.
Stay informed
Next step
What are Azure AI services?
\nPrevious updates for Azure AI Language
06/30/2025
This article contains a list of previously recorded updates for Azure AI Language. For more
current service updates, see What's new.
Quality improvements for the extractive summarization feature in model-version 2021-08-
01 .
Starting with version 3.0.017010001-onprem-amd64  The text analytics for health container
can now be called using the client library.
General availability for text analytics for health containers and API.
General availability for opinion mining.
General availability for PII extraction and redaction.
General availability for asynchronous operation.
New model-version 2021-06-01  for key phrase extraction based on transformers. It
provides:
Support for 10 languages (Latin and CJK).
Improved key phrase extraction.
The 2021-06-01  model version for Named Entity Recognition (NER) which provides
Improved AI quality and expanded language support for the Skill entity category.
Added Spanish, French, German, Italian and Portuguese language support for the Skill
entity category
October 2021
September 2021
July 2021
June 2021
General API updates
Text Analytics for health updates
\nA new model version 2021-05-15  for the /health  endpoint and on-premises container
which provides
5 new entity types: ALLERGEN , CONDITION_SCALE , COURSE , EXPRESSION  and
MUTATION_TYPE ,
14 new relation types,
Assertion detection expanded for new entity types and
Linking support for ALLERGEN  entity type
A new image for the Text Analytics for health container with tag 3.0.016230002-onprem-
amd64  and model version 2021-05-15 . This container is available for download from
Microsoft Container Registry.
Custom question answering (previously QnA maker) can now be accessed using a Text
Analytics resource.
Preview API release, including:
Asynchronous API now supports sentiment analysis and opinion mining.
A new query parameter, LoggingOptOut , is now available for customers who wish to opt
out of logging input text for incident reports.
Text analytics for health and asynchronous operations are now available in all regions.
Changes in the opinion mining JSON response body:
aspects  is now targets  and opinions  is now assessments .
Changes in the JSON response body of the hosted web API of text analytics for health:
The isNegated  boolean name of a detected entity object for negation is deprecated
and replaced by assertion detection.
A new property called role  is now part of the extracted relation between an attribute
and an entity as well as the relation between entities. This adds specificity to the
detected relation type.
Entity linking is now available as an asynchronous task.
A new pii-categories  parameter for the PII feature.
This parameter lets you specify select PII entities, as well as those not supported by
default for the input language.
May 2021
March 2021
\nUpdated client libraries, which include asynchronous and text analytics for health
operations.
A new model version 2021-03-01  for text analytics for health API and on-premises
container which provides:
A rename of the Gene  entity type to GeneOrProtein .
A new Date  entity type.
Assertion detection which replaces negation detection.
A new preferred name  property for linked entities that is normalized from various
ontologies and coding systems.
A new text analytics for health container image with tag 3.0.015490002-onprem-amd64  and
the new model-version 2021-03-01  has been released to the container preview repository.
This container image will no longer be available for download from
containerpreview.azurecr.io  after April 26th, 2021.
Processed Text Records is now available as a metric in the Monitoring section for your
text analytics resource in the Azure portal.
The 2021-01-15  model version for the PII feature, which provides:
Expanded support for 9 new languages
Improved AI quality
The S0 through S4 pricing tiers are being retired on March 8th, 2021.
The language detection container is now generally available.
The 2021-01-15  model version for Named Entity Recognition (NER), which provides
Expanded language support.
Improved AI quality of general entity categories for all supported languages.
The 2021-01-05  model version for language detection, which provides additional
language support.
Portuguese (Brazil) pt-BR  is now supported in sentiment analysis, starting with model
version 2020-04-01 . It adds to the existing pt-PT  support for Portuguese.
February 2021
January 2021
November 2020
\nUpdated client libraries, which include asynchronous and text analytics for health
operations.
Hindi support for sentiment analysis, starting with model version 2020-04-01 .
Model version 2020-09-01  for language detection, which adds additional language
support and accuracy improvements.
PII now includes the new redactedText  property in the response JSON where detected PII
entities in the input text are replaced by an *  for each character of those entities.
Entity linking endpoint now includes the bingID  property in the response JSON for linked
entities.
The following updates are specific to the September release of the text analytics for
health container only.
A new container image with tag 1.1.013530001-amd64-preview  with the new model-
version 2020-09-03  has been released to the container preview repository.
This model version provides improvements in entity recognition, abbreviation
detection, and latency enhancements.
Model version 2020-07-01  for key phrase extraction, PII detection, and language
detection. This update adds:
Additional government and country/region specific entity categories for Named Entity
Recognition.
Norwegian and Turkish support in Sentiment Analysis.
An HTTP 400 error will now be returned for API requests that exceed the published data
limits.
Endpoints that return an offset now support the optional stringIndexType  parameter,
which adjusts the returned offset  and length  values to match a supported string index
scheme.
The following updates are specific to the August release of the Text Analytics for health
container only.
New model-version for Text Analytics for health: 2020-07-24
October 2020
September 2020
August 2020
\nThe following properties in the JSON response have changed:
type  has been renamed to category
score  has been renamed to confidenceScore
Entities in the category  field of the JSON output are now in pascal case. The following
entities have been renamed:
EXAMINATION_RELATION  has been renamed to RelationalOperator .
EXAMINATION_UNIT  has been renamed to MeasurementUnit .
EXAMINATION_VALUE  has been renamed to MeasurementValue .
ROUTE_OR_MODE  has been renamed MedicationRoute .
The relational entity ROUTE_OR_MODE_OF_MEDICATION  has been renamed to
RouteOfMedication .
The following entities have been added:
Named Entity Recognition
AdministrativeEvent
CareEnvironment
HealthcareProfession
MedicationForm
Relation extraction
DirectionOfCondition
DirectionOfExamination
DirectionOfTreatment
Model version 2020-04-01 :
Updated language support for sentiment analysis
New "Address" entity category in Named Entity Recognition (NER)
New subcategories in NER:
Location - Geographical
Location - Structural
Organization - Stock Exchange
Organization - Medical
Organization - Sports
Event - Cultural
Event - Natural
Event - Sports
May 2020