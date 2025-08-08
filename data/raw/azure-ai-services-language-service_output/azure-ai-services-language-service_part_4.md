What's new in Azure AI Language?
06/02/2025
Azure AI Language is updated on an ongoing basis. Bookmark this page to stay up to date with
release notes, feature enhancements, and our newest documentation.
The latest API preview version includes updates for named entity recognition (NER) and PII
detection:
New entity type support for DateOfBirth , BankAccountNumber , PassportNumber , and
DriversLicenseNumber .
Improved AI quality for PhoneNumber  entity type.
Azure AI Language now supports the following agent templates:
Intent routing: Detects user intent and provides precise answers, ideal for deterministic
intent routing, and exact question answering with human oversight.
Exact question answering: Delivers consistent, accurate responses to high-value
predefined questions through deterministic methods.
Azure AI Language introduces new customization and entity subtype features for PII detection:
Customize PII detection using your own regex (Text PII container only).
Specify values to exclude from PII output.
Use entity synonyms for tailored PII detection.
Azure AI Foundry now offers enhanced capabilities for fine-tuning with custom conversational
language understanding (CLU) and conversational question-and-answer (CQA) AI features:
May 2025
2025-05-15-preview release
New agent templates
PII detection enhancements
Enhanced CLU and CQA Capabilities in Azure AI Foundry
\nCLU and CQA authoring tools are now available in Azure AI Foundry.
CLU offers a quick deploy option powered by large language models (LLMs) for rapid
deployment.
CQA incorporates the QnA Maker scoring algorithm for more accurate responses.
CQA enables exact match answering for precise query resolutions.
Updated and improved model GA release for NER
Expanded context window for PII redaction – This service update expands the window of
detection the PII redaction service considers, enhancing quality and accuracy.
Added prediction capability for custom models, including conversational language
Understanding (CLU), custom named entity recognition (NER), and custom text
classification, are now available in three new regions: Jio India Central, UK West, and
Canada East.
Scanned PDF PII - Document input for PII redaction now supports scanned PDFs, enabling
PII detection and redaction in both digital and nondigital documents using OCR .
Azure AI Language resource now can be deployed to three new regions, Jio India Central,
UK West, and Canada East, for the following capabilities:
Language detection
Sentiment analysis
Key phrase extraction
Named entity recognition (NER)
Personally identifiable information (PII) entity recognition
Entity linking
Text analytics for health
Extractive text summarization
Back-end infrastructure for the Named entity recognition (NER) and Text Personally
identifiable information (PII) entity recognition models is now updated with extended
context window limits.
For more updates, see our latest TechCommunity Blog Post
.
April 2025
March 2025
\nOur Conversational PII redaction service is now powered by an upgraded GA model. This
revised version enhances the quality and accuracy of Credit Card Number entities and
Numeric Identification entities. These entities include Social Security numbers, Driver's
license numbers, Policy numbers, Medicare Beneficiary Identifiers, and Financial account
numbers.
Document and text abstractive summarization is now powered by fine-tuned Phi-3.5-
mini! Check out the Announcing Blog
 for more information.
More skills are available in Azure AI Foundry
: Extract key phrase, Extract named entities,
Analyze sentiment, and Detect language. More skills are yet to come.
.NET SDK for Azure AI Language text analytics, Azure.AI.Language.Text 1.0.0-beta.2
, is
now available. This client library supports the latest REST API version, 2024-11-01 , and
2024-11-15-preview , for the following features:
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
Custom sentiment analysis (preview), custom text analytics for health (preview) and
custom summarization (preview) were retired on January 10, 2025, as Azure AI features
are constantly evaluated based on customer demand and feedback. Based on the
customers' feedback of these preview features, Microsoft is retiring this feature and
prioritize new custom model features using the power of generative AI to better serve
customers' needs.
February 2025
January 2025
November 2024
\nAzure AI Language is moving to Azure AI Foundry
. These skills are now available in AI
Foundry playground: Extract health information, Extract PII from conversation, Extract PII
from text, Summarize text, Summarize conversation, Summarize for call center. More skills
follow.
Runtime Container for Conversational Language Understanding (CLU) is available for on-
premises connections.
Both our Text PII redaction service and our Conversational PII service preview API (version
2024-11-15-preview) now support the option to mask detected sensitive entities with a
label beyond just redaction characters. Customers can specify if personal data content
such as names and phone numbers, that is, "John Doe received a call from 424-878-9192"
are masked with a redaction character, that is, "******** received a call from ************"
or masked with an entity label, that is, " PERSON_1  received a call from PHONENUMBER_1 ."
More on how to specify the redaction policy style for your outputs can be found in our
how-to guides.
Native document support gating is removed with the latest API version, 2024-11-15-
preview, allowing customers to access native document support for PII redaction and
summarization. Key updates in this version include:
Increased Maximum File Size Limits (from 1 MB to 10 MB).
Enhanced PII Redaction Customization: Customers can now specify whether they want
only the redacted document or both the redacted document and a JSON file
containing the detected entities.
Language detection is a built-in feature designed to identify the language in which a
document is written. It provides a language code that corresponds to a wide array of
languages. This feature includes not only standard languages but also their variants,
dialects, and certain regional or cultural languages. Today the general availability of script
detection capability, and 16 more languages support, which adds up to 139 total
supported languages is announced.
Named Entity Recognition service, Entity Resolution was upgraded to the Entity Metadata
starting in API version 2023-04-15-preview. If you're calling the preview version of the API
equal or newer than 2023-04-15-preview, check out the Entity Metadata article to use the
resolution feature. The service now supports the ability to specify a list of entity tags to be
included into the response or excluded from the response. If a piece of text is classified as
more than one entity type, the overlapPolicy parameter allows customers to specify how
the service handles the overlap. The inferenceOptions  parameter enables users to modify
the inference process, such as preventing detected entity values from being normalized
and added to the metadata. Along with these optional input parameters, we support an
updated output structure (with new fields tags, type, and metadata) to ensure enhanced
user customization and deeper analysis Learn more on our documentation.
Text Analytics for Health (TA4H) is a specialized tool designed to extract and categorize
key medical details from unstructured sources. These sources include doctor's notes,
\ndischarge summaries, clinical documentation, and electronic health records. Today, we
released support for Fast Healthcare Interoperability Resources (FHIR) structuring and
temporal assertion detection in the Generally Available API.
Custom language service features enable you to deploy your project to multiple
resources within a single region via the API.
PII detection now has container support. See more details in the Azure Update post:
Announcing Text PII Redaction Container Release
.
Custom sentiment analysis (preview) will be retired January 10, 2025. You can transition to
other custom model training services, such as custom text classification in Azure AI
Language.  See more details in the Azure Update post: Retirement: Announcing upcoming
retirement of custom sentiment analysis (preview) in Azure AI Language
(microsoft.com)
.
Custom text analytics for health (preview) will be retired on January 10, 2025. Transition to
other custom model training services, such as custom named entity recognition in Azure
AI Language, by that date.  See more details in the Azure Update post: Retirement:
Announcing upcoming retirement of custom text analytics for health (preview) in Azure AI
Language (microsoft.com)
.
CLU utterance limit in a project increased from 25,000 to 50,000.
CLU new version of training configuration, version 2024-08-01-preview, is available now,
which improves the quality of intent identification for out of domain utterances.
Conversational PII redaction
 service in English-language contexts is now Generally
Available (GA).
Conversation Summarization now supports 12 added languages in preview as listed here.
Summarization Meeting or Conversation Chapter titles features support reduced length
to focus on the key topics.
Enable support for data augmentation for diacritics to generate variations of training data
for diacritic variations used in some natural languages which are especially useful for
October 2024
September 2024
August 2024
July 2024
\nGermanic and Slavic languages.
Expanded language detection support for added scripts according to the ISO 15924
standard
 is now available starting in API version 2023-11-15-preview .
Native document support is now available in 2023-11-15-preview  public preview.
Text Analytics for health new model 2023-12-01  is now available.
New Relation Type: BodySiteOfExamination
Quality enhancements to support radiology documents
Significant latency improvements
Various bug fixes: Improvements across NER, Entity Linking, Relations, and Assertion
Detection
Named Entity Recognition Container is now Generally Available (GA).
Custom sentiment analysis is now available in preview.
Custom Named Entity Recognition (NER) Docker containers are now available for on-
premises deployment.
Custom Text analytics for health is available in public preview, which enables you to build
custom AI models to extract healthcare specific entities from unstructured text
February 2024
January 2024
December 2023
November 2023
July 2023
May 2023
April 2023
\nYou can now use Azure OpenAI to automatically label or generate data during authoring.
Learn more with the following links:
Autolabel your documents in Custom text classification or Custom named entity
recognition.
Generate suggested utterances in Conversational language understanding.
The latest model version ( 2022-10-01 ) for Language Detection now supports 6 more
International languages and 12 Romanized Indic languages.
New model version ('2023-01-01-preview') for Personally Identifiable Information (PII)
detection with quality updates and new language support
New versions of the text analysis client library are available in preview:
Package (NuGet)
Changelog/Release History
ReadMe
Samples
Conversational language understanding and orchestration workflow now available in the
following regions in the sovereign cloud for China:
China East 2 (Authoring and Prediction)
China North 2 (Prediction)
New model evaluation updates for Conversational language understanding and
Orchestration workflow.
New model version ('2023-01-01-preview') for Text Analytics for health featuring new
entity categories for social determinants of health.
New model version ('2023-02-01-preview') for named entity recognition features
improved accuracy and more language support with up to 79 languages.
March 2023
C#
February 2023
December 2022
\nNew version (v5.2.0-beta.1) of the text analysis client library is available in preview for
C#/.NET:
Package (NuGet)
Changelog/Release History
ReadMe
Samples
New model version ( 2022-10-01 ) released for Language Detection. The new model
version comes with improvements in language detection quality on short texts.
Expanded language support for:
Opinion mining
Conversational PII now supports up to 40,000 characters as document size.
New versions of the text analysis client library are available in preview:
Java
Package (Maven)
Changelog/Release History
ReadMe
Samples
JavaScript
Package (npm)
Changelog/Release History
ReadMe
Samples
Python
Package (PyPi)
Changelog/Release History
ReadMe
Samples
The summarization feature now has the following capabilities:
Document summarization:
Abstractive summarization, which generates a summary of a document that can't
use the same words as presented in the document, but captures the main idea.
November 2022
October 2022
\nConversation summarization
Chapter title summarization, which returns suggested chapter titles of input
conversations.
Narrative summarization, which returns call notes, meeting notes or chat summaries
of input conversations.
Expanded language support for:
Sentiment analysis
Key phrase extraction
Named entity recognition
Text Analytics for health
Multi-region deployment and project asset versioning for:
Conversational language understanding
Orchestration workflow
Custom text classification
Custom named entity recognition
Regular expressions in conversational language understanding and required components,
offering an added ability to influence entity predictions.
Entity resolution in named entity recognition
New region support for:
Conversational language understanding
Orchestration workflow
Custom text classification
Custom named entity recognition
Document type as an input supported for Text Analytics for health FHIR requests
Conversational language understanding is available in the following regions:
Central India
Switzerland North
West US 2
Text Analytics for Health now supports more languages in preview: Spanish, French,
German Italian, Portuguese and Hebrew. These languages are available when using a
docker container to deploy the API service.
The Azure.AI.TextAnalytics client library v5.2.0 are generally available and ready for use in
production applications. For more information on Language service client libraries, see
the Developer overview.
Java
Package (Maven)
Changelog/Release History
September 2022
\nReadMe
Samples
Python
Package (PyPi)
Changelog/Release History
ReadMe
Samples
C#/.NET
Package (NuGet)
Changelog/Release History
ReadMe
Samples
Role-based access control for the Language service.
New AI models for sentiment analysis and key phrase extraction based on z-code
models
, providing:
Performance and quality improvements for the following 11 languages supported by
sentiment analysis: ar , da , el , fi , hi , nl , no , pl , ru , sv , tr
Performance and quality improvements for the following 20 languages supported by
key phrase extraction: af , bg , ca , hr , da , nl , et , fi , el , hu , id , lv , no , pl , ro , ru ,
sk , sl , sv , tr
Conversational PII is now available in all Azure regions supported by the Language
service.
A new version of the Language API ( 2022-07-01-preview ) is available. It provides:
Automatic language detection for asynchronous tasks.
Text Analytics for health confidence scores are now returned in relations.
To use this version in your REST API calls, use the following URL:
HTTP
August 2022
July 2022
<your-language-resource-endpoint>/language/:analyze-text?api-version=2022-07-
01-preview