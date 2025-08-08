Model lifecycle
Article • 01/31/2025
Language service features utilize AI models. We update the language service with new
model versions to improve accuracy, support, and quality. As models become older,
they are retired. Use this article for information on that process, and what you can
expect for your applications.
Our standard (not customized) language service features are built on AI models that we
call pre-trained or prebuilt models.
We regularly update the language service with new model versions to improve model
accuracy, support, and quality.
By default, all API requests will use the latest Generally Available (GA) model.
We recommend using the latest  model version to utilize the latest and highest quality
models. As our models improve, it’s possible that some of your model results may
change. Model versions may be deprecated, so we no longer accept specified GA model
versions in your implementation.
Preview models used for preview features do not maintain a minimum retirement period
and may be deprecated at any time.
By default, API and SDK requests will use the latest Generally Available model. You can
use an optional parameter to select the version of the model to be used (not
recommended).
Use the table below to find which model versions are supported by each feature:
Prebuilt features
Choose the model-version used on your data
７ Note
If you are using a model version that is not listed in the table, then it was subjected
to the expiration policy.
ﾉ
Expand table
\nFeature
Supported generally
available (GA) version
Supported preview
versions
Sentiment Analysis and opinion
mining
latest*
Language Detection
latest*
Entity Linking
latest*
Named Entity Recognition (NER)
latest*
2024-04-15-preview**
Personally Identifiable
Information (PII) detection
latest*
2024-04-15-preview**
PII detection for conversations
latest*
2024-11-01-preview**
Question answering
latest*
Text Analytics for health
latest*
2022-08-15-preview , 2023-
01-01-preview**
Key phrase extraction
latest*
Summarization
latest*
* Latest Generally Available (GA) model version ** Latest preview version
For custom features, there are two key parts of the AI implementation: training and
deployment. New configurations are released regularly with regular AI improvements, so
older and less accurate configurations are retired.
Use the table below to find which model versions are supported by each feature:
Feature
Supported Training
Config Versions
Training Config
Expiration
Deployment
Expiration
Conversational language
understanding
2022-09-01  (latest)**
August 26, 2025
August 26, 2026
Orchestration workflow
2022-09-01  (latest)**
October 22, 2025
October 22, 2026
Custom features
Expiration timeline
ﾉ
Expand table
\nFeature
Supported Training
Config Versions
Training Config
Expiration
Deployment
Expiration
Custom named entity
recognition
2022-05-01  (latest)**
October 22, 2025
October 22, 2026
Custom text classification
2022-05-01  (latest)**
October 22, 2025
October 22, 2026
** For latest training configuration versions, the posted expiration dates are subject to
availability of a newer model version. If no newer model versions are available, the
expiration date may be extended.
Training configurations are typically available for six months after its release. If you've
assigned a trained configuration to a deployment, this deployment expires after twelve
months from the training config expiration. If your models are about to expire, you can
retrain and redeploy your models with the latest training configuration version.
After the training config expiration date, you'll have to use another supported training
configuration version to submit any training or deployment jobs. After the deployment
expiration date, your deployed model will be unavailable to be used for prediction.
After training config version expires, API calls will return an error when called or used if
called with an expired configuration version. By default, training requests use the latest
available training configuration version. To change the configuration version, use the
trainingConfigVersion  parameter when submitting a training job and assign the version
you want.
When you're making API calls to the following features, you need to specify the API-
VERISON  you want to use to complete your request. It's recommended to use the latest
available API versions.
If you're using Language Studio
 for your projects, you'll use the latest API version
available. Other API versions are only available through the REST APIs and client
libraries.
Use the following table to find which API versions are supported by each feature:
 Tip
It's recommended to use the latest supported configuration version.
API versions
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Feature
Supported versions
Latest Generally
Available version
Latest
preview
version
Custom text
classification
2022-05-01 , 2022-10-01-preview ,
2023-04-01
2022-05-01
2022-10-01-
preview
Conversational
language
understanding
2022-05-01 , 2022-10-01-preview ,
2023-04-01
2023-04-01
2022-10-01-
preview
Custom named entity
recognition
2022-05-01 , 2022-10-01-preview ,
2023-04-01 , 2023-04-15 , 2023-04-
15-preview
2023-04-15
2023-04-15-
preview
Orchestration workflow
2022-05-01 , 2022-10-01-preview ,
2023-04-01
2023-04-01
2022-10-01-
preview
Azure AI Language overview
ﾉ
Expand table
Next steps
Yes
No
\nHow to use Language service features
asynchronously
06/30/2025
The Language service enables you to send API requests asynchronously, using either the REST
API or client library. You can also include multiple different Language service features in your
request, to be performed on your data at the same time.
Currently, the following features are available to be used asynchronously:
Entity linking
Document summarization
Conversation summarization
Key phrase extraction
Language detection
Named Entity Recognition (NER)
Customer content detection
Sentiment analysis and opinion mining
Text Analytics for health
Personal Identifiable information (PII)
When you send asynchronous requests, you'll incur charges based on number of text records
you include in your request, for each feature use. For example, if you send a text record for
sentiment analysis and NER, it will be counted as sending two text records, and you'll be
charged for both according to your pricing tier
.
To submit an asynchronous job, review the reference documentation for the JSON body you'll
send in your request.
1. Add your documents to the analysisInput  object.
2. In the tasks  object, include the operations you want performed on your data. For
example, if you wanted to perform sentiment analysis, you would include the
SentimentAnalysisLROTask  object.
3. You can optionally:
a. Choose a specific version of the model used on your data.
b. Include additional Language service features in the tasks  object, to be performed on
your data at the same time.
Submit an asynchronous job using the REST API
\nOnce you've created the JSON body for your request, add your key to the Ocp-Apim-
Subscription-Key  header. Then send your API request to job creation endpoint. For example:
HTTP
A successful call will return a 202 response code. The operation-location  in the response
header will be the URL you'll use to retrieve the API results. The value will look similar to the
following URL:
HTTP
To get the status and retrieve the results of the request, send a GET request to the URL you
received in the operation-location  header from the previous API response. Remember to
include your key in the Ocp-Apim-Subscription-Key . The response will include the results of
your API call.
First, make sure you have the client library installed for your language of choice. For steps on
installing the client library, see the quickstart article for the feature you want to use.
Afterwards, use the client object to send asynchronous calls to the API. The method calls to use
will vary depending on your language. Use the available samples and reference documentation
to help you get started.
C#
Java
JavaScript
Python
When using this feature asynchronously, the API results are available for 24 hours from the
time the request was ingested, and is indicated in the response. After this time period, the
POST https://your-endpoint.cognitiveservices.azure.com/language/analyze-text/jobs?
api-version=2022-05-01
GET {Endpoint}/language/analyze-text/jobs/12345678-1234-1234-1234-12345678?api-
version=2022-05-01
Send asynchronous API requests using the client
library
Result availability
\nresults are purged and are no longer available for retrieval.
Starting in version 2022-07-01-preview  of the REST API, you can request automatic language
detection on your documents. By setting the language  parameter to auto , the detected
language code of the text will be returned as a language value in the response. This language
detection won't incur extra charges to your Language resource.
You can send up to 125,000 characters across all documents contained in the asynchronous
request, as measured by StringInfo.LengthInTextElements. This character limit is higher than the
limit for synchronous requests, to enable higher throughput.
If a document exceeds the character limit, the API will reject the entire request and return a 400
bad request  error if any document within it exceeds the maximum size.
Azure AI Language overview
Multilingual and emoji support
What's new
Automatic language detection
Data limits
７ Note
If you need to analyze larger documents than the limit allows, you can break the text
into smaller chunks of text before sending them to the API.
A document is a single string of text characters.
See also
\nTransparency Note for Azure AI Language
06/24/2025
An AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Creating a system that
is fit for its intended purpose requires an understanding of how the technology works, its
capabilities and limitations, and how to achieve the best performance. Microsoft's Transparency
Notes are intended to help you understand how our AI technology works, the choices system
owners can make that influence system performance and behavior, and the importance of
thinking about the whole system, including the technology, the people, and the environment.
You can use Transparency Notes when developing or deploying your own system, or share
them with the people who will use or be affected by your system.
Microsoft's Transparency Notes are part of a broader effort at Microsoft to put our AI principles
into practice. To find out more, see Microsoft AI principles
.
Azure AI Language is a cloud-based service that provides Natural Language Processing (NLP)
features for text mining and text analysis, including the following features:
Named Entity Recognition (NER), Personally Identifying Information (PII)
Text analytics for health
Key phrase extraction
Language detection
Sentiment analysis and opinion mining
Question answering
Summarization
Custom Named Entity Recognition (Custom NER)
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a Transparency Note?
The basics of Azure AI Language
Introduction
\nCustom text classification
Conversational language understanding
Read the overview to get an introduction to each feature and review the example use cases.
See the How-to guides and the API reference to understand more details about what each
feature does and what gets returned by the system.
This article contains basic guidelines for how to use Azure AI Language features responsibly.
Read the general information first and then jump to the specific article if you're using one of
the features below.
Transparency note for Named Entity Recognition
[Transparency note for Personally Identifying Information](/azure/ai-foundry/responsible-
ai/language-service/transparency-note-personally-identifiable-information
[Transparency note for text analytics for health](/azure/ai-foundry/responsible-
ai/language-service/transparency-note-health
Transparency note for key phrase extraction
Transparency note for language detection
Transparency note for sentiment analysis
Transparency note for question answering
Transparency note for summarization
Transparency note for custom Named Entity Recognition (custom NER)
Transparency note for custom text classification
Transparency note for conversational language understanding
Azure AI Language services can be used in multiple scenarios across a variety of industries.
Some examples listed by feature are:
Use Custom Named Entity Recognition for knowledge mining to enhance semantic
search. Search is foundational to any app that surfaces text content to users. Common
scenarios include catalog or document search, retail product search, or knowledge mining
for data science. Many enterprises across various industries want to build a rich search
experience over private, heterogeneous content, which includes both structured and
unstructured documents. As a part of their pipeline, developers can use custom NER for
extracting entities from the text that are relevant to their industry. These entities can be
used to enrich the indexing of the file for a more customized search experience.
Capabilities
Use cases
\nUse Named Entity Recognition to enhance or automate business processes. For
example, when reviewing insurance claims, recognized entities like name and location
could be highlighted to facilitate the review. Or a support ticket could be generated with
a customer's name and company automatically from an email.
Use Personally Identifiable Information to redact some categories of personal
information from documents to protect privacy. For example, if customer contact
records are accessible to first line support representatives, the company may want to
redact unnecessary customer's personal information from customer history to preserve
the customer's privacy.
Use Language Detection to detect languages for business workflow. For example, if a
company receives email in various languages from customers, they could use language
detection to route the emails by language to native speakers for ease of communication
with those customers.
Use Sentiment Analysis to monitor for positive and negative feedback trends in
aggregate. After the introduction of a new product, a retailer could use the sentiment
service to monitor multiple social media outlets for mentions of the product with their
sentiment. They could review the trending sentiment in their weekly product meetings.
Use Summarization to extract key information from public news articles. To produce
insights such as trends and news spotlights.
Use Key Phrase Extraction to view aggregate trends in text data. For example, a word
cloud can be generated with key phrases to help visualize key concepts in text comments
or feedback. For example, a hotel could generate a word cloud based on key phrases
identified in their comments and might see that people are commenting most frequently
about the location, cleanliness and helpful staff.
Use Text Analytics for Health for insights and statistics extraction. Identify medical
entities such as symptoms, medications, and diagnoses in clinical notes and diverse
clinical documents. Use this information for producing insights and statistics on patient
populations, searching clinical documents, research documents and publications.
Use Custom Text Classification for automatic email or ticket triaging. Support centers of
all types receive a high volume of emails or tickets containing unstructured, freeform text
and attachments. Timely review, acknowledgment, and routing to subject matter experts
within internal teams is critical. Email triage at this scale requires people to review and
route to the right departments, which takes time and resources. Custom text classification
can be used to analyze incoming text, and triage and categorize the content to be
automatically routed to the relevant departments for further action.