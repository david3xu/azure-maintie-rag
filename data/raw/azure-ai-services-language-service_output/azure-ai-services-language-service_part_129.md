Service
Container
Description
Availability
Language
service
Key Phrase
Extraction
(image
)
Extracts key phrases to identify the
main points. For example, for the input
text "The food was delicious and there
were wonderful staff," the API returns
the main talking points: "food" and
"wonderful staff".
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Text Language
Detection
(image
)
For up to 120 languages, detects which
language the input text is written in and
report a single language code for every
document submitted on the request.
The language code is paired with a
score indicating the strength of the
score.
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Sentiment
Analysis (image
)
Analyzes raw text for clues about
positive or negative sentiment. This
version of sentiment analysis returns
sentiment labels (for example positive
or negative) for each document and
sentence within it.
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Text Analytics for
health (image
)
Extract and label medical information
from unstructured clinical text.
Generally
available
Language
service
Named Entity
Recognition
(image
)
Extract named entities from text.
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Personally
Identifiable
Information (PII)
detection
(image
)
Detect and redact personally
identifiable information entities from
text.
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Custom Named
Entity Recognition
(image
)
Extract named entities from text, using
a custom model you create using your
data.
Generally
available
Language
service
Summarization
(image
)
Summarize text from various sources.
Public preview.
This container
can also run in
disconnected
environments.
\nService
Container
Description
Availability
[Language
service][ta-
containers-
clu]
Conversational
Language
Understanding
(image
Interpret conversational language.
Generally
Available.
This container
can also run in
disconnected
environments.
Translator
Translator
(image
)
Translate text in several languages and
dialects.
Generally
available. Gated -
request access
.
This container
can also run in
disconnected
environments.
Service
Container
Description
Availability
Speech
Service
API
Speech to text
(image
)
Transcribes continuous real-time
speech into text.
Generally available.
This container can also
run in disconnected
environments.
Speech
Service
API
Custom Speech to
text (image
)
Transcribes continuous real-time
speech into text using a custom
model.
Generally available
This container can also
run in disconnected
environments.
Speech
Service
API
Neural Text to
speech (image
)
Converts text to natural-sounding
speech using deep neural network
technology, allowing for more
natural synthesized speech.
Generally available.
This container can also
run in disconnected
environments.
Speech
Service
API
Speech language
identification
(image
)
Determines the language of spoken
audio.
Preview
Speech containers
ﾉ
Expand table
Vision containers
ﾉ
Expand table
\nService
Container
Description
Availability
Azure AI
Vision
Read OCR
(image
)
The Read OCR container allows you to extract
printed and handwritten text from images and
documents with support for JPEG, PNG, BMP,
PDF, and TIFF file formats. For more
information, see the Read API documentation.
Generally Available.
This container can
also run in
disconnected
environments.
Spatial
Analysis
Spatial
analysis
(image
)
Analyzes real-time streaming video to
understand spatial relationships between
people, their movement, and interactions with
objects in physical environments.
Preview
Additionally, some containers are supported in the Azure AI services multi-service
resource offering. You can create one single Azure AI services resource and use the same
billing key across supported services for the following services:
Azure AI Vision
LUIS
Language service
You must satisfy the following prerequisites before using Azure AI containers:
Docker Engine: You must have Docker Engine installed locally. Docker provides
packages that configure the Docker environment on macOS
, Linux
, and Windows
.
On Windows, Docker must be configured to support Linux containers. Docker containers
can also be deployed directly to Azure Kubernetes Service or Azure Container Instances.
Docker must be configured to allow the containers to connect with and send billing data
to Azure.
Familiarity with Microsoft Container Registry and Docker: You should have a basic
understanding of both Microsoft Container Registry and Docker concepts, like registries,
repositories, containers, and container images, as well as knowledge of basic docker
commands.
For a primer on Docker and container basics, see the Docker overview
.
Individual containers can have their own requirements, as well, including server and
memory allocation requirements.
Prerequisites
Azure AI services container security
\nSecurity should be a primary focus whenever you're developing applications. The
importance of security is a metric for success. When you're architecting a software
solution that includes Azure AI containers, it's vital to understand the limitations and
capabilities available to you. For more information about network security, see
Configure Azure AI services virtual networks.
The following diagram illustrates the default and non-secure approach:
As an example of an alternative and secure approach, consumers of Azure AI containers
could augment a container with a front-facing component, keeping the container
endpoint private. Let's consider a scenario where we use Istio
 as an ingress gateway.
Istio supports HTTPS/TLS and client-certificate authentication. In this scenario, the Istio
frontend exposes the container access, presenting the client certificate that is approved
beforehand with Istio.
Nginx
 is another popular choice in the same category. Both Istio and Nginx act as a
service mesh and offer additional features including things like load-balancing, routing,
and rate-control.
） Important
By default there is no security on the Azure AI services container API. The reason for
this is that most often the container will run as part of a pod which is protected
from the outside by a network bridge. However, it is possible for users to construct
their own authentication infrastructure to approximate the authentication methods
used when accessing the cloud-based Azure AI services.
\nFeedback
The Azure AI containers are required to submit metering information for billing
purposes. Failure to allowlist various network channels that the Azure AI containers rely
on will prevent the container from working.
The host should allowlist port 443 and the following domains:
*.cognitive.microsoft.com
*.cognitiveservices.azure.com
Deep packet inspection (DPI)
 is a type of data processing that inspects in detail the
data sent over a computer network, and usually takes action by blocking, rerouting, or
logging it accordingly.
Disable DPI on the secure channels that the Azure AI containers create to Microsoft
servers. Failure to do so will prevent the container from functioning correctly.
Developer samples are available at our GitHub repository
.
Learn about container recipes you can use with the Azure AI services.
Install and explore the functionality provided by containers in Azure AI services:
Anomaly Detector containers
Azure AI Vision containers
Language Understanding (LUIS) containers
Speech Service API containers
Language service containers
Translator containers
Container networking
Allowlist Azure AI services domains and ports
Disable deep packet inspection
Developer samples
Next steps
\nWas this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Yes
No
\nWhat is Text Analytics for health?
02/17/2025
Text Analytics for health is one of the prebuilt features offered by Azure AI Language. It is a
cloud-based API service that applies machine-learning intelligence to extract and label relevant
medical information from a variety of unstructured texts such as doctor's notes, discharge
summaries, clinical documents, and electronic health records.
） Important
Text Analytics for health is a capability provided “AS IS” and “WITH ALL FAULTS.” Text
Analytics for health is not intended or made available for use as a medical device, clinical
support, diagnostic tool, or other technology intended to be used in the diagnosis, cure,
mitigation, treatment, or prevention of disease or other conditions, and no license or right
is granted by Microsoft to use this capability for such purposes. This capability is not
designed or intended to be implemented or deployed as a substitute for professional
medical advice or healthcare opinion, diagnosis, treatment, or the clinical judgment of a
healthcare professional, and should not be used as such. The customer is solely
responsible for any use of Text Analytics for health. The customer must separately license
any and all source vocabularies it intends to use under the terms set for that UMLS
Metathesaurus License Agreement Appendix
 or any future equivalent link. The
customer is responsible for ensuring compliance with those license terms, including any
geographic or other applicable restrictions.
Text Analytics for health now allows extraction of Social Determinants of Health (SDOH)
and ethnicity mentions in text. This capability may not cover all potential SDOH and does
not derive inferences based on SDOH or ethnicity (for example, substance use information
is surfaced, but substance abuse is not inferred). All decisions leveraging outputs of the
Text Analytics for health that impact individuals or resource allocation (including, but not
limited to, those related to billing, human resources, or treatment managing care) should
be made with human oversight and not be based solely on the findings of the model. The
purpose of the SDOH and ethnicity extraction capability is to help providers improve
health outcomes and it should not be used to stigmatize or draw negative inferences
about the users or consumers of SDOH data, or patient populations beyond the stated
purpose of helping providers improving health outcomes.
 Tip
Try out Text Analytics for health in Azure AI Foundry portal
, where you can utilize a
currently existing Language Studio resource or create a new Azure AI Foundry resource
\nThis documentation contains the following types of articles:
The quickstart article provides a short tutorial that guides you with making your first
request to the service.
The how-to guides contain detailed instructions on how to make calls to the service
using the hosted API or using the on-premises Docker container.
The conceptual articles provide in-depth information on each of the service's features,
named entity recognition, relation extraction, entity linking, and assertion detection.
Text Analytics for health performs four key functions which are named entity recognition,
relation extraction, entity linking, and assertion detection, all with a single API call.
Named entity recognition is used to perform a semantic extraction of words and phrases
mentioned from unstructured text that are associated with any of the supported entity
types, such as diagnosis, medication name, symptom/sign, or age.
Text Analytics for health can receive unstructured text in English, German, French, Italian,
Spanish, Portuguese, and Hebrew.
Additionally, Text Analytics for health can return the processed output using the Fast
Healthcare Interoperability Resources (FHIR) structure which enables the service's integration
with other electronic health systems.
in order to use this service.
Text Analytics for health features
Named Entity Recognition
https://learn.microsoft.com/Shows/AI-Show/Introducing-Text-Analytics-for-Health/player
\n![Image](images/page1288_image1.png)
\nText Analytics for health can be used in multiple scenarios across a variety of industries. Some
common customer motivations for using Text Analytics for health include:
Assisting and automating the processing of medical documents by proper medical coding
to ensure accurate care and billing.
Increasing the efficiency of analyzing healthcare data to help drive the success of value-
based care models similar to Medicare.
Minimizing healthcare provider effort by automating the aggregation of key patient data
for trend and pattern monitoring.
Facilitating and supporting the adoption of HL7 standards for improved exchange,
integration, sharing, retrieval, and delivery of electronic health information in all
healthcare services.
Use case
Description
Extract insights and statistics
Identify medical entities such as symptoms, medications, diagnosis from
clinical and research documents in order to extract insights and statistics
for different patient cohorts.
Develop predictive
models using historic data
Power solutions for planning, decision support, risk analysis and more,
based on prediction models created from historic data.
Annotate and curate medical
information
Support solutions for clinical data annotation and curation such as
automating clinical coding and digitizing manually created data.
Review and report medical
information
Support solutions for reporting and flagging possible errors in medical
information resulting from reviewal processes such as quality assurance.
Assist with decision support
Enable solutions that provide humans with assistive information relating
to patients’ medical information for faster and more reliable decisions.
To use Text Analytics for health, you submit raw unstructured text for analysis and handle the
API output in your application. Analysis is performed as-is, with no additional customization to
the model used on your data. There are two ways to use Text Analytics for health:
Usage scenarios
Example use cases: 
ﾉ
Expand table
Get started with Text Analytics for health
\nDevelopment
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use entity linking with
text examples with your own data when you sign up. For more information, see
the Azure AI Foundry website
 or Azure AI Foundry documentation.
REST API or Client
library (Azure SDK)
Integrate Text Analytics for health into your applications using the REST API, or
the client library available in a variety of languages. For more information, see the
Text Analytics for health quickstart.
Docker container
Use the available Docker container to deploy this feature on-premises. These
docker containers enable you to bring the service closer to your data for
compliance, security, or other operational reasons.
Text Analytics for health is designed to receive unstructured text for analysis. For more
information, see data and service limits.
Text Analytics for health works with a variety of input languages. For more information, see
language support.
As you use this feature in your applications, see the following reference documentation and
samples for Azure AI Language:
Development option / language
Reference documentation
Samples
REST API
REST API documentation
C#
C# documentation
C# samples
Java
Java documentation
Java Samples
JavaScript
JavaScript documentation
JavaScript samples
Python
Python documentation
Python samples
ﾉ
Expand table
Input requirements and service limits
Reference documentation and code samples
ﾉ
Expand table
Responsible use of AI