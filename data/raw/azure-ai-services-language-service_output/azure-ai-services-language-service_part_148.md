Scalability: With the ever growing popularity of containerization and container
orchestration software, such as Kubernetes; scalability is at the forefront of
technological advancements. Building on a scalable cluster foundation, application
development caters to high availability.
Azure AI containers provide the following set of Docker containers, each of which
contains a subset of functionality from services in Azure AI services. You can find
instructions and image locations in the tables below.
Service
Container
Description
Availability
Anomaly
detector
Anomaly
Detector
(image
)
The Anomaly Detector API enables you to
monitor and detect abnormalities in your time
series data with machine learning.
Generally
available
Service
Container
Description
Availability
LUIS
LUIS (image
)
Loads a trained or published Language
Understanding model, also known as a
LUIS app, into a docker container and
provides access to the query
predictions from the container's API
endpoints. You can collect query logs
from the container and upload these
back to the LUIS portal
 to improve
the app's prediction accuracy.
Generally
available.
This container
can also run in
disconnected
environments.
Containers in Azure AI services
７ Note
See Install and run Document Intelligence containers for Azure AI Document
Intelligence container instructions and image locations.
Decision containers
ﾉ
Expand table
Language containers
ﾉ
Expand table
\nService
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
\nTransparency note for summarization
06/24/2025
An AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Creating a system that
is fit for its intended purpose requires an understanding of how the technology works, its
capabilities and limitations, and how to achieve the best performance.
Microsoft transparency notes are intended to help you understand how our AI technology
works, and the choices that you as a system owner can make that influence system
performance and behavior. It's important to think about the whole system, including the
technology, the people, and the environment. You can use transparency notes when you
develop or deploy your own system, or share them with the people who will use or be affected
by your system.
Transparency notes are part of a broader effort at Microsoft to put our AI principles into
practice. To find out more, see Microsoft AI principles
.
Summarization uses natural language processing techniques to condense articles, papers, or
documents into key sentences. This feature is provided as an API for developers to build
intelligent solutions based on the relevant information extracted and can support various use
cases.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a transparency note?
The basics of Summarization
Introduction
Capabilities
Document summarization
\nDocument summarization uses natural language processing techniques to generate a
summary for documents. There are two general approaches to auto-summarization:
extractive and abstractive.
This feature extracts sentences that collectively represent the most important or relevant
information within the original content. It locates key sentences in an unstructured text
document. These sentences collectively convey the main idea of the document.
Different from extractive summarization, document abstractive summarization generates a
summary with concise, coherent sentences or words which are not simply extracted from
the original document.
You can use document summarization in multiple scenarios, across a variety of industries.
For example, you can use extractive summarization to:
Assist the processing of documents to improve efficiency. Distill critical information
from lengthy documents, reports, and other text forms,highlight key sentences in
documents, andquickly skim documents in a library.
Extract key information from public news articles to produce insights such as trends
and news spotlights, and generate news feed content.
Classify or cluster documents by their contents. Use Summarization to surface key
concepts from documents and use those key concepts to group documents that are
similar.
Distill important information from long documents to empower solutions such as
search, question and answering, and decision support.
We encourage you to come up with use cases that most closely match your own particular
context and requirements. Draw on actionable information that enables responsible integration
in your use cases, and conduct your own testing specific to your scenarios.
The basics of document extractive summarization
The basics of document abstractive summarization
Example use cases
Considerations when you choose a use case
\nThe summarization models reflect certain societal views that are over-represented in the
training data, relative to other, marginalized perspectives. The models reflect societal biases
and other undesirable content present in the training data. As a result, we caution against
using the models in high-stakes scenarios, where unfair, unreliable, or offensive behavior might
be extremely costly or lead to harm.
Avoid real-time, critical safety alerting. Don't rely on this feature for scenarios that
require real-time alerts to trigger intervention to prevent injury. For example, don't rely
on summarization for turning off a piece of heavy machinery when a harmful action is
present.
The feature isn't suitable for scenarios where up-to-date, factually accurate information
is crucial, unless you have human reviewers. The service doesn't have information about
current events after its training date, probably has missing knowledge about some topics,
and might not always produce factually accurate information.
Avoid scenarios in which the use or misuse of the system could have a consequential
impact on life opportunities or legal status. For example, avoid scenarios in which the AI
system could affect an individual's legal status or legal rights. Additionally, avoid
scenarios in which the AI system could affect an individual's access to credit, education,
employment, healthcare, housing, insurance, social welfare benefits, services,
opportunities, or the terms on which they are provided.
Legal and regulatory considerations: Organizations need to evaluate potential specific
legal and regulatory obligations when using any AI services and solutions, which may not
be appropriate for use in every industry or scenario. Additionally, AI services or solutions
are not designed for and may not be used in ways prohibited in applicable terms of
service and relevant codes of conduct.
Transparency note for Azure AI Language
Transparency note for named entity recognition
Transparency note for health
Transparency note for key phrase extraction
Transparency note for sentiment analysis
Guidance for integration and responsible use with language
Data privacy for language
Next steps