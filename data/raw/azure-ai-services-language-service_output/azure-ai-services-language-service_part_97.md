Feedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
If you run the container with an output mount and logging enabled, the container
generates log files that are helpful to troubleshoot issues that happen while starting or
running the container.
Azure AI containers overview
 Tip
For more troubleshooting information and guidance, see Disconnected containers
Frequently asked questions (FAQ).
Next steps
Yes
No
\nWhat are Azure AI containers?
Article • 03/31/2025
Azure AI services provide several Docker containers
 that let you use the same APIs
that are available in Azure, on-premises. Using these containers gives you the flexibility
to bring Azure AI services closer to your data for compliance, security or other
operational reasons. Container support is currently available for a subset of Azure AI
services.
Containerization is an approach to software distribution in which an application or
service, including its dependencies & configuration, is packaged together as a container
image. With little or no modification, a container image can be deployed on a container
host. Containers are isolated from each other and the underlying operating system, with
a smaller footprint than a virtual machine. Containers can be instantiated from container
images for short-term tasks, and removed when no longer needed.
Immutable infrastructure: Enable DevOps teams to leverage a consistent and
reliable set of known system parameters, while being able to adapt to change.
Containers provide the flexibility to pivot within a predictable ecosystem and avoid
configuration drift.
Control over data: Choose where your data gets processed by Azure AI services.
This can be essential if you can't send data to the cloud but need access to Azure
AI services APIs. Support consistency in hybrid environments – across data,
management, identity, and security.
Control over model updates: Flexibility in versioning and updating of models
deployed in their solutions.
Portable architecture: Enables the creation of a portable application architecture
that can be deployed on Azure, on-premises and the edge. Containers can be
deployed directly to Azure Kubernetes Service, Azure Container Instances, or to a
Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
High throughput / low latency: Provide customers the ability to scale for high
throughput and low latency requirements by enabling Azure AI services to run
physically close to their application logic and data. Containers don't cap
transactions per second (TPS) and can be made to scale both up and out to handle
demand if you provide the necessary hardware resources.
https://www.youtube-nocookie.com/embed/hdfbn4Q8jbo
Features and benefits
\nScalability: With the ever growing popularity of containerization and container
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
\nSupported PII entities
06/06/2025
Use this article to find the entity categories that the personally identifiable information (PII)
detection feature returns. This feature runs a predictive model to identify, categorize, and
redact sensitive information from an input document.
The PII feature includes the ability to detect personal ( PII ) and health ( PHI ) information.
The following entity categories are returned when you're sending API requests PII feature:
This type contains the following entity:
Entity
Person
Details
Names of people. Returned as both PII and PHI.
To get this entity type, add Person  to the piiCategories  parameter. Person  is returned in
the API response if detected.
Supported languages
Supported entities
７ Note
To detect protected health information (PHI), use the domain=phi  parameter and
model version 2020-04-01  or later.
The Type  and Subtype  are new designations introduced in the 2025-05-15-preview
version.
Preview API
Type: Person