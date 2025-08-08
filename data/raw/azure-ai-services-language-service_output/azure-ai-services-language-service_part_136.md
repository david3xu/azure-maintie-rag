1. Go to the Create
 page for Container Instances.
2. On the Basics tab, enter the following details:
Setting
Value
Subscription
Select your subscription.
Resource
group
Select the available resource group or create a new one such as cognitive-
services .
Container
name
Enter a name such as cognitive-container-instance . The name must be in
lower caps.
Location
Select a region for deployment.
Image type
If your container image is stored in a container registry that doesn’t require
credentials, choose Public . If accessing your container image requires
credentials, choose Private . Refer to container repositories and images for
details on whether or not the container image is Public  or Private  ("Public
Preview").
The LUIS container requires a .gz  model file that is pulled in at runtime. The container
must be able to access this model file via a volume mount from the container instance. To
upload a model file, follow these steps:
1. Create an Azure file share. Take note of the Azure Storage account name, key, and
file share name as you'll need them later.
2. export your LUIS model (packaged app) from the LUIS portal.
3. In the Azure portal, navigate to the Overview page of your storage account resource,
and select File shares.
4. Select the file share name that you recently created, then select Upload. Then upload
your packaged app.
Azure portal
Create an Azure Container Instance resource
using the Azure portal
ﾉ
Expand table
\nSetting
Value
Image name
Enter the Azure AI services container location. The location is what's used as an
argument to the docker pull  command. Refer to the container repositories
and images for the available image names and their corresponding repository.
The image name must be fully qualified specifying three parts. First, the
container registry, then the repository, finally the image name: <container-
registry>/<repository>/<image-name> .
Here is an example, mcr.microsoft.com/azure-cognitive-services/keyphrase
would represent the Key Phrase Extraction image in the Microsoft Container
Registry under the Azure AI services repository. Another example is,
containerpreview.azurecr.io/microsoft/cognitive-services-speech-to-text
which would represent the Speech to text image in the Microsoft repository of
the Container Preview container registry.
OS type
Linux
Size
Change size to the suggested recommendations for your specific Azure AI
container:
2 CPU cores
4 GB
3. On the Networking tab, enter the following details:
Setting
Value
Ports
Set the TCP port to 5000 . Exposes the container on port 5000.
4. On the Advanced tab, enter the required Environment Variables for the container
billing settings of the Azure Container Instance resource:
Key
Value
ApiKey
Copied from the Keys and endpoint page of the resource. It is a 84 alphanumeric-
character string with no spaces or dashes, xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx .
Billing
Your endpoint URL copied from the Keys and endpoint page of the resource.
Eula
accept
5. Select Review and Create
ﾉ
Expand table
ﾉ
Expand table
\n6. After validation passes, click Create to finish the creation process
7. When the resource is successfully deployed, it's ready
1. Select the Overview and copy the IP address. It will be a numeric IP address such as
55.55.55.55 .
2. Open a new browser tab and use the IP address, for example, http://<IP-
address>:5000 (http://55.55.55.55:5000 ). You will see the container's home page,
letting you know the container is running.
3. Select Service API Description to view the swagger page for the container.
4. Select any of the POST APIs and select Try it out. The parameters are displayed
including the input. Fill in the parameters.
5. Select Execute to send the request to your Container Instance.
You have successfully created and used Azure AI containers in Azure Container
Instance.
Use the Container Instance
Azure portal
\n![Image](images/page1353_image1.png)
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