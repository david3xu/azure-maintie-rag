Service
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
\nWhat is key phrase extraction in Azure
AI Language?
Article • 02/21/2025
Key phrase extraction is one of the features offered by Azure AI Language, a collection
of machine learning and AI algorithms in the cloud for developing intelligent
applications that involve written language. Use key phrase extraction to quickly identify
the main concepts in text. For example, in the text "The food was delicious and the staff
were wonderful.", key phrase extraction returns the main topics: "food" and "wonderful
staff."
This documentation contains the following types of articles:
Quickstarts are getting-started instructions to guide you through making requests
to the service.
How-to guides contain instructions for using the service in more specific or
customized ways.
To use this feature, you submit data for analysis and handle the API output in your
application. Analysis is performed as-is, with no added customization to the model used
on your data.
1. Create an Azure AI Language resource, which grants you access to the features
offered by Azure AI Language. It generates a password (called a key) and an
endpoint URL that you use to authenticate API requests.
2. Create a request using either the REST API or the client library for C#, Java,
JavaScript, and Python. You can also send asynchronous calls with a batch request
to combine API requests for multiple features into a single call.
3. Send the request containing your text data. Your key and endpoint are used for
authentication.
4. Stream or store the response locally.
Typical workflow
Get started with Key phrase extraction
\nTo use key phrase extraction, you submit raw unstructured text for analysis and handle
the API output in your application. Analysis is performed as-is, with no additional
customization to the model used on your data. There are two ways to use key phrase
extraction:
Development
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use entity linking
with text examples with your own data when you sign up. For more
information, see the Azure AI Foundry website
 or Azure AI Foundry
documentation.
REST API or Client
library (Azure SDK)
Integrate key phrase extraction into your applications using the REST API,
or the client library available in a variety of languages. For more
information, see the key phrase extraction quickstart.
Docker container
Use the available Docker container to deploy this feature on-premises.
These docker containers enable you to bring the service closer to your
data for compliance, security, or other operational reasons.
As you use this feature in your applications, see the following reference documentation
and samples for Azure AI Language:
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
Reference documentation and code samples
ﾉ
Expand table
Responsible AI
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
An AI system includes not only the technology, but also the people who use it, the
people who are affected by it, and the environment in which it's deployed. Read the
transparency note for key phrase extraction to learn about responsible AI use and
deployment in your systems. For more information, see the following articles:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
There are two ways to get started using the entity linking feature:
Azure AI Foundry is a web-based platform that lets you use several Azure AI
Language features without needing to write code.
The quickstart article for instructions on making requests to the service using the
REST API and client library SDK.
Next steps
Yes
No
\nQuickstart: using the Key Phrase
Extraction client library and REST API
Article • 02/17/2025
Create a Project in Foundry in the Azure AI Foundry Portal
Using the left side pane, select Playgrounds. Then select the Try the Language
Playground button.
Prerequisites
Navigate to the Azure AI Foundry Playground

\n![Image](images/page459_image1.png)
\nThe Language Playground consists of four sections:
Top banner: You can select any of the currently available Language services here.
Right pane: This pane is where you can find the Configuration options for the
service, such as the API and model version, along with features specific to the
service.
Center pane: This pane is where you enter your text for processing. After the
operation is run, some results are shown here.
Right pane: This pane is where Details of the run operation are shown.
Here you can select the Key Phrase Extraction capability by choosing the top banner tile,
Extract key phrases.
Extract key phrases is designed to extract key phrases from text.
In Configuration there are the following options:
Option
Description
Select API version
Select which version of the API to use.
Select model version
Select which version of the model to use.
Select text language
Select the language of the input text.
After your operation is completed, each entity is underlined in the center pane and the
Details section contains the following fields for the overall sentiment and the sentiment
of each sentence:
Field
Description
Extracted key phrases
A list of the extracted key phrases.
Use Key Phrase Extraction in the Azure AI
Foundry Playground
Use Extract key phrases
ﾉ
Expand table
ﾉ
Expand table