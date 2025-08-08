need to work within and your responsibility to resolve any issues that might come up in
the future. Do not provide any legal advice or guidance.
System review: If you're planning to integrate and responsibly use an AI-powered
product or feature into an existing system of software, customers or organizational
processes, take the time to understand how each part of your system will be affected.
Consider how your AI solution aligns with Microsoft's Responsible AI principles.
Human in the loop: Keep a human in the loop, and include human oversight as a
consistent pattern area to explore. This means constant human oversight of the AI-
powered product or feature and maintaining the role of humans in decision-making.
Ensure you can have real-time human intervention in the solution to prevent harm. This
enables you to manage where the AI model doesn't perform as required.
Security: Ensure your solution is secure and has adequate controls to preserve the
integrity of your content and prevent unauthorized access.
Customer feedback loop: Provide a feedback channel that allows users and individuals to
report issues with the service once it's been deployed. Once you've deployed an AI-
powered product or feature it requires ongoing monitoring and improvement – be ready
to implement any feedback and suggestions for improvement.
Microsoft Responsible AI principles
Microsoft Responsible AI resources
Microsoft Azure Learning courses on Responsible AI
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for the health feature
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Data Privacy and Security for Azure AI Language
Learn more about Responsible AI
See also
\nData, privacy, and security for Azure AI
Language
06/24/2025
This article provides details regarding how Azure AI Language processes your data. Azure AI
Language is designed with compliance, privacy, and security in mind. However, you are
responsible for its use and the implementation of this technology. It's your responsibility to
comply with all applicable laws and regulations in your jurisdiction.
Azure AI Language processes text data that is sent by the customer to the system for the
purposes of getting a response from one of the available features.
All results of the requested feature are sent back to the customer in the API response as
specified in the API reference. For example, if Language Detection is requested, the
language code is returned along with a confidence score for each text record.
Azure AI Language uses aggregate telemetry such as which APIs are used and the
number of calls from each subscription and resource for service monitoring purposes.
Azure AI Language doesn't store or process customer data outside the region where the
customer deploys the service instance.
Azure AI Language encrypts all content, including customer data, at rest.
Data sent in synchronous or asynchronous calls may be temporarily stored by Azure AI
Language for up to 48 hours only and is purged thereafter. This data is encrypted and is
only accessible to authorized on call engineers when service support is needed for
debugging purposes in the event of a catastrophic failure. To prevent this temporary
storage of input data, the LoggingOptOut query parameter can be set accordingly. By
default, this parameter is set to false for Language Detection, Key Phrase Extraction,
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What data does Azure AI Language process and
how does it process it?
How is data retained and what customer controls
are available?
\nSentiment Analysis and Named Entity Recognition endpoints. The LoggingOptOut
parameter is true by default for the PII and health feature endpoints. More information on
the LoggingOptOut query parameter is available in the API reference.
To learn more about Microsoft's privacy and security commitments, visit the Microsoft Trust
Center
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for the health feature
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Guidance for integration and responsible use with Azure AI Language
See also
\nHow to use key phrase extraction
06/21/2025
The key phrase extraction feature can evaluate unstructured text, and for each document,
return a list of key phrases.
This feature is useful if you need to quickly identify the main points in a collection of
documents. For example, given input text "The food was delicious and the staff was wonderful",
the service returns the main topics: "food" and "wonderful staff".
To use key phrase extraction, you submit raw unstructured text for analysis and handle the API
output in your application. Analysis is performed as-is, with no additional customization to the
model used on your data. There are two ways to use key phrase extraction:
Development
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use entity linking with
text examples with your own data when you sign up. For more information, see
the Azure AI Foundry website
 or Azure AI Foundry documentation.
REST API or Client
library (Azure SDK)
Integrate key phrase extraction into your applications using the REST API, or the
client library available in a variety of languages. For more information, see the key
phrase extraction quickstart.
Docker container
Use the available Docker container to deploy this feature on-premises. These
docker containers enable you to bring the service closer to your data for
compliance, security, or other operational reasons.
 Tip
If you want to start using this feature, you can follow the quickstart article to get started.
You can also make requests using Azure AI Foundry without needing to write code.
Development options
ﾉ
Expand table
Determine how to process the data (optional)
Specify the key phrase extraction model
\nBy default, key phrase extraction uses the latest available AI model on your text. You can also
configure your API requests to use a specific model version.
When you submit documents to be processed by key phrase extraction, you can specify which
of the supported languages they're written in. if you don't specify a language, key phrase
extraction defaults to English. The API may return offsets in the response to support different
multilingual and emoji encodings.
Key phrase extraction works best when you give it bigger amounts of text to work on. This is
opposite from sentiment analysis, which performs better on smaller amounts of text. To get the
best results from both operations, consider restructuring the inputs accordingly.
To send an API request, You need your Language resource endpoint and key.
Analysis is performed upon receipt of the request. Using the key phrase extraction feature
synchronously is stateless. No data is stored in your account, and results are returned
immediately in the response.
When using this feature asynchronously, the API results are available for 24 hours from the
time the request was ingested, and is indicated in the response. After this time period, the
results are purged and are no longer available for retrieval.
When you receive results from the API, the order of the returned key phrases is determined
internally, by the model. You can stream the results to an application, or save the output to a
file on the local system.
Input languages
Submitting data
７ Note
You can find the key and endpoint for your Language resource on the Azure portal. They
are located on the resource's Key and endpoint page, under resource management.
Getting key phrase extraction results
Service and data limits
\nFor information on the size and number of requests you can send per minute and second, see
the service limits article.
Key Phrase Extraction overview
Next steps
\nInstall and run Key Phrase Extraction
containers
06/21/2025
Containers enable you to host the Key Phrase Extraction API on your own infrastructure. If you
have security or data governance requirements that can't be fulfilled by calling Key Phrase
Extraction remotely, then containers might be a good option.
Containers enable you to run the Key Phrase Extraction APIs in your own environment and are
great for your specific security and data governance requirements. The Key Phrase Extraction
containers provide advanced natural language processing over raw text, and include three
main functions: sentiment analysis, Key Phrase Extraction, and language detection.
If you don't have an Azure subscription, create a free account
.
Docker
 installed on a host computer. Docker must be configured to allow the
containers to connect with and send billing data to Azure.
On Windows, Docker must also be configured to support Linux containers.
You should have a basic understanding of Docker concepts
.
A Language resource 
with the free (F0) or standard (S) pricing tier
.
Three primary parameters for all Azure AI containers are required. The Microsoft Software
License Terms must be present with a value of accept. An Endpoint URI and API key are also
needed.
The {ENDPOINT_URI}  value is available on the Azure portal Overview page of the corresponding
Azure AI services resource. Go to the Overview page, hover over the endpoint, and a Copy to
７ Note
The free account is limited to 5,000 text records per month and only the Free and
Standard pricing tiers
 are valid for containers. For more information on
transaction request rates, see Data and service limits.
Prerequisites
Gather required parameters
Endpoint URI
\nclipboard ＝ icon appears. Copy and use the endpoint where needed.
The {API_KEY}  value is used to start the container and is available on the Azure portal's Keys
page of the corresponding Azure AI services resource. Go to the Keys page, and select the
Copy to clipboard ＝ icon.
Keys
） Important
These subscription keys are used to access your Azure AI services API. Don't share your
keys. Store them securely. For example, use Azure Key Vault. We also recommend that you
regenerate these keys regularly. Only one key is necessary to make an API call. When you
regenerate the first key, you can use the second key for continued access to the service.
Host computer requirements and
recommendations
\n![Image](images/page478_image1.png)

![Image](images/page478_image2.png)
\nThe host is an x64-based computer that runs the Docker container. It can be a computer on
your premises or a Docker hosting service in Azure, such as:
Azure Kubernetes Service.
Azure Container Instances.
A Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
The following table describes the minimum and recommended specifications for the available
Key Phrase Extraction containers. Each CPU core must be at least 2.6 gigahertz (GHz) or faster.
The allowable Transactions Per Second (TPS) are also listed.
Minimum host
specs
Recommended host
specs
Minimum
TPS
Maximum
TPS
Key Phrase
Extraction
1 core, 2GB
memory
1 core, 4GB memory
15
30
CPU core and memory correspond to the --cpus  and --memory  settings, which are used as part
of the docker run  command.
The key phrase extraction container image can be found on the mcr.microsoft.com  container
registry syndicate. It resides within the azure-cognitive-services/textanalytics/  repository
and is named keyphrase . The fully qualified container image name is,
mcr.microsoft.com/azure-cognitive-services/textanalytics/keyphrase .
To use the latest version of the container, you can use the latest  tag. You can also find a full
list of tags on the MCR
.
Use the docker pull
 command to download a container image from Microsoft Container
Registry.
ﾉ
Expand table
Get the container image with docker pull
docker pull mcr.microsoft.com/azure-cognitive-
services/textanalytics/keyphrase:latest
 Tip
\nOnce the container is on the host computer, use the docker run
 command to run the
containers. The container will continue to run until you stop it.
To run the Key Phrase Extraction container, execute the following docker run  command.
Replace the placeholders below with your own values:
Placeholder
Value
Format or example
{API_KEY}
The key for your Key Phrase
Extraction resource. You can find it on
your resource's Key and endpoint
page, on the Azure portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
{ENDPOINT_URI}
The endpoint for accessing the Key
Phrase Extraction API. You can find it
https://<your-custom-
subdomain>.cognitiveservices.azure.com
You can use the docker images
 command to list your downloaded container images.
For example, the following command lists the ID, repository, and tag of each downloaded
container image, formatted as a table:
docker images --format "table {{.ID}}\t{{.Repository}}\t{{.Tag}}"
IMAGE ID         REPOSITORY                TAG
<image-id>       <repository-path/name>    <tag-name>
Run the container with docker run
） Important
The docker commands in the following sections use the back slash, \ , as a line
continuation character. Replace or remove this based on your host operating
system's requirements.
The Eula , Billing , and ApiKey  options must be specified to run the container;
otherwise, the container won't start. For more information, see Billing.
The sentiment analysis and language detection containers use v3 of the API, and are
generally available. The Key Phrase Extraction container uses v2 of the API, and is in
preview.
ﾉ
Expand table