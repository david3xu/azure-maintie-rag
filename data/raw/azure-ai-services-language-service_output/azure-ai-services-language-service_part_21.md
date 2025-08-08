Take a look at the recommended guidelines for information on improving accuracy.
When you train and deploy a conversation project in any language, you can immediately try
querying it in multiple languages. You may get varied results for different languages. To
improve the accuracy of any language, add utterances to your project in that language to
introduce the trained model to more syntax of that language.
See the service limits article.
Unlike LUIS, you cannot label the same text as 2 different entities. Learned components across
different entities are mutually exclusive, and only one learned span is predicted for each set of
characters.
Yes, you can import any LUIS application JSON file from the latest version in the service.
No, the service only supports JSON format. You can go to LUIS, import the .LU  file and export
it as a JSON file.
How do I get more accurate results for my project?
How do I get predictions in different languages?
How many intents, entities, utterances can I add to
a project?
Can I label the same word as 2 different entities?
Can I import a LUIS JSON file into conversational
language understanding?
Can I import a LUIS .LU  file into conversational
language understanding?
Can I use conversational language understanding
with custom question answering?
\nYes, you can use orchestration workflow to orchestrate between different conversational
language understanding and question answering projects. Start by creating orchestration
workflow projects, then connect your conversational language understanding and custom
question answering projects. To perform this action, make sure that your projects are under the
same Language resource.
Add any out of scope utterances to the none intent.
You can control the none intent threshold from UI through the project settings, by changing
the none intent threshold value. The values can be between 0.0 and 1.0. Also, you can change
this threshold from the APIs by changing the confidenceThreshold in settings object. Learn
more about none intent
Yes, only for predictions, and samples are available for Python
 and C#
. There is currently
no authoring support for the SDK.
Training
mode
Description
Language
availability
Pricing
Standard
training
Faster training times for quicker model
iteration.
Can only train
projects in English.
Included in your
pricing tier
.
Advanced
training
Slower training times using fine-tuned
neural network transformer models.
Can train multilingual
projects.
May incur additional
charges
.
See training modes for more information.
How do I handle out of scope or domain
utterances that aren't relevant to my intents?
How do I control the none intent?
Is there any SDK support?
What are the training modes?
ﾉ
Expand table
Are there APIs for this feature?
\nYes, all the APIs are available.
Authoring APIs
Prediction API
Conversational language understanding overview
Next steps
\nGuidance for integration and
responsible use with Azure AI Language
Article • 07/18/2023
Microsoft wants to help you responsibly develop and deploy solutions that use Azure AI
Language. We are taking a principled approach to upholding personal agency and
dignity by considering the fairness, reliability & safety, privacy & security, inclusiveness,
transparency, and human accountability of our AI systems. These considerations are in
line with our commitment to developing Responsible AI.
This article discusses Azure AI Language features and the key considerations for making
use of this technology responsibly. Consider the following factors when you decide how
to use and implement AI-powered products and features.
When you're getting ready to deploy AI-powered products or features, the following
activities help to set you up for success:
Understand what it can do: Fully assess the capabilities of any AI model you are
using to understand its capabilities and limitations. Understand how it will perform
in your particular scenario and context.
Test with real, diverse data: Understand how your system will perform in your
scenario by thoroughly testing it with real life conditions and data that reflects the
diversity in your users, geography and deployment contexts. Small datasets,
synthetic data and tests that don't reflect your end-to-end scenario are unlikely to
sufficiently represent your production performance.
Respect an individual's right to privacy: Only collect data and information from
individuals for lawful and justifiable purposes. Only use data and information that
you have consent to use for this purpose.
Legal review: Obtain appropriate legal advice to review your solution, particularly if
you will use it in sensitive or high-risk applications. Understand what restrictions
you might need to work within and your responsibility to resolve any issues that
might come up in the future. Do not provide any legal advice or guidance.
System review: If you're planning to integrate and responsibly use an AI-powered
product or feature into an existing system of software, customers or organizational
processes, take the time to understand how each part of your system will be
General guidelines
\naffected. Consider how your AI solution aligns with Microsoft's Responsible AI
principles.
Human in the loop: Keep a human in the loop, and include human oversight as a
consistent pattern area to explore. This means constant human oversight of the AI-
powered product or feature and maintaining the role of humans in decision-
making. Ensure you can have real-time human intervention in the solution to
prevent harm. This enables you to manage where the AI model doesn't perform as
required.
Security: Ensure your solution is secure and has adequate controls to preserve the
integrity of your content and prevent unauthorized access.
Customer feedback loop: Provide a feedback channel that allows users and
individuals to report issues with the service once it's been deployed. Once you've
deployed an AI-powered product or feature it requires ongoing monitoring and
improvement – be ready to implement any feedback and suggestions for
improvement.
Microsoft Responsible AI principles
Microsoft Responsible AI resources
Microsoft Azure Learning courses on Responsible AI
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying
Information
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
Article • 07/18/2023
This article provides details regarding how Azure AI Language processes your data.
Azure AI Language is designed with compliance, privacy, and security in mind. However,
you are responsible for its use and the implementation of this technology. It's your
responsibility to comply with all applicable laws and regulations in your jurisdiction.
Azure AI Language processes text data that is sent by the customer to the system
for the purposes of getting a response from one of the available features.
All results of the requested feature are sent back to the customer in the API
response as specified in the API reference. For example, if Language Detection is
requested, the language code is returned along with a confidence score for each
text record.
Azure AI Language uses aggregate telemetry such as which APIs are used and the
number of calls from each subscription and resource for service monitoring
purposes.
Azure AI Language doesn't store or process customer data outside the region
where the customer deploys the service instance.
Azure AI Language encrypts all content, including customer data, at rest.
Data sent in synchronous or asynchronous calls may be temporarily stored by
Azure AI Language for up to 48 hours only and is purged thereafter. This data is
encrypted and is only accessible to authorized on call engineers when service
support is needed for debugging purposes in the event of a catastrophic failure. To
prevent this temporary storage of input data, the LoggingOptOut query parameter
can be set accordingly. By default, this parameter is set to false for Language
Detection, Key Phrase Extraction, Sentiment Analysis and Named Entity
Recognition endpoints. The LoggingOptOut parameter is true by default for the PII
What data does Azure AI Language process
and how does it process it?
How is data retained and what customer
controls are available?
\nand health feature endpoints. More information on the LoggingOptOut query
parameter is available in the API reference.
To learn more about Microsoft's privacy and security commitments, visit the
Microsoft Trust Center
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying
Information
Transparency note for the health feature
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Guidance for integration and responsible use with Azure AI Language
See also
\nInstall and run Conversational Language
Understanding (CLU) containers
Article • 04/29/2025
Containers enable you to host the CLU API on your own infrastructure. If you have security or
data governance requirements that can't be fulfilled by calling CLU remotely, then containers
might be a good option.
If you don't have an Azure subscription, create a free account
 before you begin.
You must meet the following prerequisites before using CLU containers.
If you don't have an Azure subscription, create a free account
.
Docker
 installed on a host computer. Docker must be configured to allow the
containers to connect with and send billing data to Azure.
On Windows, Docker must also be configured to support Linux containers.
You should have a basic understanding of Docker concepts
.
A Language resource
Three primary parameters for all Azure AI containers are required. The Microsoft Software
License Terms must be present with a value of accept. An Endpoint URI and API key are also
needed.
The {ENDPOINT_URI}  value is available on the Azure portal Overview page of the corresponding
Azure AI services resource. Go to the Overview page, hover over the endpoint, and a Copy to
clipboard ＝ icon appears. Copy and use the endpoint where needed.
７ Note
The data limits in a single synchronous API call for the CLU container are 5,120 characters
per document and up to 10 documents per call.
Prerequisites
Gather required parameters
Endpoint URI
\nThe {API_KEY}  value is used to start the container and is available on the Azure portal's Keys
page of the corresponding Azure AI services resource. Go to the Keys page, and select the
Copy to clipboard ＝ icon.
The host is an x64-based computer that runs the Docker container. It can be a computer on
your premises or a Docker hosting service in Azure, such as:
Keys
） Important
These subscription keys are used to access your Azure AI services API. Don't share your
keys. Store them securely. For example, use Azure Key Vault. We also recommend that you
regenerate these keys regularly. Only one key is necessary to make an API call. When you
regenerate the first key, you can use the second key for continued access to the service.
Host computer requirements and
recommendations
\n![Image](images/page209_image1.png)

![Image](images/page209_image2.png)
\nAzure Kubernetes Service.
Azure Container Instances.
A Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
The following table describes the minimum and recommended specifications for the available
container. Each CPU core must be at least 2.6 gigahertz (GHz) or faster.
It's recommended to have a CPU with AVX-512 instruction set, for the best experience
(performance and accuracy).
Minimum host specs
Recommended host specs
CLU
1 core, 2 GB memory
4 cores, 8 GB memory
CPU core and memory correspond to the --cpus  and --memory  settings, which are used as part
of the docker run  command.
Before you proceed with running the docker image, you need to export your own trained
model to expose it to your container. Use the following command to extract your model and
replace the placeholders below with your own values:
Placeholder
Value
Format or example
{API_KEY}
The key for your Language
resource. You can find it on
your resource's Key and
endpoint page, on the Azure
portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
{ENDPOINT_URI}
The endpoint for accessing
the Conversational Language
Understanding API. You can
find it on your resource's Key
and endpoint page, on the
Azure portal.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
ﾉ
Expand table
Export your Conversational Language
Understanding model
ﾉ
Expand table