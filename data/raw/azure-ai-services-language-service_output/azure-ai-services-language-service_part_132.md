time to understand how each part of your system will be affected. Consider how your AI
solution aligns with Microsoft Responsible AI principles.
Human in the loop: Keep a human in the loop and include human oversight as a
consistent pattern area to explore. This means constant human oversight of the AI-
powered product or feature and ensuring the role of humans in making any decisions
that are based on the model’s output. To prevent harm and to manage how the AI model
performs, ensure that humans have a way to intervene in the solution in real time.
Security: Ensure that your solution is secure and that it has adequate controls to preserve
the integrity of your content and prevent unauthorized access.
Customer feedback loop: Provide a feedback channel that users and individuals can use
to report issues with the service after it's deployed. After you deploy an AI-powered
product or feature, it requires ongoing monitoring and improvement. Have a plan and be
ready to implement feedback and suggestions for improvement.
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Data Privacy and Security for Azure AI Language
Guidance for integration and responsible use with Azure AI Language
See also
\nGuidance for integration and responsible
use with Azure AI Language
06/24/2025
Microsoft wants to help you responsibly develop and deploy solutions that use Azure AI
Language. We are taking a principled approach to upholding personal agency and dignity by
considering the fairness, reliability & safety, privacy & security, inclusiveness, transparency, and
human accountability of our AI systems. These considerations are in line with our commitment
to developing Responsible AI.
This article discusses Azure AI Language features and the key considerations for making use of
this technology responsibly. Consider the following factors when you decide how to use and
implement AI-powered products and features.
When you're getting ready to deploy AI-powered products or features, the following activities
help to set you up for success:
Understand what it can do: Fully assess the capabilities of any AI model you are using to
understand its capabilities and limitations. Understand how it will perform in your
particular scenario and context.
Test with real, diverse data: Understand how your system will perform in your scenario by
thoroughly testing it with real life conditions and data that reflects the diversity in your
users, geography and deployment contexts. Small datasets, synthetic data and tests that
don't reflect your end-to-end scenario are unlikely to sufficiently represent your
production performance.
Respect an individual's right to privacy: Only collect data and information from
individuals for lawful and justifiable purposes. Only use data and information that you
have consent to use for this purpose.
Legal review: Obtain appropriate legal advice to review your solution, particularly if you
will use it in sensitive or high-risk applications. Understand what restrictions you might
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
General guidelines
\nneed to work within and your responsibility to resolve any issues that might come up in
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
\nHow to use Text Analytics for health
06/21/2025
Text Analytics for health can be used to extract and label relevant medical information from
unstructured texts such as doctors' notes, discharge summaries, clinical documents, and
electronic health records. The service performs named entity recognition, relation extraction,
entity linking
, and assertion detection to uncover insights from the input text. For
information on the returned confidence scores, see the transparency note.
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
\nThere are two ways to call the service:
A Docker container (synchronous)
Using the web-based API and client libraries (asynchronous)
To use Text Analytics for health, you submit raw unstructured text for analysis and handle the
API output in your application. Analysis is performed as-is, with no additional customization to
the model used on your data. There are two ways to use Text Analytics for health:
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
Integrate Text Analytics for health into your applications using the REST API, or
the client library available in a variety of languages. For more information, see the
Text Analytics for health quickstart.
Docker container
Use the available Docker container to deploy this feature on-premises. These
docker containers enable you to bring the service closer to your data for
compliance, security, or other operational reasons.
The Text Analytics for health supports English in addition to multiple languages that are
currently in preview. You can use the hosted API or deploy the API in a container, as detailed
under Text Analytics for health languages support.
To send an API request, you need your Language resource endpoint and key.
If you want to test out the feature without writing any code, use Azure AI Foundry
.
Development options
ﾉ
Expand table
Input languages
Submitting data
７ Note
\nAnalysis is performed upon receipt of the request. If you send a request using the REST API or
client library, the results are returned asynchronously. If you're using the Docker container, they
are returned synchronously.
When using this feature asynchronously, the API results are available for 24 hours from the
time the request was ingested, and is indicated in the response. After this time period, the
results are purged and are no longer available for retrieval.
Fast Healthcare Interoperability Resources (FHIR) is the health industry communication
standard developed by the Health Level Seven International (HL7) organization. The standard
defines the data formats (resources) and API structure for exchanging electronic healthcare
data. To receive your result using the FHIR structure, you must send the FHIR version in the API
request body.
Parameter Name
Type
Value
fhirVersion
string
4.0.1
Depending on your API request, and the data you submit to the Text Analytics for health, you
get:
Named entity recognition is used to perform a semantic extraction of words and phrases
mentioned from unstructured text that are associated with any of the supported entity
types, such as diagnosis, medication name, symptom/sign, or age.
You can find the key and endpoint for your Language resource on the Azure portal. They
are located on the resource's Key and endpoint page, under resource management.
Submitting a Fast Healthcare Interoperability
Resources (FHIR) request
ﾉ
Expand table
Getting results from the feature
Named Entity Recognition
\nFor information on the size and number of requests you can send per minute and second, see
the service limits article.
Text Analytics for health overview
Text Analytics for health entity categories
Service and data limits
See also
\n![Image](images/page1319_image1.png)
\nUse Text Analytics for health containers
06/21/2025
Containers enable you to host the Text Analytics for health API on your own infrastructure. If
you have security or data governance requirements that can't be fulfilled by calling Text
Analytics for health remotely, then containers might be a good option.
If you don't have an Azure subscription, create a free account
 before you begin.
You must meet the following prerequisites before using Text Analytics for health containers. If
you don't have an Azure subscription, create a free account
 before you begin.
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
clipboard ＝ icon appears. Copy and use the endpoint where needed.
Prerequisites
Gather required parameters
Endpoint URI