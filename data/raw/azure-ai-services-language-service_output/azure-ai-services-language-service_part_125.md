false positives (neutral or positive text being recognized as negative sentiment), but fewer false
negatives (negative text not recognized as negative sentiment). For example, you might want
to read all product feedback that has some potential negative sentiment for ideas for product
improvement. In that case, you could use the negative sentiment score only and set a lower
threshold. This may lead to extra work because you'd end up reading some reviews that aren't
negative, but you're more likely to identify opportunities for improvement. If it is more
important for your system to recognize only true negative text, you can use a higher threshold
or use the overall sentiment label. For example, you may want to respond to product reviews
that are negative. If you want to minimize the work to read and respond to negative reviews,
you could only use the overall sentiment prediction and ignore the individual sentiment scores.
While there may be some negative sentiment predicted that you miss, you're likely to get most
of the truly negative reviews. Threshold values may not have consistent behavior across
scenarios. Therefore, it is critical that you test your system with real data that it will process in
production.
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for Health
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
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
\nHow to: Use Sentiment analysis and
Opinion Mining
05/23/2025
Sentiment analysis and opinion mining are two ways of detecting positive and negative
sentiment. Using sentiment analysis, you can get sentiment labels (such as "negative" "neutral"
and "positive") and confidence scores at the sentence and document-level. Opinion Mining
provides granular information about the opinions related to words (such as the attributes of
products or services) in the text.
Sentiment Analysis applies sentiment labels to text, which are returned at a sentence and
document level, with a confidence score for each.
The labels are positive, negative, and neutral. At the document level, the mixed sentiment label
also can be returned. The sentiment of the document is determined below:
Sentence sentiment
Returned document
label
At least one positive  sentence is in the document. The rest of the sentences
are neutral .
positive
At least one negative  sentence is in the document. The rest of the sentences
are neutral .
negative
At least one negative  sentence and at least one positive  sentence are in the
document.
mixed
All sentences in the document are neutral .
neutral
Confidence scores range from 1 to 0. Scores closer to 1 indicate a higher confidence in the
label's classification, while lower scores indicate lower confidence. For each document or each
sentence, the predicted scores associated with the labels (positive, negative, and neutral) add
up to 1. For more information, see the Responsible AI transparency note.
Sentiment Analysis
ﾉ
Expand table
Opinion Mining
\nOpinion Mining is a feature of Sentiment Analysis. Also known as Aspect-based Sentiment
Analysis in Natural Language Processing (NLP), this feature provides more granular information
about the opinions related to attributes of products or services in text. The API surfaces
opinions as a target (noun or verb) and an assessment (adjective).
For example, if a customer leaves feedback about a hotel such as "The room was great, but the
staff was unfriendly.", Opinion Mining will locate targets (aspects) in the text, and their
associated assessments (opinions) and sentiments. Sentiment Analysis might only report a
negative sentiment.
If you're using the REST API, to get Opinion Mining in your results, you must include the
opinionMining=true  flag in a request for sentiment analysis. The Opinion Mining results will be
included in the sentiment analysis response. Opinion mining is an extension of Sentiment
Analysis and is included in your current pricing tier
.
To use sentiment analysis, you submit raw unstructured text for analysis and handle the API
output in your application. Analysis is performed as-is, with no additional customization to the
model used on your data. There are two ways to use sentiment analysis:
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
Integrate sentiment analysis into your applications using the REST API, or the
client library available in a variety of languages. For more information, see the
sentiment analysis quickstart.
Docker container
Use the available Docker container to deploy this feature on-premises. These
docker containers enable you to bring the service closer to your data for
compliance, security, or other operational reasons.

Development options
ﾉ
Expand table
\n![Image](images/page1247_image1.png)
\nBy default, sentiment analysis will use the latest available AI model on your text. You can also
configure your API requests to use a specific model version.
When you submit documents to be processed by sentiment analysis, you can specify which of
the supported languages they're written in. If you don't specify a language, sentiment analysis
will default to English. The API may return offsets in the response to support different
multilingual and emoji encodings.
Sentiment analysis and opinion mining produce a higher-quality result when you give it smaller
amounts of text to work on. This is opposite from some features, like key phrase extraction
which performs better on larger blocks of text.
To send an API request, you'll need your Language resource endpoint and key.
Analysis is performed upon receipt of the request. Using the sentiment analysis and opinion
mining features synchronously is stateless. No data is stored in your account, and results are
returned immediately in the response.
When using this feature asynchronously, the API results are available for 24 hours from the
time the request was ingested, and is indicated in the response. After this time period, the
results are purged and are no longer available for retrieval.
Determine how to process the data (optional)
Specify the sentiment analysis model
Input languages
Submitting data
７ Note
You can find the key and endpoint for your Language resource on the Azure portal. They
will be located on the resource's Key and endpoint page, under resource management.
Getting sentiment analysis and opinion mining
results
\nWhen you receive results from the API, the order of the returned key phrases is determined
internally, by the model. You can stream the results to an application, or save the output to a
file on the local system.
Sentiment analysis returns a sentiment label and confidence score for the entire document, and
each sentence within it. Scores closer to 1 indicate a higher confidence in the label's
classification, while lower scores indicate lower confidence. A document can have multiple
sentences, and the confidence scores within each document or sentence add up to 1.
Opinion Mining will locate targets (nouns or verbs) in the text, and their associated assessment
(adjective). For example, the sentence "The restaurant had great food and our server was
friendly" has two targets: food and server. Each target has an assessment. For example, the
assessment for food would be great, and the assessment for server would be friendly.
The API returns opinions as a target (noun or verb) and an assessment (adjective).
For information on the size and number of requests you can send per minute and second, see
the service limits article.
Sentiment analysis and opinion mining overview
Service and data limits
See also
\nInstall and run Sentiment Analysis
containers
06/30/2025
Containers enable you to host the Sentiment Analysis API on your own infrastructure. If you
have security or data governance requirements that can't be fulfilled by calling Sentiment
Analysis remotely, then containers might be a good option.
If you don't have an Azure subscription, create a free account
 before you begin.
You must meet the following prerequisites before using Sentiment Analysis containers. If you
don't have an Azure subscription, create a free account
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