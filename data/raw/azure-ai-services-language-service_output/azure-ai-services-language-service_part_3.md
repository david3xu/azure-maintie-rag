Region
Authoring
Prediction
China East 2
✓
✓
China North 2
✓
East Asia
✓
East US
✓
✓
East US 2
✓
✓
France Central
✓
Germany West Central
✓
Italy North
✓
Japan East
✓
Japan West
✓
Jio India Central
✓
Jio India West
✓
Korea Central
✓
North Central US
✓
North Europe
✓
✓
Norway East
✓
Qatar Central
✓
South Africa North
✓
South Central US
✓
✓
Southeast Asia
✓
Sweden Central
✓
Switzerland North
✓
✓
UAE North
✓
UK South
✓
✓
UK West
✓
West Central US
✓
\nRegion
Authoring
Prediction
West Europe
✓
✓
West US
✓
West US 2
✓
✓
West US 3
✓
✓
Custom named entity recognition is only available in some Azure regions. Some regions are
available for both authoring and prediction, while other regions are prediction only. Language
resources in authoring regions allow you to create, edit, train, and deploy your projects.
Language resources in prediction regions allow you to get predictions from a deployment.
Region
Authoring
Prediction
Australia East
✓
✓
Brazil South
✓
Canada Central
✓
✓
Canada East
✓
Central India
✓
✓
Central US
✓
East Asia
✓
East US
✓
✓
East US 2
✓
✓
France Central
✓
Germany West Central
✓
Japan East
✓
Japan West
✓
Jio India Central
✓
Custom named entity recognition
ﾉ
Expand table
\nRegion
Authoring
Prediction
Jio India West
✓
Korea Central
✓
North Central US
✓
North Europe
✓
✓
Norway East
✓
Qatar Central
✓
South Africa North
✓
South Central US
✓
✓
Southeast Asia
✓
Sweden Central
✓
Switzerland North
✓
✓
UAE North
✓
UK South
✓
✓
UK West
✓
West Central US
✓
West Europe
✓
✓
West US
✓
West US 2
✓
✓
West US 3
✓
✓
Custom text classification is only available in some Azure regions. Some regions are available
for both authoring and prediction, while other regions are prediction only. Language
resources in authoring regions allow you to create, edit, train, and deploy your projects.
Language resources in prediction regions allow you to get predictions from a deployment.
Custom text classification
ﾉ
Expand table
\nRegion
Authoring
Prediction
Australia East
✓
✓
Brazil South
✓
Canada Central
✓
✓
Canada East
✓
Central India
✓
✓
Central US
✓
East Asia
✓
East US
✓
✓
East US 2
✓
✓
France Central
✓
Germany West Central
✓
Japan East
✓
Japan West
✓
Jio India Central
✓
Jio India West
✓
Korea Central
✓
North Central US
✓
North Europe
✓
✓
Norway East
✓
Qatar Central
✓
South Africa North
✓
South Central US
✓
✓
Southeast Asia
✓
Sweden Central
✓
Switzerland North
✓
✓
UAE North
✓
\nRegion
Authoring
Prediction
UK South
✓
✓
UK West
✓
West Central US
✓
West Europe
✓
✓
West US
✓
West US 2
✓
✓
West US 3
✓
✓
Region
Text abstractive summarization
Conversation summarization
US Gov Virginia
✓
✓
US Gov Arizona
✓
✓
Australia East
✓
✓
Canada Central
✓
✓
Central US
✓
✓
China North 3
✓
✓
East US
✓
✓
East US 2
✓
✓
France Central
✓
✓
Germany West Central
✓
✓
Italy North
✓
✓
Japan East
✓
✓
North Central US
✓
✓
North Europe
✓
✓
South Central US
✓
✓
Summarization
ﾉ
Expand table
\nRegion
Text abstractive summarization
Conversation summarization
South UK
✓
✓
Southeast Asia
✓
✓
Switzerland North
✓
✓
USNat East
✓
✓
USNat West
✓
✓
USSec East
✓
✓
USSec West
✓
✓
West Europe
✓
✓
West US
✓
✓
West US 2
✓
✓
Language support
Quotas and limits
Next steps
\nService limits for Azure AI Language
05/23/2025
Use this article to find the limits for the size, and rates that you can send data to the following
features of the language service.
Named Entity Recognition (NER)
Personally Identifiable Information (PII) detection
Key phrase extraction
Entity linking
Text Analytics for health
Sentiment analysis and opinion mining
Language detection
When using features of the Language service, keep the following information in mind:
Pricing is independent of data or rate limits. Pricing is based on the number of text
records you send to the API, and is subject to your Language resource's pricing details
.
A text record is measured as 1000 characters.
Data and rate limits are based on the number of documents you send to the API. If you
need to analyze larger documents than the limit allows, you can break the text into
smaller chunks of text before sending them to the API.
A document is a single string of text characters.
The following limit specifies the maximum number of characters that can be in a single
document.
７ Note
This article only describes the limits for preconfigured features in Azure AI Language: To
see the service limits for customizable features, see the following articles:
Custom classification
Custom NER
Conversational language understanding
Question answering
Maximum characters per document
\nFeature
Value
Text Analytics for health
125,000 characters as measured by StringInfo.LengthInTextElements.
All other preconfigured
features (synchronous)
5,120 as measured by StringInfo.LengthInTextElements. If you need to
submit larger documents, consider using the feature asynchronously.
All other preconfigured
features (asynchronous)
125,000 characters across all submitted documents, as measured by
StringInfo.LengthInTextElements (maximum of 25 documents).
If a document exceeds the character limit, the API behaves differently depending on how
you're sending requests.
If you're sending requests synchronously:
The API doesn't process documents that exceed the maximum size, and returns an invalid
document error for it. If an API request has multiple documents, the API continues
processing them if they are within the character limit.
If you're sending requests asynchronously:
The API rejects the entire request and returns a 400 bad request  error if any document
within it exceeds the maximum size.
The following limit specifies the maximum size of documents contained in the entire request.
Feature
Value
All preconfigured features
1 MB
Exceeding the following document limits generates an HTTP 400 error code.
ﾉ
Expand table
Maximum request size
ﾉ
Expand table
Maximum documents per request
７ Note
\nFeature
Max Documents Per Request
Conversation summarization
1
Language Detection
1000
Sentiment Analysis
10
Opinion Mining
10
Key Phrase Extraction
10
Named Entity Recognition (NER)
5
Personally Identifying Information (PII)
detection
5
Document summarization
25
Entity Linking
5
Text Analytics for health
25 for the web-based API, 1000 for the container. (125,000
characters in total)
Your rate limit varies with your pricing tier
. These limits are the same for both versions of the
API. These rate limits don't apply to the Text Analytics for health container, which doesn't have
a set rate limit.
Tier
Requests per second
Requests per minute
S / Multi-service
1000
1000
S0 / F0
100
300
Requests rates are measured for each feature separately. You can send the maximum number
of requests for your pricing tier to each feature, at the same time. For example, if you're in the
When sending asynchronous API requests, you can send a maximum of 25 documents per
request.
ﾉ
Expand table
Rate limits
ﾉ
Expand table
\nS  tier and send 1000 requests at once, you wouldn't be able to send another request for 59
seconds.
What is Azure AI Language
Pricing details
See also