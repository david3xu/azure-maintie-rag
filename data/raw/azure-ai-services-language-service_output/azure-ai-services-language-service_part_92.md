A more detailed tutorial can be found in the “Adapting PII to your domain” how-to guide.
Analysis is performed upon receipt of the request. Using the PII detection feature
synchronously is stateless. No data is stored in your account, and results are returned
immediately in the response.
When using this feature asynchronously, the API results are available for 24 hours from the
time the request was ingested, and is indicated in the response. After this time period, the
results are purged and are no longer available for retrieval.
When you get results from PII detection, you can stream the results to an application or save
the output to a file on the local system. The API response includes recognized entities,
including their categories and subcategories, and confidence scores. The text string with the PII
entities redacted is also returned.
For information on the size and number of requests you can send per minute and second, see
the service limits article.
Personally Identifying Information (PII) overview
Submitting data
Getting PII results
Service and data limits
Next steps
\nDetect and redact Personally Identifying
Information in conversations
Article • 03/06/2025
Azure AI Language conversation PII API analyzes audio discourse to identify and redact
sensitive information (PII) using various predefined categories. This API works on both
transcribed text (referred to as transcripts) and chats. For transcripts, it also facilitates
the redaction of audio segments containing PII by providing the timing information for
those segments.
By default, this feature uses the latest available AI model on your input. You can also
configure your API requests to use a specific model version.
For more information, see the PII Language Support page. Currently the conversational
PII GA model only supports the English language. The preview model and API support
the same list languages as the other Language services.
The conversational PII API supports all Azure regions supported by the Language
service.
You can submit the input to the API as list of conversation items. Analysis is performed
upon receipt of the request. Because the API is asynchronous, there may be a delay
between sending an API request, and receiving the results. For information on the size
and number of requests you can send per minute and second, see the following data
limits.
When you use the async feature, the API results are available for 24 hours from the time
the request was ingested, and is indicated in the response. After this time period, the
Determine how to process the data (optional)
Specify the PII detection model
Language support
Region support
Submitting data
\nresults are purged and are no longer available for retrieval.
When you submit data to conversational PII, you can send one conversation (chat or
spoken) per request.
The API attempts to detect all the defined entity categories for a given conversation
input. If you want to specify which entities are detected and returned, use the optional
piiCategories  parameter with the appropriate entity categories.
For spoken transcripts, the entities detected are returned on the redactionSource
parameter value provided. Currently, the supported values for redactionSource  are
text , lexical , itn , and maskedItn  (which maps to Speech to text REST API's
display \ displayText , lexical , itn , and maskedItn  format respectively). Additionally,
for the spoken transcript input, this API also provides audio timing information to
empower audio redaction. For using the audioRedaction feature, use the optional
includeAudioRedaction  flag with true  value. The audio redaction is performed based on
the lexical input format.
When you get results from PII detection, you can stream the results to an application or
save the output to a file on the local system. The API response includes recognized
entities, including their categories and subcategories, and confidence scores. The text
string with the PII entities redacted is also returned.
1. Go to your resource overview page in the Azure portal
2. From the menu on the left side, select Keys and Endpoint. You need one of
the keys and the endpoint to authenticate your API requests.
3. Download and install the client library package for your language of choice:
７ Note
Conversation PII now supports 40,000 characters as document size.
Getting PII results
Examples
Client libraries (Azure SDK)
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Language
Package version
.NET
1.0.0
Python
1.0.0
4. For more information on the client and return object, see the following
reference documentation:
C#
Python
For information on the size and number of requests you can send per minute and
second, see the service limits article.
ﾉ
Expand table
Service and data limits
Yes
No
\nDetect and redact Personally Identifying
Information in native documents
(preview)
Article • 03/06/2025
Azure AI Language is a cloud-based service that applies Natural Language Processing
(NLP) features to text-based data. The native document support capability enables you
to send API requests asynchronously, using an HTTP POST request body to send your
data and HTTP GET request query string to retrieve the status results. Your processed
documents are located in your Azure Blob Storage target container.
A native document refers to the file format used to create the original document such as
Microsoft Word (docx) or a portable document file (pdf). Native document support
eliminates the need for text preprocessing before using Azure AI Language resource
capabilities. Currently, native document support is available for the following
capabilities:
Personally Identifiable Information (PII). The PII detection feature can identify,
categorize, and redact sensitive information in unstructured text. The
PiiEntityRecognition  API supports native document processing.
Document summarization. Document summarization uses natural language
processing to generate extractive (salient sentence extraction) or abstractive
(contextual word extraction) summaries for documents. Both
AbstractiveSummarization  and ExtractiveSummarization  APIs support native
document processing.
Applications use native file formats to create, save, or open native documents. Currently
PII and Document summarization capabilities supports the following native document
） Important
Azure AI Language public preview releases provide early access to features
that are in active development.
Features, approaches, and processes may change, before General Availability
(GA), based on user feedback.
Supported document formats
\nformats:
File type
File extension
Description
Text
.txt
An unformatted text document.
Adobe PDF
.pdf
A portable document file formatted document.
Microsoft Word
.docx
A Microsoft Word document file.
Supported file formats
Type
support and limitations
PDFs
Fully scanned PDFs aren't supported.
Text within images
Digital images with embedded text aren't supported.
Digital tables
Tables in scanned documents aren't supported.
Document Size
Attribute
Input limit
Total number of documents per request
≤ 20
Total content size per request
≤ 10 MB
Let's get started:
For this project, we use the cURL command line tool to make REST API calls.
ﾉ
Expand table
Input guidelines
ﾉ
Expand table
ﾉ
Expand table
Include native documents with an HTTP
request
７ Note
\nIf cURL isn't installed, here are installation links for your platform:
Windows
.
Mac or Linux
.
An active Azure account
. If you don't have one, you can create a free account
.
An Azure Blob Storage account
. You also need to create containers in your
Azure Blob Storage account for your source and target files:
Source container. This container is where you upload your native files for
analysis (required).
Target container. This container is where your analyzed files are stored
(required).
A single-service Language resource
 (not a multi-service Azure AI services
resource):
Complete the Language resource project and instance details fields as follows:
1. Subscription. Select one of your available Azure subscriptions.
2. Resource Group. You can create a new resource group or add your resource
to a preexisting resource group that shares the same lifecycle, permissions,
and policies.
3. Resource Region. Choose Global unless your business or application requires
a specific region. If you're planning on using a system-assigned managed
identity for authentication, choose a geographic region like West US.
4. Name. Enter the name you chose for your resource. The name you choose
must be unique within Azure.
5. Pricing tier. You can use the free pricing tier ( Free F0 ) to try the service, and
upgrade later to a paid tier for production.
6. Select Review + Create.
7. Review the service terms and select Create to deploy your resource.
8. After your resource successfully deploys, select Go to resource.
The cURL package is preinstalled on most Windows 10 and Windows 11 and
most macOS and Linux distributions. You can check the package version with
the following commands: Windows: curl.exe -V  macOS curl -V  Linux: curl
--version
\nRequests to the Language service require a read-only key and custom endpoint to
authenticate access.
1. If you created a new resource, after it deploys, select Go to resource. If you have
an existing language service resource, navigate directly to your resource page.
2. In the left rail, under Resource Management, select Keys and Endpoint.
3. You can copy and paste your key  and your language service instance endpoint
into the code samples to authenticate your request to the Language service. Only
one key is necessary to make an API call.
Create containers in your Azure Blob Storage account
 for source and target files.
Source container. This container is where you upload your native files for analysis
(required).
Target container. This container is where your analyzed files are stored (required).
Your Language resource needs granted access to your storage account before it can
create, read, or delete blobs. There are two primary methods you can use to grant
access to your storage data:
Shared access signature (SAS) tokens. User delegation SAS tokens are secured
with Microsoft Entra credentials. SAS tokens provide secure, delegated access to
resources in your Azure storage account.
Managed identity role-based access control (RBAC). Managed identities for Azure
resources are service principals that create a Microsoft Entra identity and specific
permissions for Azure managed resources.
For this project, we authenticate access to the source location  and target location
URLs with Shared Access Signature (SAS) tokens appended as query strings. Each token
is assigned to a specific blob (file).
Retrieve your key and language service endpoint
Create Azure Blob Storage containers
Authentication
\n![Image](images/page918_image1.png)
\nYour source container or blob must designate read and list access.
Your target container or blob must designate write and list access.
parameter
Description
-X POST <endpoint>
Specifies your Language resource endpoint for accessing
the API.
--header Content-Type:
application/json
The content type for sending JSON data.
--header "Ocp-Apim-Subscription-
Key:<key>
Specifies the Language resource key for accessing the
API.
-data
The JSON file containing the data you want to pass with
your request.
The following cURL commands are executed from a BASH shell. Edit these commands
with your own resource name, resource key, and JSON values. Try analyzing native
documents by selecting the Personally Identifiable Information (PII)  or Document
Summarization  code sample project:
For this quickstart, you need a source document uploaded to your source container.
You can download our Microsoft Word sample document
 or Adobe PDF
 for this
project. The source language is English.
1. Using your preferred editor or IDE, create a new directory for your app named
native-document .
 Tip
Since we're processing a single file (blob), we recommend that you delegate SAS
access at the blob level.
Request headers and parameters
ﾉ
Expand table
PII Sample document
Build the POST request
\n2. Create a new json file called pii-detection.json in your native-document directory.
3. Copy and paste the following Personally Identifiable Information (PII) request
sample into your pii-detection.json  file. Replace {your-source-container-SAS-
URL}  and {your-target-container-SAS-URL}  with values from your Azure portal
Storage account containers instance:
Request sample
JSON
The source location  value is the SAS URL for the source document (blob), not the
source container SAS URL.
{ 
    "displayName": "Document PII Redaction example", 
    "analysisInput": { 
        "documents": [ 
            { 
                "language": "en-US", 
                "id": "Output-1", 
                "source": { 
                    "location": "{your-source-blob-with-SAS-URL}" 
                }, 
                "target": { 
                    "location": "{your-target-container-with-SAS-URL}" 
                } 
            } 
        ] 
    }, 
    "tasks": [ 
        { 
            "kind": "PiiEntityRecognition", 
            "taskName": "Redact PII Task 1", 
            "parameters": { 
                "redactionPolicy": { 
                    "policyKind": "entityMask"  // Optional. Defines 
redactionPolicy; changes behavior based on value. Options: noMask, 
characterMask (default), and entityMask. 
                }, 
                "piiCategories": [ 
                    "Person", 
                    "Organization" 
                ], 
                "excludeExtractionData": false  // Default is false. If 
true, only the redacted document is stored, without extracted entities data. 
            } 
        } 
    ] 
}