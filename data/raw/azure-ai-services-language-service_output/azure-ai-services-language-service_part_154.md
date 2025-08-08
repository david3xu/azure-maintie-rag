3. Enter a name for your flow such as LanguageFlow . Then select Skip to continue without
choosing a trigger.
4. Under Triggers select Manually trigger a flow.


\n![Image](images/page1531_image1.png)

![Image](images/page1531_image2.png)
\n5. Select + New step to begin adding a Language service connector.
6. Under Choose an operation search for Azure AI Language. Then select Azure AI
Language. This will narrow down the list of actions to only those that are available for
Language.
7. Under Actions search for Named Entity Recognition, and select the connector.


\n![Image](images/page1532_image1.png)

![Image](images/page1532_image2.png)
\n8. Get the endpoint and key for your Language resource, which will be used for
authentication. You can find your key and endpoint by navigating to your resource in the
Azure portal
, and selecting Keys and Endpoint from the left side menu.
9. Once you have your key and endpoint, add it to the connector in Power Automate.


\n![Image](images/page1533_image1.png)

![Image](images/page1533_image2.png)
\n10. Add the data in the connector
11. From the top navigation menu, save the flow and select Test the flow. In the window that
appears, select Test.


７ Note
You will need deployment name and project name if you are using custom language
capability.
\n![Image](images/page1534_image1.png)

![Image](images/page1534_image2.png)
\n12. After the flow runs, you will see the response in the outputs field.
Triage incoming emails with custom text classification
Available Language service connectors


Next steps
\n![Image](images/page1535_image1.png)

![Image](images/page1535_image2.png)
\nNative document support for Azure AI
Language (preview)
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
formats:
） Important
Azure AI Language public preview releases provide early access to features
that are in active development.
Features, approaches, and processes can change, before General Availability
(GA), based on user feedback.
Supported document formats
\nFile type
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
parameter
Description
-X POST <endpoint>
Specifies your Language resource endpoint for accessing
the API.
--header Content-Type:
application/json
The content type for sending JSON data.
ﾉ
Expand table
Input guidelines
ﾉ
Expand table
ﾉ
Expand table
Request headers and parameters
ﾉ
Expand table
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
parameter
Description
--header "Ocp-Apim-Subscription-
Key:<key>
Specifies the Language resource key for accessing the
API.
-data
The JSON file containing the data you want to pass with
your request.
 
Related content
PII detection overview
Document Summarization overview
Yes
No
\nSAS tokens for your storage containers
Article • 03/05/2025
Learn to create user delegation, shared access signature (SAS) tokens, using the Azure
portal. User delegation SAS tokens are secured with Microsoft Entra credentials. SAS
tokens provide secure, delegated access to resources in your Azure storage account.
At a high level, here's how SAS tokens work:
Your application submits the SAS token to Azure Storage as part of a REST API
request.
The storage service verifies that the SAS is valid and then the request is authorized.
If the SAS token is deemed invalid, the request is declined, and the error code 403
(Forbidden) is returned.
Azure Blob Storage offers three resource types:
Storage accounts provide a unique namespace in Azure for your data.
Data storage containers are located in storage accounts and organize sets of
blobs (files, text, or images).
Blobs are located in containers and store text and binary data such as files, text,
and images.
 Tip
Role-based access control (managed identities) provide an alternate method for
granting access to your storage data without the need to include SAS tokens with
your HTTP requests.
Using managed identities grants access to any resource that supports
Microsoft Entra authentication, including your own applications.
Using managed identities replaces the requirement for you to include shared
access signature tokens (SAS) with your source and target URLs.
Using managed identities doesn't require an added cost in Azure.
） Important
\n![Image](images/page1539_image1.png)
\nTo get started, you need the following resources:
An active Azure account
. If you don't have one, you can create a free account
.
An Azure AI Language
 resource.
A standard performance Azure Blob Storage account
. You also need to create
containers to store and organize your files within your storage account. If you
don't know how to create an Azure storage account with a storage container,
follow these quickstarts:
Create a storage account. When you create your storage account, select
Standard performance in the Instance details > Performance field.
Create a container. When you create your container, set Public access level to
Container (anonymous read access for containers and files) in the New
Container window.
Go to the Azure portal
 and navigate to your container or a specific file as follows and
continue with these steps:
Workflow: Your storage account → containers → your container → your file
1. Right-click the container or file and select Generate SAS from the drop-down
menu.
2. Select Signing method → User delegation key.
3. Define Permissions by checking and/or clearing the appropriate check box:
Your source file must designate read and list access.
Your target file must designate write and list access.
SAS tokens are used to grant permissions to storage resources, and should be
protected in the same manner as an account key.
Operations that use SAS tokens should be performed only over an HTTPS
connection, and SAS URI s should only be distributed on a secure connection
such as HTTPS.
Prerequisites
Create SAS tokens in the Azure portal