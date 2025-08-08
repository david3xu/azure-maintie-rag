For the sentenceCount  parameter, you can input a value 1-20 to indicate the desired
number of output sentences.
Here's is an example request:
Bash
        "text": "At Microsoft, we have been on a quest to advance AI beyond 
existing techniques, by taking a more holistic, human-centric approach to 
learning and understanding. As Chief Technology Officer of Azure AI 
services, I have been working with a team of amazing scientists and 
engineers to turn this quest into a reality. In my role, I enjoy a unique 
perspective in viewing the relationship among three attributes of human 
cognition: monolingual text (X), audio or visual sensory signals, (Y) and 
multilingual (Z). At the intersection of all three, there's magic—what we 
call XYZ-code as illustrated in Figure 1—a joint representation to create 
more powerful AI that can speak, hear, see, and understand humans better. We 
believe XYZ-code enables us to fulfill our long-term vision: cross-domain 
transfer learning, spanning modalities and languages. The goal is to have 
pretrained models that can jointly learn representations to support a broad 
range of downstream AI tasks, much in the way humans do today. Over the past 
five years, we have achieved human performance on benchmarks in 
conversational speech recognition, machine translation, conversational 
question answering, machine reading comprehension, and image captioning. 
These five breakthroughs provided us with strong signals toward our more 
ambitious aspiration to produce a leap in AI capabilities, achieving multi-
sensory and multilingual learning that is closer in line with how humans 
learn and understand. I believe the joint XYZ-code is a foundational 
component of this aspiration, if grounded with external knowledge sources in 
the downstream AI tasks."
      }
    ]
  },
  "tasks": [
    {
      "kind": "AbstractiveSummarization",
      "taskName": "Length controlled Abstractive Summarization",
          "parameters": {
          "sentenceLength": "short"
      }
    }
  ]
}
'
Using the sentenceCount parameter in extractive summarization
curl -i -X POST https://<your-language-resource-endpoint>/language/analyze-
text/jobs?api-version=2023-11-15-preview \
-H "Content-Type: application/json" \
-H "Ocp-Apim-Subscription-Key: <your-language-resource-key>" \
\nFor information on the size and number of requests you can send per minute and
second, see the service limits article.
-d \
' 
{
  "displayName": "Text Extractive Summarization Task Example",
  "analysisInput": {
    "documents": [
      {
        "id": "1",
        "language": "en",
        "text": "At Microsoft, we have been on a quest to advance AI beyond 
existing techniques, by taking a more holistic, human-centric approach to 
learning and understanding. As Chief Technology Officer of Azure AI 
services, I have been working with a team of amazing scientists and 
engineers to turn this quest into a reality. In my role, I enjoy a unique 
perspective in viewing the relationship among three attributes of human 
cognition: monolingual text (X), audio or visual sensory signals, (Y) and 
multilingual (Z). At the intersection of all three, there's magic—what we 
call XYZ-code as illustrated in Figure 1—a joint representation to create 
more powerful AI that can speak, hear, see, and understand humans better. We 
believe XYZ-code enables us to fulfill our long-term vision: cross-domain 
transfer learning, spanning modalities and languages. The goal is to have 
pretrained models that can jointly learn representations to support a broad 
range of downstream AI tasks, much in the way humans do today. Over the past 
five years, we have achieved human performance on benchmarks in 
conversational speech recognition, machine translation, conversational 
question answering, machine reading comprehension, and image captioning. 
These five breakthroughs provided us with strong signals toward our more 
ambitious aspiration to produce a leap in AI capabilities, achieving multi-
sensory and multilingual learning that is closer in line with how humans 
learn and understand. I believe the joint XYZ-code is a foundational 
component of this aspiration, if grounded with external knowledge sources in 
the downstream AI tasks."
      }
    ]
  },
"tasks": [
    {
      "kind": "ExtractiveSummarization",
      "taskName": "Length controlled Extractive Summarization",
      "parameters": {
          "sentenceCount": "5"
      }
    }
  ]
}
'
Service and data limits
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Summarization overview
See also
Yes
No
\nHow to use native document
summarization (preview)
Article • 03/06/2025
Azure AI Language is a cloud-based service that applies Natural Language Processing
(NLP) features to text-based data. Document summarization uses natural language
processing to generate extractive (salient sentence extraction) or abstractive (contextual
word extraction) summaries for documents. Both AbstractiveSummarization  and
ExtractiveSummarization  APIs support native document processing. A native document
refers to the file format used to create the original document such as Microsoft Word
(docx) or a portable document file (pdf). Native document support eliminates the need
for text preprocessing before using Azure AI Language resource capabilities. The native
document support capability enables you to send API requests asynchronously, using an
HTTP POST request body to send your data and HTTP GET request query string to
retrieve the status results. Your processed documents are located in your Azure Blob
Storage target container.
Applications use native file formats to create, save, or open native documents. Currently
PII and Document summarization capabilities supports the following native document
formats:
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
） Important
Azure AI Language public preview releases provide early access to features
that are in active development.
Features, approaches, and processes may change, before General Availability
(GA), based on user feedback.
Supported document formats
ﾉ
Expand table
\nSupported file formats
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
If cURL isn't installed, here are installation links for your platform:
Windows
.
Mac or Linux
.
An active Azure account
. If you don't have one, you can create a free account
.
Input guidelines
ﾉ
Expand table
ﾉ
Expand table
Include native documents with an HTTP
request
７ Note
The cURL package is preinstalled on most Windows 10 and Windows 11 and
most macOS and Linux distributions. You can check the package version with
the following commands: Windows: curl.exe -V  macOS curl -V  Linux: curl
--version
\nAn Azure Blob Storage account
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
Requests to the Language service require a read-only key and custom endpoint to
authenticate access.
1. If you created a new resource, after it deploys, select Go to resource. If you have
an existing language service resource, navigate directly to your resource page.
2. In the left rail, under Resource Management, select Keys and Endpoint.
Retrieve your key and language service endpoint
\n3. You can copy and paste your key  and your language service instance endpoint
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
Your source container or blob must designate read and list access.
Your target container or blob must designate write and list access.
The extractive summarization API uses natural language processing techniques to locate
key sentences in an unstructured text document. These sentences collectively convey
the main idea of the document.
Extractive summarization returns a rank score as a part of the system response along
with extracted sentences and their position in the original documents. A rank score is an
Create Azure Blob Storage containers
Authentication
\n![Image](images/page1427_image1.png)
\nindicator of how relevant a sentence is determined to be, to the main idea of a
document. The model gives a score between 0 and 1 (inclusive) to each sentence and
returns the highest scored sentences per request. For example, if you request a three-
sentence summary, the service returns the three highest scored sentences.
There's another feature in Azure AI Language, key phrase extraction, that can extract key
information. To decide between key phrase extraction and extractive summarization,
here are helpful considerations:
Key phrase extraction returns phrases while extractive summarization returns
sentences.
Extractive summarization returns sentences together with a rank score, and top
ranked sentences are returned per request.
Extractive summarization also returns the following positional information:
Offset: The start position of each extracted sentence.
Length: The length of each extracted sentence.
You submit documents to the API as strings of text. Analysis is performed upon receipt
of the request. Because the API is asynchronous, there might be a delay between
sending an API request, and receiving the results.
When you use this feature, the API results are available for 24 hours from the time the
request was ingested, and is indicated in the response. After this time period, the results
are purged and are no longer available for retrieval.
When you get results from language detection, you can stream the results to an
application or save the output to a file on the local system.
Here's an example of content you might submit for summarization, which is extracted
using the Microsoft blog article A holistic representation toward integrative AI
. This
article is only an example. The API can accept longer input text. For more information,
see data and service limits.
"At Microsoft, we have been on a quest to advance AI beyond existing techniques, by
taking a more holistic, human-centric approach to learning and understanding. As Chief
Technology Officer of Azure AI services, I have been working with a team of amazing
Determine how to process the data (optional)
Submitting data
Getting text summarization results
\nscientists and engineers to turn this quest into a reality. In my role, I enjoy a unique
perspective in viewing the relationship among three attributes of human cognition:
monolingual text (X), audio or visual sensory signals, (Y) and multilingual (Z). At the
intersection of all three, there's magic—what we call XYZ-code as illustrated in Figure 1—a
joint representation to create more powerful AI that can speak, hear, see, and understand
humans better. We believe XYZ-code enables us to fulfill our long-term vision: cross-
domain transfer learning, spanning modalities and languages. The goal is to have
pretrained models that can jointly learn representations to support a broad range of
downstream AI tasks, much in the way humans do today. Over the past five years, we have
achieved human performance on benchmarks in conversational speech recognition,
machine translation, conversational question answering, machine reading comprehension,
and image captioning. These five breakthroughs provided us with strong signals toward
our more ambitious aspiration to produce a leap in AI capabilities, achieving multi-sensory
and multilingual learning that is closer in line with how humans learn and understand. I
believe the joint XYZ-code is a foundational component of this aspiration, if grounded with
external knowledge sources in the downstream AI tasks."
The text summarization API request is processed upon receipt of the request by creating
a job for the API backend. If the job succeeded, the output of the API is returned. The
output is available for retrieval for 24 hours. After this time, the output is purged. Due to
multilingual and emoji support, the response might contain text offsets. For more
information, see how to process offsets.
When you use the preceding example, the API might return these summarized
sentences:
Extractive summarization:
"At Microsoft, we have been on a quest to advance AI beyond existing techniques,
by taking a more holistic, human-centric approach to learning and understanding."
"We believe XYZ-code enables us to fulfill our long-term vision: cross-domain
transfer learning, spanning modalities and languages."
"The goal is to have pretrained models that can jointly learn representations to
support a broad range of downstream AI tasks, much in the way humans do
today."
Abstractive summarization:
"Microsoft is taking a more holistic, human-centric approach to learning and
understanding. We believe XYZ-code enables us to fulfill our long-term vision:
cross-domain transfer learning, spanning modalities and languages. Over the past
five years, we have achieved human performance on benchmarks in."
\nYou can use text extractive summarization to get summaries of articles, papers, or
documents. To see an example, see the quickstart article.
You can use the sentenceCount  parameter to guide how many sentences are returned,
with 3  being the default. The range is from 1 to 20.
You can also use the sortby  parameter to specify in what order the extracted sentences
are returned - either Offset  or Rank , with Offset  being the default.
parameter
value
Description
Rank
Order sentences according to their relevance to the input document, as
decided by the service.
Offset
Keeps the original order in which the sentences appear in the input document.
The following example gets you started with text abstractive summarization:
1. Copy the following command into a text editor. The BASH example uses the \  line
continuation character. If your console or terminal uses a different line
continuation character, use that character instead.
Bash
Try text extractive summarization
ﾉ
Expand table
Try text abstractive summarization
curl -i -X POST https://<your-language-resource-endpoint>/language/analyze-
text/jobs?api-version=2023-04-01 \
-H "Content-Type: application/json" \
-H "Ocp-Apim-Subscription-Key: <your-language-resource-key>" \
-d \
' 
{
  "displayName": "Text Abstractive Summarization Task Example",
  "analysisInput": {
    "documents": [
      {
        "id": "1",
        "language": "en",
        "text": "At Microsoft, we have been on a quest to advance AI beyond 
existing techniques, by taking a more holistic, human-centric approach to 
learning and understanding. As Chief Technology Officer of Azure AI 
services, I have been working with a team of amazing scientists and