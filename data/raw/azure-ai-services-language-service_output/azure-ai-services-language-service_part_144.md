2. Make the following changes in the command where needed:
Replace the value your-language-resource-key  with your key.
Replace the first part of the request URL your-language-resource-endpoint
with your endpoint URL.
3. Open a command prompt window (for example: BASH).
4. Paste the command from the text editor into the command prompt window, then
run the command.
5. Get the operation-location  from the response header. The value looks similar to
the following URL:
HTTP
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
      "taskName": "Text Abstractive Summarization Task 1",
    }
  ]
}
'
https://<your-language-resource-endpoint>/language/analyze-
text/jobs/12345678-1234-1234-1234-12345678?api-version=2022-10-01-preview
\n6. To get the results of the request, use the following cURL command. Be sure to
replace <my-job-id>  with the numerical ID value you received from the previous
operation-location  response header:
Bash
JSON
curl -X GET https://<your-language-resource-endpoint>/language/analyze-
text/jobs/<my-job-id>?api-version=2022-10-01-preview \
-H "Content-Type: application/json" \
-H "Ocp-Apim-Subscription-Key: <your-language-resource-key>"
Abstractive text summarization example JSON response
{
    "jobId": "cd6418fe-db86-4350-aec1-f0d7c91442a6",
    "lastUpdateDateTime": "2022-09-08T16:45:14Z",
    "createdDateTime": "2022-09-08T16:44:53Z",
    "expirationDateTime": "2022-09-09T16:44:53Z",
    "status": "succeeded",
    "errors": [],
    "displayName": "Text Abstractive Summarization Task Example",
    "tasks": {
        "completed": 1,
        "failed": 0,
        "inProgress": 0,
        "total": 1,
        "items": [
            {
                "kind": "AbstractiveSummarizationLROResults",
                "taskName": "Text Abstractive Summarization Task 1",
                "lastUpdateDateTime": "2022-09-08T16:45:14.0717206Z",
                "status": "succeeded",
                "results": {
                    "documents": [
                        {
                            "summaries": [
                                {
                                    "text": "Microsoft is taking a more 
holistic, human-centric approach to AI. We've developed a joint 
representation to create more powerful AI that can speak, hear, see, and 
understand humans better. We've achieved human performance on benchmarks in 
conversational speech recognition, machine translation, ...... and image 
captions.",
                                    "contexts": [
                                        {
                                            "offset": 0,
                                            "length": 247
                                        }
\nparameter
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
For this project, you need a source document uploaded to your source container. You
can download our Microsoft Word sample document
 or Adobe PDF
 for this
quickstart. The source language is English.
1. Using your preferred editor or IDE, create a new directory for your app named
native-document .
                                    ]
                                }
                            ],
                            "id": "1"
                        }
                    ],
                    "errors": [],
                    "modelVersion": "latest"
                }
            }
        ]
    }
}
ﾉ
Expand table
Summarization sample document
Build the POST request
\n2. Create a new json file called document-summarization.json in your native-
document directory.
3. Copy and paste the Document Summarization request sample into your document-
summarization.json  file. Replace {your-source-container-SAS-URL}  and {your-
target-container-SAS-URL}  with values from your Azure portal Storage account
containers instance:
Request sample
JSON
Before you run the POST request, replace {your-language-resource-endpoint}  and
{your-key}  with the endpoint value from your Azure portal Language resource instance.
  {
  "tasks": [
    {
      "kind": "ExtractiveSummarization",
      "parameters": {
        "sentenceCount": 6
      }
    }
  ],
  "analysisInput": {
    "documents": [
      {
        "source": {
          "location": "{your-source-blob-SAS-URL}"
        },
        "targets": {
          "location": "{your-target-container-SAS-URL}"
        }
      }
    ]
  }
}
Run the POST request
） Important
Remember to remove the key from your code when you're done, and never post it
publicly. For production, use a secure way of storing and accessing your credentials
like Azure Key Vault. For more information, see Azure AI services security.
\nPowerShell
PowerShell
command prompt / terminal
Bash
HTTP
You receive a 202 (Success) response that includes a read-only Operation-Location
header. The value of this header contains a jobId that can be queried to get the status of
the asynchronous operation and retrieve the results using a GET request:
 cmd /c curl "{your-language-resource-endpoint}/language/analyze-
documents/jobs?api-version=2024-11-15-preview" -i -X POST --header "Content-
Type: application/json" --header "Ocp-Apim-Subscription-Key: {your-key}" --
data "@document-summarization.json"
curl -v -X POST "{your-language-resource-endpoint}/language/analyze-
documents/jobs?api-version=2024-11-15-preview" --header "Content-Type: 
application/json" --header "Ocp-Apim-Subscription-Key: {your-key}" --data 
"@document-summarization.json"
Sample response:
HTTP/1.1 202 Accepted
Content-Length: 0
operation-location: https://{your-language-resource-
endpoint}/language/analyze-documents/jobs/f1cc29ff-9738-42ea-afa5-
98d2d3cabf94?api-version=2024-11-15-preview
apim-request-id: e7d6fa0c-0efd-416a-8b1e-1cd9287f5f81
x-ms-region: West US 2
Date: Thu, 25 Jan 2024 15:12:32 GMT
POST response (jobId)
Get analyze results (GET request)
\n![Image](images/page1435_image1.png)
\n1. After your successful POST request, poll the operation-location header returned in
the POST request to view the processed data.
2. Here's the structure of the GET request:
Bash
3. Before you run the command, make these changes:
Replace {jobId} with the Operation-Location header from the POST response.
Replace {your-language-resource-endpoint} and {your-key} with the values
from your Language service instance in the Azure portal.
PowerShell
Bash
You receive a 200 (Success) response with JSON output. The status field indicates the
result of the operation. If the operation isn't complete, the value of status is "running" or
"notStarted", and you should call the API again, either manually or through a script. We
recommend an interval of one second or more between calls.
JSON
GET {cognitive-service-endpoint}/language/analyze-
documents/jobs/{jobId}?api-version=2024-11-15-preview
Get request
    cmd /c curl "{your-language-resource-endpoint}/language/analyze-
documents/jobs/{jobId}?api-version=2024-11-15-preview" -i -X GET --header 
"Content-Type: application/json" --header "Ocp-Apim-Subscription-Key: {your-
key}"
    curl -v -X GET "{your-language-resource-endpoint}/language/analyze-
documents/jobs/{jobId}?api-version=2024-11-15-preview" --header "Content-
Type: application/json" --header "Ocp-Apim-Subscription-Key: {your-key}"
Examine the response
Sample response
\nUpon successful completion:
The analyzed documents can be found in your target container.
The successful POST method returns a 202 Accepted  response code indicating that
the service created the batch request.
{
  "jobId": "f1cc29ff-9738-42ea-afa5-98d2d3cabf94",
  "lastUpdatedDateTime": "2024-01-24T13:17:58Z",
  "createdDateTime": "2024-01-24T13:17:47Z",
  "expirationDateTime": "2024-01-25T13:17:47Z",
  "status": "succeeded",
  "errors": [],
  "tasks": {
    "completed": 1,
    "failed": 0,
    "inProgress": 0,
    "total": 1,
    "items": [
      {
        "kind": "ExtractiveSummarizationLROResults",
        "lastUpdateDateTime": "2024-01-24T13:17:58.33934Z",
        "status": "succeeded",
        "results": {
          "documents": [
            {
              "id": "doc_0",
              "source": {
                "kind": "AzureBlob",
                "location": "https://myaccount.blob.core.windows.net/sample-
input/input.pdf"
              },
              "targets": [
                {
                  "kind": "AzureBlob",
                  "location": 
"https://myaccount.blob.core.windows.net/sample-output/df6611a3-fe74-44f8-
b8d4-58ac7491cb13/ExtractiveSummarization-0001/input.result.json"
                }
              ],
              "warnings": []
            }
          ],
          "errors": [],
          "modelVersion": "2023-02-01-preview"
        }
      }
    ]
  }
}
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
The POST request also returned response headers including Operation-Location
that provides a value used in subsequent GET requests.
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other
resources associated with it.
Azure portal
Azure CLI
Clean up resources
Next steps
Document summarization overview
Yes
No
\nUse summarization Docker containers on-
premises
06/21/2025
Containers enable you to host the Summarization API on your own infrastructure. If you have
security or data governance requirements that can't be fulfilled by calling Summarization
remotely, then containers might be a good option.
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
. For disconnected
containers, the DC0 tier is required.
Three primary parameters for all Azure AI containers are required. The Microsoft Software
License Terms must be present with a value of accept. An Endpoint URI and API key are also
needed.
The {ENDPOINT_URI}  value is available on the Azure portal Overview page of the corresponding
Azure AI services resource. Go to the Overview page, hover over the endpoint, and a Copy to
clipboard ＝ icon appears. Copy and use the endpoint where needed.
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
\n![Image](images/page1440_image1.png)

![Image](images/page1440_image2.png)