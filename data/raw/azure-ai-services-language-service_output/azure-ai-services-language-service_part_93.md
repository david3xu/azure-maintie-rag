The redactionPolicy  possible values are UseRedactionCharacterWithRefId  (default)
or UseEntityTypeName . For more information, see PiiTask Parameters.
1. Here's the preliminary structure of the POST request:
Bash
2. Before you run the POST request, replace {your-language-resource-endpoint}  and
{your-key}  with the values from your Azure portal Language service instance.
PowerShell
PowerShell
command prompt / terminal
Bash
3. Here's a sample response:
HTTP
Run the POST request
   POST {your-language-endpoint}/language/analyze-documents/jobs?api-
version=2024-11-15-preview
） Important
Remember to remove the key from your code when you're done, and never
post it publicly. For production, use a secure way of storing and accessing
your credentials like Azure Key Vault. For more information, see Azure AI
services security.
   cmd /c curl "{your-language-resource-endpoint}/language/analyze-
documents/jobs?api-version=2024-11-15-preview" -i -X POST --header 
"Content-Type: application/json" --header "Ocp-Apim-Subscription-Key: 
{your-key}" --data "@pii-detection.json"
   curl -v -X POST "{your-language-resource-endpoint}/language/analyze-
documents/jobs?api-version=2024-11-15-preview" --header "Content-Type: 
application/json" --header "Ocp-Apim-Subscription-Key: {your-key}" --
data "@pii-detection.json"
\nYou receive a 202 (Success) response that includes a read-only Operation-Location
header. The value of this header contains a jobId that can be queried to get the status
of the asynchronous operation and retrieve the results using a GET request:
1. After your successful POST request, poll the operation-location header returned in
the POST request to view the processed data.
2. Here's the preliminary structure of the GET request:
Bash
3. Before you run the command, make these changes:
Replace {jobId} with the Operation-Location header from the POST response.
Replace {your-language-resource-endpoint} and {your-key} with the values
from your Language service instance in the Azure portal.
PowerShell
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
  GET {your-language-endpoint}/language/analyze-documents/jobs/{jobId}?
api-version=2024-11-15-preview
Get request
    cmd /c curl "{your-language-resource-endpoint}/language/analyze-
documents/jobs/{jobId}?api-version=2024-11-15-preview" -i -X GET --header 
\n![Image](images/page922_image1.png)
\nBash
You receive a 200 (Success) response with JSON output. The status field indicates the
result of the operation. If the operation isn't complete, the value of status is "running" or
"notStarted", and you should call the API again, either manually or through a script. We
recommend an interval of one second or more between calls.
JSON
"Content-Type: application/json" --header "Ocp-Apim-Subscription-Key: {your-
key}"
    curl -v -X GET "{your-language-resource-endpoint}/language/analyze-
documents/jobs/{jobId}?api-version=2024-11-15-preview" --header "Content-
Type: application/json" --header "Ocp-Apim-Subscription-Key: {your-key}"
Examine the response
Sample response
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
        "kind": "PiiEntityRecognitionLROResults",
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
\nUpon successful completion:
The analyzed documents can be found in your target container.
The successful POST method returns a 202 Accepted  response code indicating that
the service created the batch request.
The POST request also returned response headers including Operation-Location
that provides a value used in subsequent GET requests.
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other
resources associated with it.
Azure portal
Azure CLI
                  "kind": "AzureBlob",
                  "location": 
"https://myaccount.blob.core.windows.net/sample-output/df6611a3-fe74-44f8-
b8d4-58ac7491cb13/PiiEntityRecognition-0001/input.result.json"
                },
                {
                  "kind": "AzureBlob",
                  "location": 
"https://myaccount.blob.core.windows.net/sample-output/df6611a3-fe74-44f8-
b8d4-58ac7491cb13/PiiEntityRecognition-0001/input.docx"
                }
              ],
              "warnings": []
            }
          ],
          "errors": [],
          "modelVersion": "2023-09-01"
        }
      }
    ]
  }
}
Clean up resources
Next steps
PII detection overview
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Yes
No
\nAdapting Personally Identifying
Information (PII) to your domain
Article • 05/19/2025
To accommodate and adapt to a customer’s custom vocabulary used to identify entities (also
known as the "context"), the entitySynonyms  feature allows customers to define their own
synonyms for specific entity types. The goal of this feature is to help detect entities in contexts
that the model isn't familiar with but are used in the customer’s inputs by ensuring that the
customer’s unique terms are recognized and correctly associated during the detection process.
This adapts the prebuilt PII service which is trained to detect entities based on general domain
text which may not match a customer’s custom input vocabulary, such as writing "BAN" instead
of "InternationalBankAccountNumber".
This means PII detection can catch sensitive information even when it’s written in different
styles, slang, or casual language. That makes the system better at protecting privacy in real-
world situations.
We strongly recommend that customers first test the quality of predictions without introducing
synonyms and only use them if the model isn't performing well. For example, "Org" may be
something that the model already understands as "organization" and there's no need to use
the Synonym feature.
After testing the service on their data, customers can use entitySynonyms  to:
Specify particular entities within the prebuilt service for which there are custom synonym
context words in their input vocabulary.
List the custom synonyms.
Specify the language of each synonym.
JSON
Overview
API Schema for the 'entitySynoyms' parameter
{ 
    "parameter":  
    "entitySynonyms": [  
        { 
            "entityType": "InternationalBankAccountNumber", 
            "synonyms": [ {"synonym": "BAN", "language": "en"} ] 
\n1. Synonyms must be restricted to phrases that directly refer to the type, and preserve
semantic correctness. For example, for the entity type InternationalBankAccountNumber , a
valid synonym could be "Financial Account Number" or "FAN". But, the word "deposit"
though may be associated with type, as it doesn't directly have a meaning of a bank
account number and therefore shouldn't be used.
2. Synonyms should be country agnostic. For example, "German passport" wouldn't be
helpful to include.
3. Synonyms can't be reused for more than one entity type.
4. This synonym recognition feature only accepts a subset of entity types supported by the
service. The supported entity types and example synonyms include:
Supported
entity type
Entity Type
Example synonyms
ABA Routing
Number
ABARoutingNumber
Routing transit number (RTN)
Address
Address
My place is
Age
Age
Years old, age in years, current age, person’s
age, biological age
Bank Account
Number
BankAccountNumber
Bank acct no., savings account number,
checking account number, financial account
number
Credit Card
Number
CreditCardNumber
Cc number, payment card number, credit acct
no.
Date
DateTime
Given date, specified date
Date of Birth
DateOfBirth
Birthday, DOB, birthdate
International
Bank Account
Number
InternationalBankingAccountNumber
IBAN, intl bank acct no.
Organization
Organization
company, business, firm, corporation, agency,
group, institution, entity, legal entity, party,
        } 
    ]
} 
Usage guidelines
ﾉ
Expand table
\nSupported
entity type
Entity Type
Example synonyms
respondent, plaintiff, defendant, jurisdiction,
partner, provider, facility, practice, network,
institution, enterprise, LLC, Inc, LLP,
incorporated, employer, brand, subsidiary
Person
Person
Name, individual, account holder
Person Type
PersonType
Role, title, position
Phone number
PhoneNumber
Landline, cell, mobile
Swift Code
SWIFTCode
SWIFT code, BIC (Bank Identifier Code), SWIFT
Identifier
The valueExclusionPolicy  option allows customers to adapt the PII service for scenarios where
customers prefer certain terms be undetected and redacted even if those terms fall into a PII
category they're interested in detected. For example, a police department might want personal
identifiers redacted in most cases except for terms like "police officer", "suspect", and "witness".
In the following example, customers can use the valueExclusionPolicy  option to specify a list
of values which they wouldn't like to be detected or redacted from the input text. In the
example below, if the user specifies the value "1 Microsoft Way, Redmond, WA 98052, US",
even if the Address entity is turned-on, this value isn't redacted or listed in the returned API
payload output.
A subset of the specified excluded value, such as "1 Microsoft Way" isn't excluded.
JSON
Customizing PII output by specifying values to
exclude
Input
{ 
  "kind": "PiiEntityRecognition", 
  "parameters": { 
    "modelVersion": "latest", 
    "redactionPolicy": { 
      "policyKind": "characterMask", 
      "redactionCharacter": "-" 
    }, 
    "valueExclusionPolicy": { 
\nJSON
      "caseSensitive": false, 
      "excludedValues": { 
        "1 Microsoft Way, Redmond, WA 98052", 
        "1045 La Avenida St, Mountain View, CA 94043" 
      } 
    } 
  }, 
  "analysisInput": { 
    "documents": [ 
      { 
        "id": "1", 
        "text": "The police and John Doe inspected the storage garages located at 
123 Main St, 1 Microsoft Way, Redmond, WA 98052, 456 Washington Blvd, Portland, 
OR, and 1045 La Avenida St, Mountain View, CA 94043" 
      } 
    ] 
  } 
} 
Output
{ 
    "kind": "PiiEntityRecognitionResults", 
    "results": { 
        "documents": [ 
            { 
                "redactedText": "The police and John Doe inspected the storage 
garages located at **********, 1 Microsoft Way, Redmond, WA 98052, 
********************************, and 1045 La Avenida St, Mountain View, CA 94043" 
                "id": "1", 
                "entities": [ 
                    { 
                        "text": "John Doe", 
                        "category": "Person", 
                        "offset": 16, 
                        "length": 5, 
                        "confidenceScore": 0.98 
                    } 
                ], 
                "warnings": [] 
            } 
        ], 
        "errors": [], 
        "modelVersion": "2021-01-15" 
    } 
} 
\nCustomers can now adapt the PII service’s detecting by specifying their own regex using a
regex recognition configuration file. See our container how-to guides for a tutorial on how to
install and run Personally Identifiable Information (PII) Detection containers.
Bash
UserRegexRuleFilePath  is the file path of the user defined regex rules.
JSON
Customizing PII detection using your own regex
(only available for Text PII container)
７ Note
This only available for the Text PII container
docker run --rm -it -p 5000:5000 --memory 8g --cpus 1 \ 
mcr.microsoft.com/azure-cognitive-services/textanalytics/pii:{IMAGE_TAG} \ 
Eula=accept \ 
Billing={ENDPOINT_URI} \ 
ApiKey={API_KEY} \ 
UserRegexRuleFilePath={REGEX_RULE_FILE_PATH} 
Regex recognition file format
[ 
    { 
      "name": "USSocialSecurityNumber", // category, type and tag to be returned. 
This name must be unique 
      "description": "Rule to identify USSocialSecurityNumber in text", // used to 
describe the category 
      "regexPatterns": [ // list of regex patterns to identify the entities 
        { 
          "id": "StrongSSNPattern", // id for the regex pattern 
          "pattern": "(?<!\\d)([0-9]{3}-[0-9]{2}-[0-9]{4}|[0-9]{3} [0-9]{2} [0-9]
{4}|[0-9]{3}.[0-9]{2}.[0-9]{4})(?!\\d)", // regex pattern to provide matches 
          "matchScore": 0.65, // score to assign if the regex matches 
          "locales": [ // list of languages valid for this regex 
            "en" 
         ] 
        }, 
        { 
          "id": "WeakSSNPattern", 
          "pattern": "(?<!\\d)([0-9]{9})(?!\\d)", 
          "matchScore": 0.55,