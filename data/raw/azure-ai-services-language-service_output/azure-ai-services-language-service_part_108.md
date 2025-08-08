5. Go to Networking tab of language resource and under the Allow access from, select the
Selected Networks and private endpoints option and select save.
This will establish a private endpoint connection between language resource and Azure AI
Search service instance. You can verify the Private endpoint connection on the Networking tab
of the Azure AI Search service instance. Once the whole operation is completed, you are good
to use your language resource with question answering enabled.
\n![Image](images/page1071_image1.png)

![Image](images/page1071_image2.png)
\nWe don't support changes to Azure AI Search service once you enable private access to
your language resources. If you change the Azure AI Search service via 'Features' tab after
you have enabled private access, the language resource will become unusable.
After establishing Private Endpoint Connection, if you switch Azure AI Search Service
Networking to 'Public', you won't be able to use the language resource. Azure Search
Service Networking needs to be 'Private' for the Private Endpoint Connection to work.
Follow these steps to restrict public access to custom question answering language resources.
Protect an AI Foundry resource from public access by configuring the virtual network.
After you restrict access to an AI Foundry resource based on virtual network, to browse projects
on Language Studio from your on-premises network or your local browser:
Grant access to on-premises network.
Grant access to your local browser/machine.
Add the public IP address of the machine under the Firewall section of the Networking
tab. By default portal.azure.com  shows the current browsing machine's public IP (select
this entry) and then select Save.
Support details
Restrict access to Azure AI Search resource
\n![Image](images/page1072_image1.png)
\n
\n![Image](images/page1073_image1.png)
\nAuthoring API
06/30/2025
The custom question answering Authoring API is used to automate common tasks like adding
new question answer pairs, as well as creating, publishing, and maintaining projects.
The current version of cURL
. Several command-line switches are used in this article,
which are noted in the cURL documentation
.
The commands in this article are designed to be executed in a Bash shell. These
commands will not always work in a Windows command prompt or in PowerShell without
modification. If you do not have a Bash shell installed locally, you can use the Azure Cloud
Shell's bash environment.
To create a project programmatically:
Variable
name
Value
ENDPOINT
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. Alternatively, you can find the value in Language Studio > question
answering > Deploy project > Get prediction URL. An example endpoint is:
https://southcentralus.api.cognitive.microsoft.com/ . If the prior example was your
endpoint in the code sample below, you would only need to add the region specific portion
of southcentral  as the rest of the endpoint path is already present.
API-KEY
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. You can use either Key1 or Key2. Always having two valid keys allows for
secure key rotation with zero downtime. Alternatively, you can find the value in Language
７ Note
Authoring functionality is available via the REST API and Authoring SDK (preview). This
article provides examples of using the REST API with cURL. For full documentation of all
parameters and functionality available consult the REST API reference content.
Prerequisites
Create a project
ﾉ
Expand table
\nVariable
name
Value
Studio > question answering > Deploy project > Get prediction URL. The key value is part
of the sample request.
NEW-
PROJECT-
NAME
The name for your new custom question answering project.
You can also adjust additional values like the project language, the default answer given when
no answer can be found that meets or exceeds the confidence threshold, and whether this
language resource will support multiple languages.
Bash
JSON
Example query
curl -X PATCH -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '{
      "description": "proj1 is a test project.",
      "language": "en",
      "settings": {
        "defaultAnswer": "No good match found for your question in the project."
      },
      "multilingualResource": true
    }
  }'  'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{NEW-PROJECT-NAME}?api-version=2021-10-01'
Example response
{
 "200": {
      "headers": {},
      "body": {
        "projectName": "proj1",
        "description": "proj1 is a test project.",
        "language": "en",
        "settings": {
          "defaultAnswer": "No good match found for your question in the project."
        },
        "multilingualResource": true,
        "createdDateTime": "2021-05-01T15:13:22Z",
        "lastModifiedDateTime": "2021-05-01T15:13:22Z",
        "lastDeployedDateTime": "2021-05-01T15:13:22Z"
\nTo delete a project programmatically:
Variable
name
Value
ENDPOINT
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. Alternatively you can find the value in Language Studio > question
answering > Deploy project > Get prediction URL. An example endpoint is:
https://southcentralus.api.cognitive.microsoft.com/ . If the prior example was your
endpoint in the code sample below, you would only need to add the region specific portion
of southcentral  as the rest of the endpoint path is already present.
API-KEY
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. You can use either Key1 or Key2. Always having two valid keys allows for
secure key rotation with zero downtime. Alternatively you can find the value in Language
Studio > question answering > Deploy project > Get prediction URL. The key value is part
of the sample request.
PROJECT-
NAME
The name of project you would like to delete.
Bash
A successful call to delete a project results in an Operation-Location  header being returned,
which can be used to check the status of the delete project job. In most our examples, we
haven't needed to look at the response headers and thus haven't been displaying them. To
retrieve the response headers our curl command uses -i . Without this parameter prior to the
endpoint address, the response to this command would appear empty as if no response
occurred.
      }
 }
}
Delete Project
ﾉ
Expand table
Example query
curl -X DELETE -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -i 
'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}?api-version=2021-10-01'
\nBash
If the project was already deleted or could not be found, you would receive a message like:
JSON
To check on the status of your delete project request:
Variable
name
Value
ENDPOINT
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. Alternatively, you can find the value in Language Studio > question
answering > Deploy project > Get prediction URL. An example endpoint is:
https://southcentralus.api.cognitive.microsoft.com/ . If this was your endpoint in the code
sample below, you would only need to add the region specific portion of southcentral  as the
rest of the endpoint path is already present.
Example response
HTTP/2 202
content-length: 0
operation-location: 
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/deletion-jobs/{JOB-ID-GUID}
x-envoy-upstream-service-time: 324
apim-request-id:
strict-transport-security: max-age=31536000; includeSubDomains; preload
x-content-type-options: nosniff
date: Tue, 23 Nov 2021 20:56:18 GMT
{
  "error": {
    "code": "ProjectNotFound",
    "message": "The specified project was not found.",
    "details": [
      {
        "code": "ProjectNotFound",
        "message": "{GUID}"
      }
    ]
  }
}
Get project deletion status
ﾉ
Expand table
\nVariable
name
Value
API-KEY
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. You can use either Key1 or Key2. Always having two valid keys allows for
secure key rotation with zero downtime. Alternatively you can find the value in Language
Studio > question answering > Deploy project > Get prediction URL. The key value is part
of the sample request.
PROJECT-
NAME
The name of project you would like to check on the deployment status for.
JOB-ID
When you delete a project programmatically, a JOB-ID  is generated as part of the operation-
location  response header to the deletion request. The JOB-ID  is the guid at the end of the
operation-location . For example: operation-location:
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/deletion-jobs/{THIS GUID IS YOUR JOB ID}
Bash
JSON
To retrieve information about a given project, update the following values in the query below:
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/deletion-jobs/{JOB-ID}?api-version=2021-10-01'
Example response
{
  "createdDateTime": "2021-11-23T20:56:18+00:00",
  "expirationDateTime": "2021-11-24T02:56:18+00:00",
  "jobId": "GUID",
  "lastUpdatedDateTime": "2021-11-23T20:56:18+00:00",
  "status": "succeeded"
}
Get project settings
ﾉ
Expand table
\nVariable
name
Value
ENDPOINT
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. Alternatively you can find the value in Language Studio > question
answering > Deploy project > Get prediction URL. An example endpoint is:
https://southcentralus.api.cognitive.microsoft.com/ . If this was your endpoint in the code
sample below, you would only need to add the region specific portion of southcentral  as the
rest of the endpoint path is already present.
API-KEY
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. You can use either Key1 or Key2. Always having two valid keys allows for
secure key rotation with zero downtime. Alternatively you can find the value in Language
Studio > question answering > Deploy project > Get prediction URL. The key value is part
of the sample request.
PROJECT-
NAME
The name of project you would like to retrieve information about.
Bash
JSON
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}?api-version=2021-10-01'
Example response
 {
    "200": {
      "headers": {},
      "body": {
        "projectName": "proj1",
        "description": "proj1 is a test project.",
        "language": "en",
        "settings": {
          "defaultAnswer": "No good match found for your question in the project."
        },
        "createdDateTime": "2021-05-01T15:13:22Z",
        "lastModifiedDateTime": "2021-05-01T15:13:22Z",
        "lastDeployedDateTime": "2021-05-01T15:13:22Z"
      }
\nTo retrieve question answer pairs and related information for a given project, update the
following values in the query below:
Variable
name
Value
ENDPOINT
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. Alternatively you can find the value in Language Studio > question
answering > Deploy project > Get prediction URL. An example endpoint is:
https://southcentralus.api.cognitive.microsoft.com/ . If this was your endpoint in the code
sample below, you would only need to add the region specific portion of southcentral  as the
rest of the endpoint path is already present.
API-KEY
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. You can use either Key1 or Key2. Always having two valid keys allows for
secure key rotation with zero downtime. Alternatively you can find the value in Language
Studio > question answering > Deploy project > Get prediction URL. The key value is part
of the sample request.
PROJECT-
NAME
The name of project you would like to retrieve all the question answer pairs for.
Bash
JSON
    }
  }
Get question answer pairs
ﾉ
Expand table
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/qnas?api-version=2021-10-01'
Example response
{
    "200": {