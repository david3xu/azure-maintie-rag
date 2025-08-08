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
The name of project you would like to be the destination for the import.
JOB-ID
When you import a project programmatically, a JOB-ID  is generated as part of the operation-
location  response header to the export request. The JOB-ID  is the GUID at the end of the
operation-location . For example: operation-location:
https://southcentralus.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/sample-proj1/import/jobs/{THIS GUID IS YOUR JOB ID}
Bash
Bash
ﾉ
Expand table
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '' 
'https://southcentralus.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME/import/jobs/{JOB-ID-GUID}?api-version=2021-
10-01' 
Example query response
{
  "errors": [],
  "createdDateTime": "2021-05-01T17:21:14Z",
  "expirationDateTime": "2021-05-01T17:21:14Z",
  "jobId": "JOB-ID-GUID",
  "lastUpdatedDateTime": "2021-05-01T17:21:14Z",
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
The name of project you would like to generate a deployment list for.
Bash
JSON
  "status": "succeeded"
}
List deployments
ﾉ
Expand table
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '' 
'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/deployments?api-version=2021-10-01' 
Example response
[
  {
    "deploymentName": "production",
    "lastDeployedDateTime": "2021-10-26T15:12:02Z"
\nRetrieve a list of all question answering projects your account has access to.
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
Bash
JSON
  }
]
List Projects
ﾉ
Expand table
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '' 
'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects?api-version=2021-10-01' 
Example response
{
  "value": [
    {
      "projectName": "Sample-project",
      "description": "My first question answering project",
      "language": "en",
      "multilingualResource": false,
\nIn this example, we will add a new source to an existing project. You can also replace and
delete existing sources with this command depending on what kind of operations you pass as
part of your query body.
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
The name of project where you would like to update sources.
METHOD
PATCH
Bash
      "createdDateTime": "2021-10-07T04:51:15Z",
      "lastModifiedDateTime": "2021-10-27T00:42:01Z",
      "lastDeployedDateTime": "2021-11-24T01:34:18Z",
      "settings": {
        "defaultAnswer": "No good match found in KB"
      }
    }
  ]
}
Update sources
ﾉ
Expand table
Example query
curl -X PATCH -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '[
  {
    "op": "add",
\nA successful call to update a source results in an Operation-Location  header being returned
which can be used to check the status of the import job. In many of our examples, we haven't
needed to look at the response headers and thus haven't always been displaying them. To
retrieve the response headers our curl command uses -i . Without this parameter prior to the
endpoint address, the response to this command would appear empty as if no response
occurred.
Bash
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
    "value": {
      "displayName": "source5",
      "sourceKind": "url",
      "sourceUri": "https://download.microsoft.com/download/7/B/1/7B10C82E-F520-
4080-8516-5CF0D803EEE0/surface-book-user-guide-EN.pdf",
      "sourceContentStructureKind": "semistructured"
    }
  }
]'  -i '{LanguageServiceName}.cognitiveservices.azure.com//language/query-
knowledgebases/projects/{projectName}/sources?api-version=2021-10-01'
Example response
HTTP/2 202
content-length: 0
operation-location: 
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/Sample-project/sources/jobs/{JOB_ID_GUID}
x-envoy-upstream-service-time: 412
apim-request-id: dda23d2b-f110-4645-8bce-1a6f8d504b33
strict-transport-security: max-age=31536000; includeSubDomains; preload
x-content-type-options: nosniff
date: Wed, 24 Nov 2021 02:47:53 GMT
Get update source status
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
The name of project you would like to be the destination for the import.
JOB-ID
When you update a source programmatically, a JOB-ID  is generated as part of the operation-
location  response header to the update source request. The JOB-ID  is the GUID at the end of
the operation-location . For example: operation-location:
https://southcentralus.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/sample-proj1/sources/jobs/{THIS GUID IS YOUR JOB ID}
Bash
Bash
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '' 
'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/sources/jobs/{JOB-ID}?api-version=2021-10-
01' 
Example response
{
  "createdDateTime": "2021-11-24T02:47:53+00:00",
  "expirationDateTime": "2021-11-24T08:47:53+00:00",
  "jobId": "{JOB-ID-GUID}",
  "lastUpdatedDateTime": "2021-11-24T02:47:56+00:00",
  "status": "succeeded",
  "resultUrl": "/knowledgebases/Sample-project"
}
Update question and answer pairs
\nIn this example, we will add a question answer pair to an existing source. You can also modify,
or delete existing question answer pairs with this query depending on what operation you pass
in the query body. If you don't have a source named source5 , this example query will fail. You
can adjust the source value in the body of the query to a source that exists for your target
project.
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
The name of project you would like to be the destination for the import.
Bash
ﾉ
Expand table
curl -X PATCH -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '[
    {
        "op": "add",
        "value":{
            "id": 1,
            "answer": "The latest question answering docs are on 
https://learn.microsoft.com",
            "source": "source5",
            "questions": [
                "Where do I find docs for question answering?"
            ],
            "metadata": {},
            "dialog": {
                "isContextOnly": false,
                "prompts": []
            }
        }
    }
\nA successful call to update a question answer pair results in an Operation-Location  header
being returned which can be used to check the status of the update job. In many of our
examples, we haven't needed to look at the response headers and thus haven't always been
displaying them. To retrieve the response headers our curl command uses -i . Without this
parameter prior to the endpoint address, the response to this command would appear empty
as if no response occurred.
Bash
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
]'  -i 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/qnas?api-version=2021-10-01'
Example response
HTTP/2 202
content-length: 0
operation-location: 
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/Sample-project/qnas/jobs/{JOB-ID-GUID}
x-envoy-upstream-service-time: 507
apim-request-id: 
strict-transport-security: max-age=31536000; includeSubDomains; preload
x-content-type-options: nosniff
date: Wed, 24 Nov 2021 03:16:01 GMT
Get update question answer pairs status
ﾉ
Expand table
\nVariable
name
Value
PROJECT-
NAME
The name of project you would like to be the destination for the question answer pairs
updates.
JOB-ID
When you update a question answer pair programmatically, a JOB-ID  is generated as part of
the operation-location  response header to the update request. The JOB-ID  is the GUID at
the end of the operation-location . For example: operation-location:
https://southcentralus.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/sample-proj1/qnas/jobs/{THIS GUID IS YOUR JOB ID}
Bash
Bash
Variable
name
Value
ENDPOINT
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. Alternatively you can find the value in Language Studio > question
answering > Deploy project > Get prediction URL. An example endpoint is:
https://southcentralus.api.cognitive.microsoft.com/ . If this was your endpoint in the code
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '' 
'https://southcentralus.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/qnas/jobs/{JOB-ID}?api-version=2021-10-01' 
Example response
  "createdDateTime": "2021-11-24T03:16:01+00:00",
  "expirationDateTime": "2021-11-24T09:16:01+00:00",
  "jobId": "{JOB-ID-GUID}",
  "lastUpdatedDateTime": "2021-11-24T03:16:06+00:00",
  "status": "succeeded",
  "resultUrl": "/knowledgebases/Sample-project"
Update Synonyms
ﾉ
Expand table
\nVariable
name
Value
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
The name of project you would like to add synonyms.
Bash
Bash
Example query
curl -X PUT -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '{
"value": [
    {
      "alterations": [
        "qnamaker",
        "qna maker"
      ]
    },
    {
      "alterations": [
        "botframework",
        "bot framework"
      ]
    }
  ]
}' -i 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/synonyms?api-version=2021-10-01'
Example response
0HTTP/2 200
content-length: 17
content-type: application/json; charset=utf-8
x-envoy-upstream-service-time: 39
apim-request-id: 5deb2692-dac8-43a8-82fe-36476e407ef6
strict-transport-security: max-age=31536000; includeSubDomains; preload
x-content-type-options: nosniff