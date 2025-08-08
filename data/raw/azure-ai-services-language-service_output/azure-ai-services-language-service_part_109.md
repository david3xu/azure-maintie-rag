To retrieve the sources and related information for a given project, update the following values
in the query below:
      "headers": {},
      "body": {
        "value": [
          {
            "id": 1,
            "answer": "ans1",
            "source": "source1",
            "questions": [
              "question 1.1",
              "question 1.2"
            ],
            "metadata": {
              "k1": "v1",
              "k2": "v2"
            },
            "dialog": {
              "isContextOnly": false,
              "prompts": [
                {
                  "displayOrder": 1,
                  "qnaId": 11,
                  "displayText": "prompt 1.1"
                },
                {
                  "displayOrder": 2,
                  "qnaId": 21,
                  "displayText": "prompt 1.2"
                }
              ]
            },
            "lastUpdatedDateTime": "2021-05-01T17:21:14Z"
          },
          {
            "id": 2,
            "answer": "ans2",
            "source": "source2",
            "questions": [
              "question 2.1",
              "question 2.2"
            ],
            "lastUpdatedDateTime": "2021-05-01T17:21:14Z"
          }
        ]
      }
    }
  }
Get sources
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
The name of project you would like to retrieve all the source information for.
Bash
JSON
ﾉ
Expand table
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT_NAME}/sources?api-version=2021-10-01'
Example response
{
    "200": {
      "headers": {},
      "body": {
        "value": [
          {
            "displayName": "source1",
            "sourceUri": "https://learn.microsoft.com/azure/ai-
services/qnamaker/overview/overview",
            "sourceKind": "url",
            "lastUpdatedDateTime": "2021-05-01T15:13:22Z"
          },
          {
            "displayName": "source2",
            "sourceUri": "https://download.microsoft.com/download/2/9/B/29B20383-
\nTo retrieve synonyms and related information for a given project, update the following values
in the query below:
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
The name of project you would like to retrieve synonym information for.
Bash
302C-4517-A006-B0186F04BE28/surface-pro-4-user-guide-EN.pdf",
            "sourceKind": "file",
            "contentStructureKind": "unstructured",
            "lastUpdatedDateTime": "2021-05-01T15:13:22Z"
          }
        ]
      }
    }
  }
Get synonyms
ﾉ
Expand table
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/synonyms?api-version=2021-10-01'
\nJSON
To deploy a project to production, update the following values in the query below:
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
Example response
 {
    "200": {
      "headers": {},
      "body": {
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
      }
    }
  }
Deploy project
ﾉ
Expand table
\nVariable
name
Value
PROJECT-
NAME
The name of project you would like to deploy to production.
Bash
A successful call to deploy a project results in an Operation-Location  header being returned
which can be used to check the status of the deployment job. In most our examples, we
haven't needed to look at the response headers and thus haven't been displaying them. To
retrieve the response headers our curl command uses -i . Without this parameter prior to the
endpoint address, the response to this command would appear empty as if no response
occurred.
Bash
Example query
curl -X PUT -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '' -i 
'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/deployments/production?api-version=2021-10-
01'  
Example response
0HTTP/2 202
content-length: 0
operation-location: 
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/deployments/production/jobs/{JOB-ID-GUID}
x-envoy-upstream-service-time: 31
apim-request-id:
strict-transport-security: max-age=31536000; includeSubDomains; preload
x-content-type-options: nosniff
date: Tue, 23 Nov 2021 20:35:00 GMT
Get project deployment status
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
The name of project you would like to check on the deployment status for.
JOB-ID
When you deploy a project programmatically, a JOB-ID  is generated as part of the operation-
location  response header to the deployment request. The JOB-ID  is the guid at the end of
the operation-location . For example: operation-location:
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/deployments/production/jobs/{THIS GUID IS YOUR JOB
ID}
Bash
JSON
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '' 
'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/deployments/production/jobs/{JOB-ID}?api-
version=2021-10-01' 
Example response
    {
    "200": {
      "headers": {},
      "body": {
        "errors": [],
        "createdDateTime": "2021-05-01T17:21:14Z",
        "expirationDateTime": "2021-05-01T17:21:14Z",
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
The name of project you would like to export.
Bash
Bash
        "jobId": "{JOB-ID-GUID}",
        "lastUpdatedDateTime": "2021-05-01T17:21:14Z",
        "status": "succeeded"
      }
    }
  }
Export project metadata and assets
ﾉ
Expand table
Example query
curl -X POST -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '{exportAssetTypes": ["qnas","synonyms"]}' -i 
'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/:export?api-version=2021-10-01&format=tsv'
Example response
HTTP/2 202
content-length: 0
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
The name of project you would like to check on the export status for.
JOB-ID
When you export a project programmatically, a JOB-ID  is generated as part of the operation-
location  response header to the export request. The JOB-ID  is the guid at the end of the
operation-location . For example: operation-location:
https://southcentralus.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/sample-proj1/export/jobs/{THIS GUID IS YOUR JOB ID}
Bash
operation-location: 
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/Sample-project/export/jobs/{JOB-ID_GUID}
x-envoy-upstream-service-time: 214
apim-request-id:
strict-transport-security: max-age=31536000; includeSubDomains; preload
x-content-type-options: nosniff
date: Tue, 23 Nov 2021 21:24:03 GMT
Check export status
ﾉ
Expand table
Example query
curl -X GET -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '' 
'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/sample-proj1/export/jobs/{JOB-ID}?api-version=2021-10-01' 
\nJSON
If you try to access the resultUrl directly, you will get a 404 error. You must append ?api-
version=2021-10-01  to the path to make it accessible by an authenticated request:
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/export/jobs/{JOB-ID_GUID}/result?api-version=2021-
10-01
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
Studio > question answering > Deploy project > Get prediction URL. The key value is part of
the sample request.
PROJECT-
NAME
The name of project you would like to be the destination for the import.
FILE-
URI-PATH
When you export a project programmatically, and then check the status the export a
resultUrl  is generated as part of the response. For example: "resultUrl":
Example response
{
  "createdDateTime": "2021-11-23T21:24:03+00:00",
  "expirationDateTime": "2021-11-24T03:24:03+00:00",
  "jobId": "JOB-ID-GUID",
  "lastUpdatedDateTime": "2021-11-23T21:24:08+00:00",
  "status": "succeeded",
  "resultUrl": 
"https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/export/jobs/{JOB-ID_GUID}/result"
}
Import project
ﾉ
Expand table
\nVariable
name
Value
"https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/export/jobs/{JOB-ID_GUID}/result"  you can use the
resultUrl with the API version appended as a source file to import a project from:
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/export/jobs/{JOB-ID_GUID}/result?api-version=2021-
10-01 .
Bash
A successful call to import a project results in an Operation-Location  header being returned,
which can be used to check the status of the import job. In many of our examples, we haven't
needed to look at the response headers and thus haven't been displaying them. To retrieve the
response headers our curl command uses -i . Without this additional parameter prior to the
endpoint address, the response to this command would appear empty as if no response
occurred.
Bash
Example query
curl -X POST -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '{
      "fileUri": "FILE-URI-PATH"
  }' -i 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/:import?api-version=2021-10-01&format=tsv'
Example response
HTTP/2 202
content-length: 0
operation-location: 
https://southcentralus.api.cognitive.microsoft.com:443/language/query-
knowledgebases/projects/sample-proj1/import/jobs/{JOB-ID-GUID}
x-envoy-upstream-service-time: 417
apim-request-id: 
strict-transport-security: max-age=31536000; includeSubDomains; preload
x-content-type-options: nosniff
date: Wed, 24 Nov 2021 00:35:11 GMT
Check import status