Create, test, and deploy a custom question answering project
Next Steps
\nGet analytics for your project
06/21/2025
Custom question answering uses Azure diagnostic logging to store the telemetry data and chat
logs. Follow the below steps to run sample queries to get analytics on the usage of your
custom question answering project.
1. Enable diagnostics logging for your language resource with custom question answering
enabled.
2. In the previous step, select Trace in addition to Audit, RequestResponse and AllMetrics
for logging
Kusto
Kusto queries
Chat log
// All QnA Traffic
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where OperationName=="CustomQuestionAnswering QueryKnowledgebases" // This 
OperationName is valid for custom question answering enabled resources
| extend answer_ = tostring(parse_json(properties_s).answer)
| extend question_ = tostring(parse_json(properties_s).question)
\n![Image](images/page1062_image1.png)
\nKusto
Kusto
Kusto
| extend score_ = tostring(parse_json(properties_s).score)
| extend kbId_ = tostring(parse_json(properties_s).kbId)
| project question_, answer_, score_, kbId_
Traffic count per project and user in a time period
// Traffic count per KB and user in a time period
let startDate = todatetime('2019-01-01');
let endDate = todatetime('2020-12-31');
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where OperationName=="CustomQuestionAnswering QueryKnowledgebases" // This 
OperationName is valid for custom question answering enabled resources
| where TimeGenerated <= endDate and TimeGenerated >=startDate
| extend kbId_ = tostring(parse_json(properties_s).kbId)
| extend userId_ = tostring(parse_json(properties_s).userId)
| summarize ChatCount=count() by bin(TimeGenerated, 1d), kbId_, userId_
Latency of GenerateAnswer API
// Latency of GenerateAnswer
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where OperationName=="Generate Answer"
| project TimeGenerated, DurationMs
| render timechart
Average latency of all operations
// Average Latency of all operations
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| project DurationMs, OperationName
| summarize count(), avg(DurationMs) by OperationName
| render barchart
Unanswered questions
\nKusto
Kusto
// All unanswered questions
AzureDiagnostics
| where ResourceProvider == "MICROSOFT.COGNITIVESERVICES"
| where OperationName=="CustomQuestionAnswering QueryKnowledgebases" // This 
OperationName is valid for custom question answering enabled resources
| extend answer_ = tostring(parse_json(properties_s).answer)
| extend question_ = tostring(parse_json(properties_s).question)
| extend score_ = tostring(parse_json(properties_s).score)
| extend kbId_ = tostring(parse_json(properties_s).kbId)
| where score_ == 0
| project question_, answer_, score_, kbId_
Prebuilt custom question answering inference calls
// Show logs from AzureDiagnostics table 
// Lists the latest logs in AzureDiagnostics table, sorted by time (latest first). 
AzureDiagnostics
| where OperationName == "CustomQuestionAnswering QueryText"
| extend answer_ = tostring(parse_json(properties_s).answer)
| extend question_ = tostring(parse_json(properties_s).question)
| extend score_ = tostring(parse_json(properties_s).score)
| extend requestid = tostring(parse_json(properties_s)["apim-request-id"])
| project TimeGenerated, requestid, question_, answer_, score_
Next steps
\nCreate and manage project settings
06/21/2025
Custom question answering allows you to manage your projects by providing access to the
project settings and data sources. If you haven't created a custom question answering project
before we recommend starting with the getting started article.
If you don't have an Azure subscription, create a free account
 before you begin.
A Language resource
 with the custom question answering feature enabled in the
Azure portal. Remember your Microsoft Entra ID, Subscription, and language resource
name you selected when you created the resource.
1. Sign in to the Language Studio
 portal with your Azure credentials.
2. Open the question answering
 page.
3. Select create new project.
4. If you are creating the first project associated with your language resource, you have the
option of creating future projects with multiple languages for the same resource. If you
choose to explicitly set the language to a single language in your first project, you will not
be able to modify this setting later and all subsequent projects for that resource will use
the language selected during the creation of your first project.
Prerequisites
Create a project
\n5. Enter basic project settings:
Setting
Value
Name
Enter your unique project name here
Description
Enter a description for your project
Source
language
Whether or not this value is greyed out, is dependent on the selection that was
made when the first project associated with the language resource was created.
Default
answer
The default answer the system will send if there was no answer found for the
question. You can change this at any time in Project settings.
From the main custom question answering page in Language Studio you can:
Create projects
Delete projects
Export existing projects for backup or to migrate to other language resources
ﾉ
Expand table
Manage projects
\n![Image](images/page1066_image1.png)
\nImport projects. (The expected file format is a .zip  file containing a project that was
exported in excel  or .tsv  format).
Projects can be ordered by either Last modified or Last published date.
1. Select Manage sources in the left pane.
2. There are three types of sources: URLS, Files, and Chitchat
Goal
Action
Add Source
You can add new sources and FAQ content to your project by selecting Add
source > and choosing URLs, Files, or Chitchat
Delete Source
You can delete existing sources by selecting to the left of the source, which
will cause a blue circle with a checkmark to appear > select the trash can
icon.
Mark content as
unstructured
If you want to mark the uploaded file content as unstructured select
Unstructured content from the dropdown when adding the source.
Auto-detect
Allow question and answering to attempt to determine if content is
structured versus unstructured.
From the Edit project page you can:
Search project: You can search the project by typing in the text box at the top of question
answer panel. Hit enter to search on the question, answer, or metadata content.
Pagination: Quickly move through data sources to manage large projects. Page numbers
appear at the bottom of the UI and are sometimes off screen.
Deleting a project is a permanent operation. It can't be undone. Before deleting a project, you
should export the project from the main custom question answering page within Language
Studio.
Manage sources
ﾉ
Expand table
Manage large projects
Delete project
\nIf you share your project with collaborators and then later delete it, everyone loses access to
the project.
Configure resources
Next steps
\nNetwork isolation and private endpoints
06/06/2025
The following steps describe how to restrict public access to custom question answering
resources as well as how to enable Azure Private Link. Protect an AI Foundry resource from
public access by configuring the virtual network.
Azure Private Endpoint is a network interface that connects you privately and securely to a
service powered by Azure Private Link. Custom question answering provides you support to
create private endpoints to the Azure Search Service.
Private endpoints are provided by Azure Private Link, as a separate service. For more
information about costs, see the pricing page.
1. Assign the contributor role to your resource in the Azure Search Service instance. This
operation requires Owner access to the subscription. Go to Identity tab in the service
resource to get the identity.
2. Add the above identity as Contributor by going to the Azure Search Service access control
tab.
Private Endpoints
Steps to enable private endpoint
\n![Image](images/page1069_image1.png)
\n3. Select on Add role assignments, add the identity and select Save.
4. Now, go to Networking tab in the Azure Search Service instance and switch Endpoint
connectivity data from Public to Private. This operation is a long running process and can
take up to 30 mins to complete.
\n![Image](images/page1070_image1.png)

![Image](images/page1070_image2.png)