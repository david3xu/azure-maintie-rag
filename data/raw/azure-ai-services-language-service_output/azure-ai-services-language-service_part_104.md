![Image](images/page1031_image1.png)
\nIf you check the box for include short answer response you will also see a precise answer,
if available, along with the answer passage in the test pane when you ask a question.
3. Select Inspect to examine the response in more detail. The test window is used to test
your changes to your project before deploying your project.
From the Inspect interface, you can see the level of confidence that this response will
answer the question and directly edit a given question and answer response pair.
1. Select the Deploy project icon to enter the deploy project menu.
Deploy your project
\n![Image](images/page1032_image1.png)

![Image](images/page1032_image2.png)
\nWhen you deploy a project, the contents of your project move from the test  index to a
prod  index in Azure Search.
2. Select Deploy > and then when prompted select Deploy again.
Your project is now successfully deployed. You can use the endpoint to answer questions
in your own custom application to answer or in a bot.
If you will not continue to test custom question answering, you can delete the associated
resource.
Clean up resources
Next steps
\n![Image](images/page1033_image1.png)

![Image](images/page1033_image2.png)
\nExport-import-refresh in custom question
answering
06/21/2025
You might want to create a copy of your custom question answering project or related
question and answer pairs for several reasons:
To implement a backup and restore process
To integrate with your CI/CD pipeline
To move your data to different regions
If you don't have an Azure subscription, create a free account
 before you begin.
A language resource
 with the custom question answering feature enabled. Remember
your Microsoft Entra ID, Subscription, language resource name you selected when you
created the resource.
1. Sign in to the Language Studio
 with your Azure credentials.
2. Scroll down to the Answer questions section and select Open custom question
answering.
3. Select the project you wish to export > Select Export > You’ll have the option to export as
an Excel or TSV file.
4. You’ll be prompted to save your exported file locally as a zip file.
To automate the export process, use the export functionality of the authoring API
1. Sign in to the Language Studio
 with your Azure credentials.
2. Scroll down to the Answer questions section and select Open custom question
answering.
Prerequisites
Export a project
Export a project programmatically
Import a project
\n3. Select Import and specify the file type you selected for the export process. Either Excel, or
TSV.
4. Select Choose File and browse to the local zipped copy of your project that you exported
previously.
5. Provide a unique name for the project you’re importing.
6. Remember that a project that has only been imported still needs to be
deployed/published if you want it to be live.
To automate the import process, use the import functionality of the authoring API
1. Sign in to the Language Studio
 with your Azure credentials.
2. Scroll down to the Answer questions section and select Open custom question
answering.
3. Select the project that contains the source you want to refresh > select manage sources.
4. We recommend having a backup of your project/question answer pairs prior to running
each refresh so that you can always roll-back if needed.
5. Select a URL-based source to refresh > Select Refresh URL.
6. Only one URL can be refreshed at a time.
To automate the URL refresh process, use the update sources functionality of the authoring API
The update sources example in the Authoring API docs shows the syntax for adding a new
URL-based source. An example query for an update would be as follows:
Variable
name
Value
ENDPOINT
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. Alternatively you can find the value in Language Studio > question
Import a project programmatically
Refresh source url
Refresh a URL programmatically
ﾉ
Expand table
\nVariable
name
Value
answering > Deploy project > Get prediction URL. An example endpoint is:
https://southcentralus.api.cognitive.microsoft.com/ . If this was your endpoint in the
following code sample, you would only need to add the region-specific portion of
southcentral  as the rest of the endpoint path is already present.
API-KEY
This value can be found in the Keys & Endpoint section when examining your resource from
the Azure portal. You can use either Key1 or Key2. Always having two valid keys allows for
secure key rotation with zero downtime. Alternatively you can find the value in Language
Studio > question answering > Deploy project > Get prediction URL. The key value is part
of the sample request.
PROJECT-
NAME
The name of project where you would like to update sources.
Bash
It’s also possible to export/import a specific project of question and answers rather than the
entire custom question answering project.
1. Sign in to the Language Studio
 with your Azure credentials.
2. Scroll down to the Answer questions section and select Open custom question
answering.
3. Select the project that contains the project question and answer pairs you want to export.
4. Select Edit project.
curl -X PATCH -H "Ocp-Apim-Subscription-Key: {API-KEY}" -H "Content-Type: 
application/json" -d '[
  {
    "op": "replace",
    "value": {
      "displayName": "source5",
      "sourceKind": "url",
      "sourceUri": https://download.microsoft.com/download/7/B/1/7B10C82E-F520-
4080-8516-5CF0D803EEE0/surface-book-user-guide-EN.pdf,
      "refresh": "true"
    }
  }
]'  -i 'https://{ENDPOINT}.api.cognitive.microsoft.com/language/query-
knowledgebases/projects/{PROJECT-NAME}/sources?api-version=2021-10-01'
Export questions and answers
\n5. To the right of show columns are ...  an ellipsis button. > Select the ...  > a dropdown
will reveal the option to export/import questions and answers.
Depending on the size of your web browser, you may experience the UI differently.
Smaller browsers will see two separate ellipsis buttons.
It’s also possible to export/import a specific project of question and answers rather than the
entire custom question answering project.
1. Sign in to the Language Studio
 with your Azure credentials.
2. Scroll down to the Answer questions section and select Open custom question
answering.
3. Select the project that contains the project question and answer pairs you want to export.
4. Select Edit project.
5. To the right of show columns are ...  an ellipsis button. > Select the ...  > a dropdown
will reveal the option to export/import questions and answers.
Depending on the size of your web browser, you may experience the UI differently.
Smaller browsers will see two separate ellipsis buttons.
Import questions and answers
\n![Image](images/page1037_image1.png)
\nLearn how to use the Authoring API
Next steps
\n![Image](images/page1038_image1.png)
\nUse smart URL refresh with a project
06/21/2025
Custom question answering gives you the ability to refresh your source contents by getting the
latest content from a source URL and updating the corresponding project with one click. The
service will ingest content from the URL and either create, merge, or delete question-and-
answer pairs in the project.
This functionality is provided to support scenarios where the content in the source URL
changes frequently, such as the FAQ page of a product that's updated often. The service will
refresh the source and update the project to the latest content while retaining any manual
edits made previously.
If you have a project with a URL source that has changed, you can trigger a smart URL refresh
to keep your project up to date. The service will scan the URL for updated content and
generate QnA pairs. It will add any new QnA pairs to your project and also delete any pairs that
have disappeared from the source (with exceptions—see below). It also merges old and new
QnA pairs in some situations (see below).
You can trigger a URL refresh in Language Studio by opening your project, selecting the source
in the Manage sources list, and selecting Refresh URL.
７ Note
This feature is only applicable to URL sources, and they must be refreshed individually, not
in bulk.
） Important
This feature is only available in the 2021-10-01  version of the Language API.
How it works
） Important
Because smart URL refresh can involve deleting old content from your project, you might
want to create a backup of your project before you do any refresh operations.
\nYou can also trigger a refresh programmatically using the REST API. See the Update Sources
reference documentation for parameters and a sample request.
When the user refreshes content using this feature, the project of QnA pairs may be updated in
the following ways:
If the content of the URL is updated so that an existing QnA pair from the old content of the
URL is no longer found in the source, that pair is deleted from the refreshed project. For
example, if a QnA pair Q1A1 existed in the old project, but after refreshing, there's no A1
answer generated by the newly refreshed source, then the pair Q1A1 is considered outdated
and is dropped from the project altogether.
However, if the old QnA pairs have been manually edited in the authoring portal, they won't be
deleted.
If the content of the URL is updated in such a way that a new QnA pair exists which didn't exist
in the old KB, then it's added to the KB. For example, if the service finds that a new answer A2
can be generated, then the QnA pair Q2A2 is inserted into the KB.
Smart refresh behavior
Delete old pair
Add new pair
\n![Image](images/page1040_image1.png)