We offer quickstarts in most popular programming languages, each designed to teach you
basic design patterns, and have you running code in less than 10 minutes.
Get started with the custom question answering client library
Custom question answering provides everything you need to build, manage, and deploy your
custom project.
Complete a quickstart
Next steps
\nQuickstart: custom question answering
Article • 03/24/2025
Get started with the custom question answering client library. Follow these steps to
install the package and try out the example code for basic tasks.
You can create a custom question answering project from your own content, such as
FAQs or product manuals. This article includes an example of creating a custom question
answering project from a product manual, to answer questions.
1. Sign in to the Language Studio
 with your Azure credentials.
2. Scroll down to the Answer questions section and select Open custom question
answering.
７ Note
Are you looking to migrate your workloads from QnA Maker? See our migration
guide for information on feature comparisons and migration steps.
Prerequisites
If you don't have an Azure subscription, create a free account
 before you begin.
＂
A language resource
 with the custom question answering feature enabled.
Remember your Microsoft Entra ID, Subscription, language resource name you
selected when you created the resource.
＂
Create your first custom question answering
project
\n![Image](images/page992_image1.png)
\n3. If your resource is not yet connected to Azure Search select Connect to Azure
Search. This will open a new browser tab to Features pane of your resource in the
Azure portal.
4. Select Enable custom question answering, choose the Azure Search resource to
link to, and then select Apply.
5. Return to the Language Studio tab. You might need to refresh this page for it to
register the change to your resource. Select Create new project.
6. Choose the option I want to set the language for all projects created in this
resource > select English > Select Next.
7. Enter a project name of Sample-project, a description of My first question
answering project, and leave the default answer with a setting of No answer
found.
\n![Image](images/page993_image1.png)

![Image](images/page993_image2.png)
\n8. Review your choices and select Create project
9. From the Manage sources page select Add source > URLS.
10. Select Add url enter the following values and then select Add all:
URL Name
URL Value
Surface
Book User
Guide
https://download.microsoft.com/download/7/B/1/7B10C82E-F520-4080-
8516-5CF0D803EEE0/surface-book-user-guide-EN.pdf
The extraction process takes a few moments to read the document and identify
questions and answers.
After successfully adding the source, you can then edit the source contents to add
more custom question answer sets.
1. Select the link to your source, this will open the edit project page.
2. Select Test from the menu bar > Enter the question How do I setup my surface
book?. An answer will be generated based on the question answer pairs that were
automatically identified and extracted from your source URL:
ﾉ
Expand table
Test your project
\n![Image](images/page995_image1.png)
\nIf you check the box for include short answer response you will also see a precise
answer, if available, along with the answer passage in the test pane when you ask a
question.
3. Select Inspect to examine the response in more detail. The test window is used to
test your changes to your project before deploying your project.
From the Inspect interface, you can see the level of confidence that this response
will answer the question and directly edit a given question and answer response
pair.
Deploy your project
\n![Image](images/page996_image1.png)

![Image](images/page996_image2.png)
\n1. Select the Deploy project icon to enter the deploy project menu.
When you deploy a project, the contents of your project move from the test
index to a prod  index in Azure Search.
2. Select Deploy > and then when prompted select Deploy again.
Your project is now successfully deployed. You can use the endpoint to answer
questions in your own custom application to answer or in a bot.
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other
resources associated with it.
Azure portal
Azure CLI
Clean up resources
\n![Image](images/page997_image1.png)

![Image](images/page997_image2.png)
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
To learn about automating your custom question answering pipeline consult the REST
API documentation. Currently authoring functionality is only available via REST API:
Authoring API reference
Authoring API cURL examples
Runtime API reference
Tutorial: Create an FAQ bot
Explore the REST API
Next steps
Yes
No
\nLanguage support for custom question
answering and projects
06/30/2025
This article describes the language support options for custom question answering enabled
resources and projects.
In custom question answering, you have the option to either select the language each time you
add a new project to a resource allowing multiple language support, or you can select a
language that will apply to all future projects for a resource.
When you are creating the first project in your service, you get a choice pick the language
each time you create a new project. Select this option, to create projects belonging to
different languages within one service.
Supporting multiple languages in one custom
question answering enabled resource
\n![Image](images/page999_image1.png)
\nThe language setting option cannot be modified for the service once the first project is
created.
If you enable multiple languages for the project, then instead of having one test index for
the service you will have one test index per project.
If you need to support a project system, which includes several languages, you can:
Use the Translator service to translate a question into a single language before sending
the question to your project. This allows you to focus on the quality of a single language
and the quality of the alternate questions and answers.
Create a custom question answering enabled language resource, and a project inside that
resource, for every language. This allows you to manage separate alternate questions and
answer text that is more nuanced for each language. This provides more flexibility but
requires a much higher maintenance cost when the questions or answers change across
all languages.
If you select the option to set the language used by all projects associated with the resource,
consider the following:
A language resource, and all its projects, will support one language only.
The language is explicitly set when the first project of the service is created.
The language can't be changed for any other projects associated with the resource.
The language is used by the Azure AI Search service (ranker #1) and custom question
answering (ranker #2) to generate the best answer to a query.
The following list contains the languages supported for a custom question answering resource.
Arabic
Armenian
Bangla
Basque
Bulgarian
Catalan
Chinese_Simplified
Chinese_Traditional
Supporting multiple languages in one project
Single language per resource
Languages supported