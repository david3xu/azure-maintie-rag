\nDeploy custom language projects to
multiple regions
Article • 04/29/2025
Custom language service features enable you to deploy your project to more than one region.
This capability makes it much easier to access your project globally while you manage only one
instance of your project in one place. As of November 2024, custom language service features
also enable you to deploy your project to multiple resources within a single region via the API,
so that you can use your custom model wherever you need.
Before you deploy a project, you can assign deployment resources in other regions. Each
deployment resource is a different Language resource from the one that you use to author
your project. You deploy to those resources and then target your prediction requests to that
resource in their respective regions and your queries are served directly from that region.
When you create a deployment, you can select which of your assigned deployment resources
and their corresponding regions you want to deploy to. The model you deploy is then
replicated to each region and accessible with its own endpoint dependent on the deployment
resource's custom subdomain.
Suppose you want to make sure your project, which is used as part of a customer support
chatbot, is accessible by customers across the United States and India. You author a project
with the name ContosoSupport  by using a West US 2 Language resource named MyWestUS2 .
Before deployment, you assign two deployment resources to your project: MyEastUS  and
MyCentralIndia  in East US and Central India, respectively.
When you deploy your project, you select all three regions for deployment: the original West
US 2 region and the assigned ones through East US and Central India.
７ Note
This article applies to the following custom features in Azure AI Language:
Conversational language understanding
Custom text classification
Custom named entity recognition (NER)
Orchestration workflow
Example
\nYou now have three different endpoint URLs to access your project in all three regions:
West US 2: https://mywestus2.cognitiveservices.azure.com/language/:analyze-
conversations
East US: https://myeastus.cognitiveservices.azure.com/language/:analyze-
conversations
Central India: https://mycentralindia.cognitiveservices.azure.com/language/:analyze-
conversations
The same request body to each of those different URLs serves the exact same response directly
from that region.
Assigning deployment resources requires Microsoft Entra authentication. Microsoft Entra ID is
used to confirm that you have access to the resources that you want to assign to your project
for multiregion deployment. In Language Studio, you can automatically enable Microsoft Entra
authentication
 by assigning yourself the Azure Cognitive Services Language Owner role to
your original resource. To programmatically use Microsoft Entra authentication, learn more
from the Azure AI services documentation.
Your project name and resource are used as its main identifiers. A Language resource can only
have a specific project name in each resource. Any other projects with the same name can't be
deployed to that resource.
For example, if a project ContosoSupport  was created by the resource MyWestUS2  in West US 2
and deployed to the resource MyEastUS  in East US, the resource MyEastUS  can't create a
different project called ContosoSupport  and deploy a project to that region. Similarly, your
collaborators can't then create a project ContosoSupport  with the resource MyCentralIndia  in
Central India and deploy it to either MyWestUS2  or MyEastUS .
You can only swap deployments that are available in the exact same regions. Otherwise,
swapping fails.
If you remove an assigned resource from your project, all of the project deployments to that
resource are deleted.
Some regions are only available for deployment and not for authoring projects.
Learn how to deploy models for:
Validations and requirements
Related content
\nConversational language understanding
Custom text classification
Custom NER
Orchestration workflow
\nProject versioning
06/30/2025
Building your project typically happens in increments. You may add, remove, or edit intents,
entities, labels and data at each stage. Every time you train, a snapshot of your current project
state is taken to produce a model. That model saves the snapshot to be loaded back at any
time. Every model acts as its own version of the project.
For example, if your project has 10 intents and/or entities, with 50 training documents or
utterances, it can be trained to create a model named v1. Afterwards, you might make changes
to the project to alter the numbers of training data. The project can be trained again to create
a new model named v2. If you don't like the changes you've made in v2 and would like to
continue from where you left off in model v1, then you would just need to load the model data
from v1 back into the project. Loading a model's data is possible through both the Language
Studio and API. Once complete, the project will have the original amount and types of training
data.
If the project data is not saved in a trained model, it can be lost. For example, if you loaded
model v1, your project now has the data that was used to train it. If you then made changes,
didn't train, and loaded model v2, you would lose those changes as they weren't saved to any
specific snapshot.
If you overwrite a model with a new snapshot of data, you won't be able to revert back to any
previous state of that model.
You always have the option to locally export the data for every model.
The data for your model versions will be saved in different locations, depending on the custom
feature you're using.
７ Note
This article applies to the following custom features in Azure AI Language:
Conversational language understanding
Custom text classification
Custom NER
Orchestration workflow
Data location
\nIn custom named entity recognition, the data being saved to the snapshot is the labels file.
Learn how to load or export model data for:
Conversational language understanding
Custom text classification
Custom NER
Orchestration workflow
Custom NER
Next steps
\nWhat is orchestration workflow?
06/04/2025
Orchestration workflow is one of the features offered by Azure AI Language. It is a cloud-based
API service that applies machine-learning intelligence to enable you to build orchestration
models to connect Conversational Language Understanding (CLU), Question Answering
projects and LUIS applications. By creating an orchestration workflow, developers can
iteratively tag utterances, train and evaluate model performance before making it available for
consumption. To simplify building and customizing your model, the service offers a custom
web portal that can be accessed through the Language studio
. You can easily get started
with the service by following the steps in this quickstart.
This documentation contains the following article types:
Quickstarts are getting-started instructions to guide you through making requests to the
service.
Concepts provide explanations of the service functionality and features.
How-to guides contain instructions for using the service in more specific or customized
ways.
Orchestration workflow can be used in multiple scenarios across a variety of industries. Some
examples are:
In a large corporation, an enterprise chat bot might handle a variety of employee affairs. It
might be able to handle frequently asked questions served by a custom question answering
knowledge base, a calendar specific skill served by conversational language understanding, and
an interview feedback skill served by LUIS. The bot needs to be able to appropriately route
incoming requests to the correct service. Orchestration workflow allows you to connect those
skills to one project that handles the routing of incoming requests appropriately to power the
enterprise bot.
Creating an orchestration workflow project typically involves several different steps.
Example usage scenarios
Enterprise chat bot
Project development lifecycle
\nFollow these steps to get the most out of your model:
1. Define your schema: Know your data and define the actions and relevant information
that needs to be recognized from user's input utterances. Create the intents that you
want to assign to user's utterances and the projects you want to connect to your
orchestration project.
2. Label your data: The quality of data tagging is a key factor in determining model
performance.
3. Train a model: Your model starts learning from your tagged data.
4. View the model's performance: View the evaluation details for your model to determine
how well it performs when introduced to new data.
5. Improve the model: After reviewing the model's performance, you can then learn how
you can improve the model.
6. Deploy the model: Deploying a model makes it available for use via the prediction API.
7. Predict intents: Use your custom model to predict intents from user's utterances.
As you use orchestration workflow, see the following reference documentation and samples for
Azure AI Language:
Development option / language
Reference documentation
Samples
REST APIs (Authoring)
REST API documentation
REST APIs (Runtime)
REST API documentation

Reference documentation and code samples
ﾉ
Expand table
\n![Image](images/page778_image1.png)
\nDevelopment option / language
Reference documentation
Samples
C# (Runtime)
C# documentation
C# samples
Python (Runtime)
Python documentation
Python samples
An AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Read the transparency
note for CLU and orchestration workflow to learn about responsible AI use and deployment in
your systems. You can also see the following articles for more information:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
Use the quickstart article to start using orchestration workflow.
As you go through the project development lifecycle, review the glossary to learn more
about the terms used throughout the documentation for this feature.
Remember to view the service limits for information such as regional availability.
Responsible AI
Next steps
\nQuickstart: Orchestration workflow
06/30/2025
Use this article to get started with Orchestration workflow projects using Language Studio and
the REST API. Follow these steps to try out an example.
Azure subscription - Create one for free
.
A conversational language understanding project.
1. Go to the Language Studio
 and sign in with your Azure account.
2. In the Choose a language resource window that appears, find your Azure subscription,
and choose your Language resource. If you don't have a resource, you can create a new
one.
Instance detail
Required value
Azure subscription
Your Azure subscription.
Azure resource
group
Your Azure resource group.
Azure resource
name
Your Azure resource name.
Location
A valid location for your Azure resource. For example, "West US 2".
Pricing tier
A supported pricing tier for your Azure resource. You can use the Free (F0)
tier to try the service.
Prerequisites
Sign in to Language Studio
ﾉ
Expand table