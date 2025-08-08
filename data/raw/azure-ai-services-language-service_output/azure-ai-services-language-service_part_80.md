How to create projects in orchestration
workflow
06/21/2025
Orchestration workflow allows you to create projects that connect your applications to:
Custom Language Understanding
Question Answering
LUIS
Before you start using orchestration workflow, you will need several things:
An Azure subscription - Create one for free
.
An Azure AI Language resource
Before you start using orchestration workflow, you will need an Azure AI Language resource.
1. Go to the Azure portal
 to create a new Azure AI Language resource.
2. Select Continue to create your resource
3. Create a Language resource with following details.
Prerequisites
Create a Language resource
７ Note
You need to have an owner role assigned on the resource group to create a
Language resource.
If you are planning to use question answering, you have to enable question
answering in resource creation
Create a new resource from the Azure portal
ﾉ
Expand table
\nInstance detail
Required value
Region
One of the supported regions.
Name
A name for your Language resource.
Pricing tier
One of the supported pricing tiers.
If it's your first time logging in, you'll see a window in Language Studio
 that will let you
choose an existing Language resource or create a new one. You can also create a resource by
clicking the settings icon in the top-right corner, selecting Resources, then clicking Create a
new resource.
Create a Language resource with following details.
Instance detail
Required value
Azure subscription
Your Azure subscription
Azure resource group
Your Azure resource group
Azure resource name
Your Azure resource name
Location
Learn more about supported regions.
Pricing tier
Learn more about supported pricing tiers.
To create a new intent, select +Add button and start by giving your intent a name. You will see
two options, to connect to a project or not. You can connect to (LUIS, question answering, or
Conversational Language Understanding) projects, or choose the no option.
Create a new Language resource from Language Studio
ﾉ
Expand table
） Important
Make sure to enable Managed Identity when you create a Language resource.
Read and confirm Responsible AI notice
Sign in to Language Studio
Create an orchestration workflow project
\nOnce you have a Language resource created, create an orchestration workflow project.
1. In Language Studio
, find the section labeled Understand questions and
conversational language and select Orchestration Workflow.
2. This will bring you to the Orchestration workflow project page. Select Create new
project. To create a project, you will need to provide the following details:
Value
Description
Name
A name for your project.
Description
Optional project description.
Utterances primary
language
The primary language of your project. Your training data should primarily
be in this language.
Once you're done, select Next and review the details. Select create project to complete
the process. You should now see the Build Schema screen in your project.
You can export an orchestration workflow project as a JSON file at any time by going to
the orchestration workflow projects page, selecting a project, and from the top menu,
Language Studio

ﾉ
Expand table
Import an orchestration workflow project
Language Studio
\n![Image](images/page793_image1.png)
\nclicking on Export.
That project can be reimported as a new project. If you import a project with the exact
same name, it replaces the project's data with the newly imported project's data.
To import a project, select the arrow button next to Create a new project and select
Import, then select the JSON file.
You can export an orchestration workflow project as a JSON file at any time by going to
the orchestration workflow projects page, selecting a project, and pressing Export.
1. Go to your project settings page in Language Studio
.
2. You can see project details.
3. In this page you can update project description.
4. You can also retrieve your resource primary key from this page.

Export project
Language Studio
Get orchestration project details
Language Studio
\n![Image](images/page794_image1.png)
\nWhen you don't need your project anymore, you can delete your project using Language
Studio. Select Projects from the left pane, select the project you want to delete, and then

Delete project
Language Studio
\n![Image](images/page795_image1.png)
\nselect Delete from the top menu.
Build schema

Next Steps
\n![Image](images/page796_image1.png)
\nHow to build your project schema for
orchestration workflow
06/21/2025
In orchestration workflow projects, the schema is defined as the combination of intents within
your project. Schema design is a crucial part of your project's success. When creating a schema,
you want think about which intents that should be included in your project.
Consider the following guidelines and recommendations for your project:
Build orchestration projects when you need to manage the NLU for a multi-faceted virtual
assistant or chatbot, where the intents, entities, and utterances would begin to be far
more difficult to maintain over time in one project.
Orchestrate between different domains. A domain is a collection of intents and entities
that serve the same purpose, such as Email commands vs. Restaurant commands.
If there is an overlap of similar intents between domains, create the common intents in a
separate domain and removing them from the others for the best accuracy.
For intents that are general across domains, such as “Greeting”, “Confirm”, “Reject”, you
can either add them in a separate domain or as direct intents in the Orchestration project.
Orchestrate to Custom question answering knowledge base when a domain has FAQ type
questions with static answers. Ensure that the vocabulary and language used to ask
questions is distinctive from the one used in the other Conversational Language
Understanding projects and LUIS applications.
If an utterance is being misclassified and routed to an incorrect intent, then add similar
utterances to the intent to influence its results. If the intent is connected to a project, then
add utterances to the connected project itself. After you retrain your orchestration
project, the new utterances in the connected project will influence predictions.
Add test data to your orchestration projects to validate there isn’t confusion between
linked projects and other intents.
To build a project schema within Language Studio
:
1. Select Schema definition from the left side menu.
2. To create an intent, select Add from the top menu. You will be prompted to type in a
name for the intent.
Guidelines and recommendations
Add intents
\n3. To connect your intent to other existing projects, select Yes, I want to connect it to an
existing project option. You can alternatively create a non-connected intent by selecting
the No, I don't want to connect to a project option.
4. If you choose to create a connected intent, choose from Connected service the service
you are connecting to, then choose the project name. You can connect your intent to
only one project from the following services: CLU , LUIS or Question answering.
5. Select Add intent to add your intent.
Add utterances

 Tip
Use connected intents to connect to other projects (conversational language
understanding, LUIS, and question answering)
Next steps
\n![Image](images/page798_image1.png)
\nAdd utterances in Language Studio
06/21/2025
Once you have built a schema, you should add training and testing utterances to your project.
The utterances should be similar to what your users will use when interacting with the project.
When you add an utterance, you have to assign which intent it belongs to.
Adding utterances is a crucial step in project development lifecycle; this data will be used in the
next step when training your model so that your model can learn from the added data. If you
already have utterances, you can directly import it into your project, but you need to make sure
that your data follows the accepted data format. Labeled data informs the model how to
interpret text, and is used for training and evaluation.
A successfully created project.
See the project development lifecycle for more information.
Use the following steps to add utterances:
1. Go to your project page in Language Studio
.
2. From the left side menu, select Add utterances.
3. From the top pivots, you can change the view to be training set or testing set. Learn
more about training and testing sets and how they're used for model training and
evaluation.
4. From the Select intent dropdown menu, select one of the intents. Type in your utterance,
and press the enter key in the utterance's text box to add the utterance. You can also
upload your utterances directly by clicking on Upload utterance file from the top menu,
make sure it follows the accepted format.
Prerequisites
How to add utterances
７ Note
If you are planning on using Automatically split the testing set from training data
splitting, add all your utterances to the training set. You can add training utterances
to non-connected intents only.
\n5. Under Distribution you can view the distribution across training and testing sets. You can
view utterances per intent:
Utterance per non-connected intent
Utterances per connected intent
Train Model

Next Steps
\n![Image](images/page800_image1.png)