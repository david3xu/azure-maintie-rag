Custom text classification overview
\nWhat is conversational language
understanding?
05/19/2025
Conversational language understanding is one of the custom features offered by Azure AI
Language. It's a cloud-based API service that applies machine-learning intelligence to enable
you to build natural language understanding component to be used in an end-to-end
conversational application.
Conversational language understanding (CLU) enables users to build custom natural language
understanding models to predict the overall intention of an incoming utterance and extract
important information from it. CLU only provides the intelligence to understand the input text
for the client application and doesn't perform any actions. By creating a CLU project,
developers can iteratively label utterances, train and evaluate model performance before
making it available for consumption. The quality of the labeled data greatly impacts model
performance. To simplify building and customizing your model, the service offers a custom web
portal that can be accessed through the Azure AI Foundry
. You can easily get started with the
service by following the steps in this quickstart.
This documentation contains the following article types:
Quickstarts are getting-started instructions to guide you through making requests to the
service.
Concepts provide explanations of the service functionality and features.
How-to guides contain instructions for using the service in more specific or customized
ways.
CLU can be used in multiple scenarios across various industries. Some examples are:
Use CLU to build and train a custom natural language understanding model based on a
specific domain and the expected users' utterances. Integrate it with any end-to-end
conversational bot so that it can process and analyze incoming text in real time to identify the
intention of the text and extract important information from it. Have the bot perform the
desired action based on the intention and extracted information. An example would be a
customized retail bot for online shopping or food ordering.
Example usage scenarios
End-to-end conversational bot
\nOne example of a human assistant bot is to help staff improve customer engagements by
triaging customer queries and assigning them to the appropriate support engineer. Another
example would be a human resources bot in an enterprise that allows employees to
communicate in natural language and receive guidance based on the query.
When you integrate a client application with a speech to text component, users can speak a
command in natural language for CLU to process, identify intent, and extract information from
the text for the client application to perform an action. This use case has many applications,
such as to stop, play, forward, and rewind a song or turn lights on or off.
In a large corporation, an enterprise chat bot may handle various employee affairs. It might
handle frequently asked questions served by a custom question answering knowledge base, a
calendar specific skill served by conversational language understanding, and an interview
feedback skill served by LUIS. Use Orchestration workflow to connect all these skills together
and appropriately route the incoming requests to the correct service.
CLU is utilized by the intent routing
 agent template, which detects user intent and provides
exact answering. Perfect for deterministically intent routing and exact question answering with
human control.
Creating a CLU project typically involves several different steps.
Human assistant bots
Command and control application
Enterprise chat bot
Agents
Project development lifecycle
\nCLU offers two paths for you to get the most out of your implementation.
Option 1 (LLM-powered quick deploy):
1. Define your schema: Know your data and define the actions and relevant information
that needs to be recognized from user's input utterances. In this step, you create
the intents and provide a detailed description on the meaning of your intents that you
want to assign to user's utterances.
2. Deploy the model: Deploying a model with the LLM-based training config makes it
available for use via the Runtime API.
3. Predict intents and entities: Use your custom model deployment to predict custom
intents and prebuilt entities from user’s utterances.
Option 2 (Custom machine learned model)
Follow these steps to get the most out of your trained model:
1. Define your schema: Know your data and define the actions and relevant information
that needs to be recognized from user's input utterances. In this step, you create the
intents that you want to assign to user's utterances, and the relevant entities you want
extracted.
2. Label your data: The quality of data labeling is a key factor in determining model
performance.

７ Note
In the Azure AI Foundry, you’ll create a fine-tuning task as your workspace for customizing
your CLU model. Formerly, a CLU fine-tuning task was called a CLU project. You may see
these terms used interchangeably in legacy CLU documentation.
\n![Image](images/page184_image1.png)
\n3. Train the model: Your model starts learning from your labeled data.
4. View the model's performance: View the evaluation details for your model to determine
how well it performs when introduced to new data.
5. Improve the model: After reviewing the model's performance, you can then learn how
you can improve the model.
6. Deploy the model: Deploying a model makes it available for use via the Runtime API
.
7. Predict intents and entities: Use your custom model to predict intents and entities from
user's utterances.
As you use CLU, see the following reference documentation and samples for Azure AI
Language:
Development option / language
Reference documentation
Samples
REST APIs (Authoring)
REST API documentation
REST APIs (Runtime)
REST API documentation
C# (Runtime)
C# documentation
C# samples
Python (Runtime)
Python documentation
Python samples
An AI system includes not only the technology, but also the people who use it, the people who
are affected by it, and the environment in which it's deployed. Read the transparency note for
CLU to learn about responsible AI use and deployment in your systems. You can also see the
following articles for more information:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
Reference documentation and code samples
ﾉ
Expand table
Responsible AI
Next steps
\nUse the quickstart article to start using conversational language understanding.
As you go through the project development lifecycle, review the glossary to learn more
about the terms used throughout the documentation for this feature.
Remember to view the service limits for information such as regional availability.
\nQuickstart: Conversational language
understanding
Article • 04/29/2025
Use this article to get started with Conversational Language understanding using Language
Studio and the REST API. Follow these steps to try out an example.
Azure subscription - Create one for free
.
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
Your Azure resource group name.
Azure resource
name
Your Azure resource name.
Location
One of the supported regions for your Language resource. For example "West
US 2".
Pricing tier
One of the valid pricing tiers for your Language resource. You can use the Free
(F0) tier to try the service.
Prerequisites
Sign in to Language Studio
ﾉ
Expand table
\nOnce you have a Language resource selected, create a conversational language understanding
project. A project is a work area for building your custom ML models based on your data. Your
project can only be accessed by you and others who have access to the Language resource
being used.
For this quickstart, you can download this sample project file
 and import it. This project can
predict the intended commands from user input, such as: reading emails, deleting emails, and
attaching a document to an email.
1. Under the Understand questions and conversational language section of Language
Studio, select Conversational language understanding.

Create a conversational language understanding
project
\n![Image](images/page188_image1.png)
\n2. This will bring you to the Conversational language understanding projects page. Next to
the Create new project button select Import.
3. In the window that appears, upload the JSON file you want to import. Make sure that
your file follows the supported JSON format.
Once the upload is complete, you will land on Schema definition page. For this quickstart, the
schema is already built, and utterances are already labeled with intents and entities.
Typically, after you create a project, you should build a schema and label utterances. For this
quickstart, we already imported a ready project with built schema and labeled utterances.
To train a model, you need to start a training job. The output of a successful training job is your
trained model.
To start training your model from within the Language Studio
:
1. Select Train model from the left side menu.
2. Select Start a training job from the top menu.


Train your model
\n![Image](images/page189_image1.png)

![Image](images/page189_image2.png)
\n3. Select Train a new model and enter a new model name in the text box. Otherwise to
replace an existing model with a model trained on the new data, select Overwrite an
existing model and then select an existing model. Overwriting a trained model is
irreversible, but it won't affect your deployed models until you deploy the new model.
4. Select training mode. You can choose Standard training for faster training, but it is only
available for English. Or you can choose Advanced training which is supported for other
languages and multilingual projects, but it involves longer training times. Learn more
about training modes.
5. Select a data splitting method. You can choose Automatically splitting the testing set
from training data where the system will split your utterances between the training and
testing sets, according to the specified percentages. Or you can Use a manual split of
training and testing data, this option is only enabled if you have added utterances to
your testing set when you labeled your utterances.
6. Select the Train button.
7. Select the training job ID from the list. A panel will appear where you can check the
training progress, job status, and other details for this job.

７ Note
Only successfully completed training jobs will generate models.
Training can take some time between a couple of minutes and couple of hours
based on the count of utterances.
You can only have one training job running at a time. You can't start other
training jobs within the same project until the running job is completed.
\n![Image](images/page190_image1.jpeg)