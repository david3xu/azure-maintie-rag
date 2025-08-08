Terms and definitions used in conversation
language understanding
06/30/2025
Use this article to learn about some of the definitions and terms you may encounter when
using conversation language understanding.
Entities are words in utterances that describe information used to fulfill or identify an intent. If
your entity is complex and you would like your model to identify specific parts, you can break
your model into subentities. For example, you might want your model to predict an address,
but also the subentities of street, city, state, and zipcode.
The F1 score is a function of Precision and Recall. It's needed when you seek a balance between
precision and recall.
An intent represents a task or action the user wants to perform. It's a purpose or goal
expressed in a user's input, such as booking a flight, or paying a bill.
A list entity represents a fixed, closed set of related words along with their synonyms. List
entities are exact matches, unlike machined learned entities.
The entity will be predicted if a word in the list entity is included in the list. For example, if you
have a list entity called "size" and you have the words "small, medium, large" in the list, then
the size entity will be predicted for all utterances where the words "small", "medium", or "large"
are used regardless of the context.
A model is an object that's trained to do a certain task, in this case conversation understanding
tasks. Models are trained by providing labeled data to learn from so they can later be used to
Entity
F1 score
Intent
List entity
Model
\nunderstand utterances.
Model evaluation is the process that happens right after training to know how well does
your model perform.
Deployment is the process of assigning your model to a deployment to make it available
for use via the prediction API
.
Overfitting happens when the model is fixated on the specific examples and isn't able to
generalize well.
Measures how precise/accurate your model is. It's the ratio between the correctly identified
positives (true positives) and all identified positives. The precision metric reveals how many of
the predicted classes are correctly labeled.
A project is a work area for building your custom ML models based on your data. Your project
can only be accessed by you and others who have access to the Azure resource being used.
Measures the model's ability to predict actual positive classes. It's the ratio between the
predicted true positives and what was actually tagged. The recall metric reveals how many of
the predicted classes are correct.
A regular expression entity represents a regular expression. Regular expression entities are
exact matches.
Schema is defined as the combination of intents and entities within your project. Schema
design is a crucial part of your project's success. When creating a schema, you want to think
about which intents and entities should be included in your project.
Overfitting
Precision
Project
Recall
Regular expression
Schema
\nTraining data is the set of information that is needed to train a model.
An utterance is user input that is short text representative of a sentence in a conversation. It's a
natural language phrase such as "book 2 tickets to Seattle next Tuesday". Example utterances
are added to train the model and the model predicts on new utterance at runtime
Data and service limits.
Conversation language understanding overview.
Training data
Utterance
Next steps
\nWhat is entity linking in Azure AI
Language?
06/04/2025
Entity linking is one of the features offered by Azure AI Language, a collection of machine
learning and AI algorithms in the cloud for developing intelligent applications that involve
written language. Entity linking identifies and disambiguates the identity of entities found in
text. For example, in the sentence "We went to Seattle last week.", the word "Seattle" would be
identified, with a link to more information on Wikipedia.
This documentation contains the following types of articles:
Quickstarts are getting-started instructions to guide you through making requests to the
service.
How-to guides contain instructions for using the service in more specific ways.
To use entity linking, you submit raw unstructured text for analysis and handle the API output
in your application. Analysis is performed as-is, with no additional customization to the model
used on your data. There are two ways to use entity linking:
Development option
Description
Language studio
Language Studio is a web-based platform that lets you try entity linking with text
examples without an Azure account, and your own data when you sign up. For
more information, see the Language Studio website
.
REST API or Client
library (Azure SDK)
Integrate entity linking into your applications using the REST API, or the client
library available in a variety of languages. For more information, see the entity
linking quickstart.
As you use this feature in your applications, see the following reference documentation and
samples for Azure AI Language:
Get started with entity linking
ﾉ
Expand table
Reference documentation and code samples
ﾉ
Expand table
\nDevelopment option / language
Reference documentation
Samples
REST API
REST API documentation
C#
C# documentation
C# samples
Java
Java documentation
Java Samples
JavaScript
JavaScript documentation
JavaScript samples
Python
Python documentation
Python samples
An AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Read the transparency
note for entity linking to learn about responsible AI use and deployment in your systems. You
can also see the following articles for more information:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
There are two ways to get started using the entity linking feature:
Language Studio
, which is a web-based platform that enables you to try several Azure
AI Language features without needing to write code.
The quickstart article for instructions on making requests to the service using the REST
API and client library SDK.
Responsible AI
Next steps
\nQuickstart: Entity Linking using the client
library and REST API
06/21/2025
Reference documentation | More samples
 | Package (NuGet)
 | Library source code
Use this quickstart to create an entity linking application with the client library for .NET. In the
following example, you create a C# application that can identify and disambiguate entities
found in text.
Azure subscription - Create one for free
The Visual Studio IDE
To use the code sample below, you'll need to deploy an Azure resource. This resource will
contain a key and endpoint you'll use to authenticate the API calls you send to the Language
service.
1. Use the following link to create a language resource
 using the Azure portal. You will
need to sign in using your Azure subscription.
2. On the Select additional features screen that appears, select Continue to create your
resource.
 Tip
You can use Azure AI Foundry to try summarization without needing to write code.
Prerequisites
Setting up
Create an Azure resource
\n3. In the Create language screen, provide the following information:
Detail
Description
Subscription
The subscription account that your resource will be associated with. Select your
Azure subscription from the drop-down menu.
Resource
group
A resource group is a container that stores the resources you create. Select Create
new to create a new resource group.
Region
The location of your Language resource. Different regions may introduce latency
depending on your physical location, but have no impact on the runtime availability
of your resource. For this quickstart, either select an available region near you, or
choose East US.
Name
The name for your Language resource. This name will also be used to create an
endpoint URL that your applications will use to send API requests.
Pricing tier
The pricing tier
 for your Language resource. You can use the Free F0 tier to try
the service and upgrade later to a paid tier for production.

ﾉ
Expand table
\n![Image](images/page377_image1.png)
\n4. Make sure the Responsible AI Notice checkbox is checked.
5. Select Review + Create at the bottom of the page.
6. In the screen that appears, make sure the validation has passed, and that you entered
your information correctly. Then select Create.
Next you will need the key and endpoint from the resource to connect your application to the
API. You'll paste your key and endpoint into the code later in the quickstart.
1. After the Language resource deploys successfully, click the Go to Resource button under
Next Steps.

Get your key and endpoint
\n![Image](images/page378_image1.png)
\n2. On the screen for your resource, select Keys and endpoint on the left pane. You will use
one of your keys and your endpoint in the steps below.
Your application must be authenticated to send API requests. For production, use a secure way
of storing and accessing your credentials. In this example, you will write your credentials to
environment variables on the local machine running the application.
To set the environment variable for your Language resource key, open a console window, and
follow the instructions for your operating system and development environment.
To set the LANGUAGE_KEY  environment variable, replace your-key  with one of the keys for
your resource.
To set the LANGUAGE_ENDPOINT  environment variable, replace your-endpoint  with the
endpoint for your resource.


Create environment variables
\n![Image](images/page379_image1.png)

![Image](images/page379_image2.png)
\nConsole
Console
After you add the environment variables, you might need to restart any running programs
that will need to read the environment variables, including the console window. For
example, if you are using Visual Studio as your editor, restart Visual Studio before running
the example.
Using the Visual Studio IDE, create a new .NET Core console app. This will create a "Hello
World" project with a single C# source file: program.cs.
） Important
We recommend Microsoft Entra ID authentication with managed identities for Azure
resources to avoid storing credentials with your applications that run in the cloud.
Use API keys with caution. Don't include the API key directly in your code, and never post
it publicly. If using API keys, store them securely in Azure Key Vault, rotate the keys
regularly, and restrict access to Azure Key Vault using role based access control and
network access restrictions. For more information about using API keys securely in your
apps, see API keys with Azure Key Vault.
For more information about AI services security, see Authenticate requests to Azure AI
services.
Windows
setx LANGUAGE_KEY your-key
setx LANGUAGE_ENDPOINT your-endpoint
７ Note
If you only need to access the environment variables in the current running console,
you can set the environment variable with set  instead of setx .
Create a new .NET Core application