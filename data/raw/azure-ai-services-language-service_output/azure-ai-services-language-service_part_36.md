Key
Placeholder
Value
Example
category
The type of entity associated with the span of text specified.
Entity1
offset
The inclusive character position of the start of the text.
0
length
The length of the bounding box in terms of UTF16 characters.
Training only considers the data in this region.
500
For more information on importing your labeled data into your project directly, see
Import project.
For more information about labeling your data, see Label your utterances in Language
Studio. After you label your data, you can train your model.
Related content
\nNone intent
06/30/2025
Every project in conversational language understanding includes a default None intent. The
None intent is a required intent and can't be deleted or renamed. The intent is meant to
categorize utterances that don't belong to any of your other custom intents.
An utterance can be predicted as the None intent if the top scoring intent's score is lower than
the None score threshold. It can also be predicted if the utterance is similar to examples added
to the None intent.
You can go to the project settings of any project and set the None score threshold. The
threshold is a decimal score from 0.0 to 1.0.
For any query and utterance, the highest scoring intent ends up lower than the threshold score,
so the top intent is automatically replaced with the None intent. The scores of all the other
intents remain unchanged.
The score should be set according to your own observations of prediction scores because they
might vary by project. A higher threshold score forces the utterances to be more similar to the
examples you have in your training data.
When you export a project's JSON file, the None score threshold is defined in the settings
parameter of the JSON as the confidenceThreshold . The threshold accepts a decimal value
between 0.0 and 1.0.
The None intent is also treated like any other intent in your project. If there are utterances that
you want predicted as None, consider adding similar examples to them in your training data. If
you want to categorize utterances that aren't important to your project as None, add those
utterances to your intent. Examples might include greetings, yes-and-no answers, and
responses to questions such as providing a number.
None score threshold
７ Note
During model evaluation of your test set, the None score threshold isn't applied.
Add examples to the None intent
\nYou should also consider adding false positive examples to the None intent. For example, in a
flight booking project it's likely that the utterance "I want to buy a book" could be confused
with a Book Flight intent. You can add "I want to buy a book" or "I love reading books" as None
training utterances. They help to alter the predictions of those types of utterances toward the
None intent instead of Book Flight.
Conversational language understanding overview
Related content
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
\nWhen to use conversational language
understanding or orchestration workflow
apps
06/30/2025
When you create large applications, you should consider whether your use case is best served
by a single conversational app (flat architecture) or by multiple apps that are orchestrated.
Orchestration workflow is a feature that allows you to connect different projects from LUIS,
conversational language understanding, and custom question answering in one project. You
can then use this project for predictions by using one endpoint. The orchestration project
makes a prediction on which child project should be called, automatically routes the request,
and returns with its response.
Orchestration involves two steps:
1. Predicting which child project to call.
2. Routing the utterance to the destination child app and returning the child app's response.
Clear decomposition and faster development:
If your overall schema has a substantial number of domains, the orchestration
approach can help decompose your application into several child apps (each serving a
specific domain). For example, an automotive conversational app might have a
navigation domain or a media domain.
Developing each domain app in parallel is easier. People and teams with specific
domain expertise can work on individual apps collaboratively and in parallel.
Because each domain app is smaller, the development cycle becomes faster. Smaller-
sized domain apps take much less time to train than a single large app.
More flexible confidence score thresholds:
Because separate child apps serve each domain, it's easy to set separate thresholds for
different child apps.
AI-quality improvements where appropriate:
Orchestration overview
Orchestration advantages
\nSome applications require that certain entities must be domain restricted.
Orchestration makes this task easy to achieve. After the orchestration project predicts
which child app should be called, the other child apps aren't called.
For example, if your app contains a Person.Name  prebuilt entity, consider the utterance
"How do I use a jack?" in the context of a vehicle question. In this context, jack is an
automotive tool and shouldn't be recognized as a person's name. When you use
orchestration, this utterance can be redirected to a child app created to answer such a
question, which doesn't have a Person.Name  entity.
Redundant entities in child apps:
If you need a particular prebuilt entity being returned in all utterances irrespective of
the domain, for example Quantity.Number  or Geography.Location , there's no way of
adding an entity to the orchestration app (it's an intent-only model). You would need
to add it to all individual child apps.
Efficiency:
Orchestration apps take two model inferences. One for predicting which child app to
call, and another for the prediction in the child app. Inference times are typically slower
than single apps with a flat architecture.
Train/test split for orchestrator:
Training an orchestration app doesn't allow you to granularly split data between the
testing and training sets. For example, you can't train a 90-10 split for child app A, and
then train an 80-20 split for child app B. This limitation might be minor, but it's worth
keeping in mind.
Flat architecture is the other method of developing conversational apps. Instead of using an
orchestration app to send utterances to one of multiple child apps, you develop a singular (or
flat) app to handle utterances.
Simplicity:
For small-sized apps or domains, the orchestrator approach can be overly complex.
Because all intents and entities are at the same app level, it might be easier to make
changes to all of them together.
Orchestration disadvantages
Flat architecture overview
Flat architecture advantages
\nIt's easier to add entities that should always be returned:
If you want certain prebuilt or list entities to be returned for all utterances, you only
need to add them alongside other entities in a single app. If you use orchestration, as
mentioned, you need to add it to every child app.
Unwieldy for large apps:
For large apps (say, more than 50 intents or entities), it can become difficult to keep
track of evolving schemas and datasets. This difficulty is evident in cases where the app
has to serve several domains. For example, an automotive conversational app might
have a navigation domain or a media domain.
Limited control over entity matches:
In a flat architecture, there's no way to restrict entities to be returned only in certain
cases. When you use orchestration, you can assign those specific entities to particular
child apps.
Orchestration workflow overview
Conversational language understanding overview
Flat architecture disadvantages
Related content
\nIntegrate conversational language
understanding with Bot Framework
06/04/2025
A dialog is the interaction that occurs between user queries and an application. Dialog
management is the process that defines the automatic behavior that should occur for different
customer interactions. While conversational language understanding can classify intents and
extract information through entities, the Bot Framework SDK allows you to configure the
applied logic for the responses returned from it.
This tutorial will explain how to integrate your own conversational language understanding
(CLU) project for a flight booking project in the Bot Framework SDK that includes three intents:
Book Flight, Get Weather, and None.
Create a Language resource
 in the Azure portal to get your key and endpoint. After it
deploys, select Go to resource.
You will need the key and endpoint from the resource you create to connect your bot
to the API. You'll paste your key and endpoint into the code below later in the tutorial.
Download the CoreBotWithCLU sample
.
Clone the entire samples repository to get access to this solution.
1. Download the FlightBooking.json
 file in the Core Bot with CLU sample, in the Cognitive
Models folder.
2. Sign into the Language Studio
 and select your Language resource.
3. Navigate to Conversational Language Understanding
 and select the service. This will
route you the projects page. Select the Import button next to the Create New Project
button. Import the FlightBooking.json file with the project name as FlightBooking. This
will automatically import the CLU project with all the intents, entities, and utterances.
Prerequisites
Import a project in conversational language
understanding