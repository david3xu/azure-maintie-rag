Key
Placeholder
Value
Example
api-version
{API-VERSION}
The version of the API you are calling. The value
referenced here is for the latest released model
version released.
2022-03-
01-preview
confidenceThreshold
{CONFIDENCE-
THRESHOLD}
This is the threshold score below which the intent
will be predicted as none intent
0.7
projectName
{PROJECT-NAME}
The name of your project. This value is case-
sensitive.
EmailApp
multilingual
false
Orchestration doesn't support the multilingual
feature
false
language
{LANGUAGE-
CODE}
A string specifying the language code for the
utterances used in your project. See Language
support for more information about supported
language codes.
en-us
intents
[]
Array containing all the intent types you have in
the project. These are the intents used in the
orchestration project.
[]
JSON
        "text": "utterance 1",
        "language": "{LANGUAGE-CODE}",
        "dataset": "{DATASET}",
        "intent": "intent1"
      }
    ]
  }
}
ﾉ
Expand table
Utterance format
[
    {
        "intent": "intent1",
        "language": "{LANGUAGE-CODE}",
        "text": "{Utterance-Text}",
    },
    {
        "intent": "intent2",
        "language": "{LANGUAGE-CODE}",
        "text": "{Utterance-Text}",
\nYou can import your labeled data into your project directly. Learn how to import project
See the how-to article more information about labeling your data. When you're done
labeling your data, you can train your model.
    }
]
Next steps
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