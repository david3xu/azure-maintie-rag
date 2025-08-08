If you choose to projects with multiple languages in a single language resource, there is a
dedicated test index per project. So the limit is applied per project in the language service.
Azure AI Search tier
Free
Basic
S1
S2
S3
S3
HD
Maximum metadata fields per language service (per
project)
1,000
100*
1,000
1,000
1,000
1,000
If you don't choose the option to have projects with multiple different languages, then the
limits are applied across all projects in the language service.
Azure AI Search tier
Free
Basic
S1
S2
S3
S3
HD
Maximum metadata fields per Language service (across
all projects)
1,000
100*
1,000
1,000
1,000
1,000
The length and acceptable characters for metadata name and value are listed in the following
table.
Item
Allowed chars
Regex pattern match
Max chars
Name (key)
Allows
Alphanumeric (letters and digits)
_  (underscore)
Must not contain spaces.
^[a-zA-Z0-9_]+$
100
Value
Allows everything except
:  (colon)
|  (vertical pipe)
Only one value allowed.
^[^:|]+$
500
Overall limits on the content in the project:
ﾉ
Expand table
ﾉ
Expand table
By name and value
ﾉ
Expand table
Project content limits
\nLength of answer text: 25,000 characters
Length of question text: 1,000 characters
Length of metadata key text: 100 characters
Length of metadata value text: 500 characters
Supported characters for metadata name: Alphabets, digits, and _
Supported characters for metadata value: All except :  and |
Length of file name: 200
Supported file formats: ".tsv", ".pdf", ".txt", ".docx", ".xlsx".
Maximum number of alternate questions: 300
Maximum number of question-answer pairs: Depends on the Azure AI Search tier
chosen. A question and answer pair maps to a document on Azure AI Search index.
URL/HTML page: 1 million characters
These represent the limits for each create project action; that is, selecting Create new project or
calling the REST API to create a project.
Recommended maximum number of alternate questions per answer: 300
Maximum number of URLs: 10
Maximum number of files: 10
Maximum number of QnAs permitted per call: 1000
These represent the limits for each update action; that is, selecting Save or calling the REST API
with an update request.
Length of each source name: 300
Recommended maximum number of alternate questions added or deleted: 300
Maximum number of metadata fields added or deleted: 10
Maximum number of URLs that can be refreshed: 5
Maximum number of QnAs permitted per call: 1000
Create project call limits:
Update project call limits
Add unstructured file limits
７ Note
\nThese represent the limits when unstructured files are used to Create new project or call the
REST API to create a project:
Length of file: We will extract first 32000 characters
Maximum three responses per file.
These represent the limits when REST API is used to answer a question based without having to
create a project:
Number of documents: 5
Maximum size of a single document: 5,120 characters
Maximum three responses per document.
If you need to use larger files than the limit allows, you can break the file into smaller
files before sending them to the API.
Prebuilt custom question answering limits
７ Note
If you need to use larger documents than the limit allows, you can break the text into
smaller chunks of text before sending them to the API.
A document is a single string of text characters.
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
\nTutorial: Create an FAQ bot
06/04/2025
Create an FAQ Bot with custom question answering and Azure Bot Service
 with no code.
In this tutorial, you learn how to:
Follow the getting started article. Once the project has been successfully deployed, you will be
ready to start this article.
After deploying your project, you can create a bot from the Deploy project page:
You can create several bots quickly, all pointing to the same project for different regions
or pricing plans for the individual bots.
When you make changes to the project and redeploy, you don't need to take further
action with the bot. It's already configured to work with the project, and works with all
future changes to the project. Every time you publish a project, all the bots connected to
it are automatically updated.
1. In Language Studio
, on the custom question answering Deploy project page, select the
Create a bot button.
Link a custom question answering project to an Azure AI Bot Service
＂
Deploy a Bot
＂
Chat with the Bot in web chat
＂
Enable the Bot in supported channels
＂
Create and publish a project
Create a bot
\n2. A new browser tab opens for the Azure portal, with the Azure AI Bot Service's creation
page. Configure the Azure AI Bot Service and hit the Create button.
Setting
Value
Bot handle
Unique identifier for your bot. This value needs to be distinct from your App
name
Subscription
Select your subscription
Resource group
Select an existing resource group or create a new one
Location
Select your desired location
Pricing tier
Choose pricing tier
App name
App service name for your bot
SDK language
C# or Node.js. Once the bot is created, you can download the code to your
local development environment and continue the development process.
Language Resource
Key
This key is automatically populated deployed custom question answering
project
App service
plan/Location
This value is automatically populated, do not change this value
3. After the bot is created, open the Bot service resource.
ﾉ
Expand table
\n![Image](images/page1158_image1.png)
\n4. Under Settings, select Test in Web Chat.
5. At the chat prompt of Type your message, enter:
How do I setup my surface book?
The chat bot responds with an answer from your project.
\n![Image](images/page1159_image1.png)
\nSelect Channels in the Bot service resource that you have created. You can activate the Bot in
additional supported channels.
Integrate the bot with channels
\n![Image](images/page1160_image1.png)