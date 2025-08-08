Connect different services with
Orchestration workflow
06/21/2025
Orchestration workflow is a feature that allows you to connect different projects from LUIS,
conversational language understanding, and custom question answering in one project. You
can then use this project for predictions under one endpoint. The orchestration project makes
a prediction on which project should be called and automatically routes the request to that
project, and returns with its response.
In this tutorial, you will learn how to connect a custom question answering knowledge base
with a conversational language understanding project. You will then call the project using the
.NET SDK sample for orchestration.
This tutorial will include creating a chit chat knowledge base and email commands project.
Chit chat will deal with common niceties and greetings with static responses. Email commands
will predict among a few simple actions for an email assistant. The tutorial will then teach you
to call the Orchestrator using the SDK in a .NET environment using a sample solution.
Create a Language resource
 and select the custom question answering feature in the
Azure portal to get your key and endpoint. After it deploys, select Go to resource.
You will need the key and endpoint from the resource you create to connect your bot
to the API. You'll paste your key and endpoint into the code below later in the tutorial.
Copy them from the Keys and Endpoint tab in your resource.
When you enable custom question answering, you must select an Azure search
resource to connect to.
Make sure the region of your resource is supported by conversational language
understanding.
Download the OrchestrationWorkflowSample sample
.
1. Sign into the Language Studio
 and select your Language resource.
2. Find and select the Custom question answering
 card in the homepage.
Prerequisites
Create a custom question answering knowledge
base
\n3. Select Create new project and add the name chitchat with the language English before
clicking on Create project.
4. When the project loads, select Add source and select Chit chat. Select the professional
personality for chit chat before
5. Go to Deploy knowledge base from the left pane and select Deploy and confirm the
popup that shows up.
You are now done with deploying your knowledge base for chit chat. You can explore the type
of questions and answers to expect in the Edit knowledge base page.
1. In Language Studio, go to the Conversational language understanding
 service.
2. Download the EmailProject.json  sample file here
.
3. Select the Import button. Browse to the `EmailProject.json`` file you downloaded and
press Done.

Create a conversational language understanding
project
\n![Image](images/page872_image1.png)
\n4. Once the project is loaded, select Training jobs on the left. Press on Start a training job,
provide the model name v1 and press Train.
5. Once training is complete, click to Deploying a model on the left. Select Add
Deployment and create a new deployment with the name Testing, and assign model v1
to the deployment.


\n![Image](images/page873_image1.png)

![Image](images/page873_image2.png)
\nYou are now done with deploying a conversational language understanding project for email
commands. You can explore the different commands in the Data labeling page.
1. In Language Studio, go to the Orchestration workflow
 service.
2. Select Create new project. Use the name Orchestrator and the language English before
clicking next then done.
3. Once the project is created, select Add in the Schema definition page.
4. Select Yes, I want to connect it to an existing project. Add the intent name EmailIntent and
select Conversational Language Understanding as the connected service. Select the
recently created EmailProject project for the project name before clicking on Add Intent.

Create an Orchestration workflow project
\n![Image](images/page874_image1.png)
\n5. Add another intent but now select Question Answering as the service and select chitchat
as the project name.
6. Similar to conversational language understanding, go to Training jobs and start a new
training job with the name v1 and press Train.
7. Once training is complete, click to Deploying a model on the left. Select Add deployment
and create a new deployment with the name Testing, and assign model v1 to the
deployment and press Next.
8. On the next page, select the deployment name Testing for the EmailIntent. This tells the
orchestrator to call the Testing deployment in EmailProject when it routes to it. Custom
question answering projects only have one deployment by default.

\n![Image](images/page875_image1.png)
\nNow your orchestration project is ready to be used. Any incoming request will be routed to
either EmailIntent and the EmailProject in conversational language understanding or
ChitChatIntent and the chitchat knowledge base.
1. In the downloaded sample, open OrchestrationWorkflowSample.sln in Visual Studio.
2. In the OrchestrationWorkflowSample solution, make sure to install all the required
packages. In Visual Studio, go to Tools, NuGet Package Manager and select Package
Manager Console and run the following command.
PowerShell
Alternatively, you can search for "Azure.AI.Language.Conversations" in the NuGet package
manager and install the latest release.
3. In Program.cs , replace {api-key}  and the {endpoint}  variables. Use the key and endpoint
for the Language resource you created earlier. You can find them in the Keys and
Endpoint tab in your Language resource in Azure.

Call the orchestration project with the
Conversations SDK
dotnet add package Azure.AI.Language.Conversations
\n![Image](images/page876_image1.png)
\nC#
4. Replace the project and deployment parameters to Orchestrator and Testing as below if
they are not set already.
C#
5. Run the project or press F5 in Visual Studio.
6. Input a query such as "read the email from matt" or "hello how are you". You'll now
observe different responses for each, a conversational language understanding
EmailProject response from the first query, and the answer from the chitchat knowledge
base for the second query.
Conversational Language Understanding:
Uri endpoint = new Uri("{endpoint}");
AzureKeyCredential credential = new AzureKeyCredential("{api-key}");
string projectName = "Orchestrator";
string deploymentName = "Testing";

\n![Image](images/page877_image1.png)
\nCustom Question Answering:
You can now connect other projects to your orchestrator and begin building complex
architectures with various different projects.
Learn more about conversational language understanding.
Learn more about custom question answering.

Next steps
\n![Image](images/page878_image1.png)
\nOrchestration workflow limits
06/21/2025
Use this article to learn about the data and service limits when using orchestration workflow.
Your Language resource has to be created in one of the supported regions.
Pricing tiers
Tier
Description
Limit
F0
Free tier
You are only allowed one Language resource with the F0 tier per subscription.
S
Paid tier
You can have up to 100 Language resources in the S tier per region.
See pricing
 for more information.
You can have up to 500 projects per resource.
Project names have to be unique within the same resource across all custom features.
See Language service regional availability.
Item
Request type
Maximum limit
Authoring API
POST
10 per minute
Authoring API
GET
100 per minute
Prediction API
GET/POST
1,000 per minute
Language resource limits
ﾉ
Expand table
Regional availability
API limits
ﾉ
Expand table
Quota limits
\nPricing tier
Item
Limit
F
Training time
1 hour per month
S
Training time
Unlimited, Standard
F
Prediction Calls
5,000 request per month
S
Prediction Calls
Unlimited, Standard
The following limits are observed for orchestration workflow.
Item
Lower Limit
Upper Limit
Count of utterances per project
1
15,000
Utterance length in characters
1
500
Count of intents per project
1
500
Count of trained models per project
0
10
Count of deployments per project
0
10
Attribute
Limits
Project name
You can only use letters (a-z, A-Z) , and numbers (0-9)  , symbols _ . - , with no
spaces. Maximum allowed length is 50 characters.
Model name
You can only use letters (a-z, A-Z) , numbers (0-9)  and symbols _ . - . Maximum
allowed length is 50 characters.
Deployment
name
You can only use letters (a-z, A-Z) , numbers (0-9)  and symbols _ . - . Maximum
allowed length is 50 characters.
Intent name
You can only use letters (a-z, A-Z) , numbers (0-9)  and all symbols except ":", $ & % *
( ) + ~ # / ? . Maximum allowed length is 50 characters.
ﾉ
Expand table
Data limits
ﾉ
Expand table
Naming limits
ﾉ
Expand table