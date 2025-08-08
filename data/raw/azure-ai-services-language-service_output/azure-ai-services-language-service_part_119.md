The language for the incoming user query can be detected with the Language Detection API
and the user can call the appropriate endpoint and project depending on the detected
language.

\n![Image](images/page1181_image1.png)
\nAdd multiple categories to your FAQ bot
06/21/2025
In this tutorial, you learn how to:
When building a FAQ bot, you may encounter use cases that require you to address queries
across multiple domains. Let's say the marketing team at Microsoft wants to build a customer
support bot that answers common user queries on multiple Surface Products. For the sake of
simplicity here, we will be using two FAQ URLs, Surface Pen
, and Surface Earbuds
 to create
the project.
The content authors can use documents to extract question answer pairs or add custom
question answer pairs to the project. In order to group these question and answers into
specific domains or categories, you can add metadata.
For the bot on Surface products, you can take the following steps to create a bot that answers
queries for both product types:
1. Add the following FAQ URLs as sources by selecting Add source > URLs > and then Add
all once you have added each of the URLS below:
Surface Pen FAQ
Surface Earbuds FAQ
Create a project and tag question answer pairs into distinct categories with metadata
＂
Create a separate project for each domain
＂
Create a separate language resource for each domain
＂
Create project with domain specific metadata

\n![Image](images/page1182_image1.png)
\n2. In this project, we have question answer pairs on two products and we would like to
distinguish between them such that we can search for responses among question and
answers for a given product. In order to do this, we could update the metadata field for
the question answer pairs.
As you can see in the example below, we have added a metadata with product as key and
surface_pen or surface_earbuds as values wherever applicable. You can extend this
example to extract data on multiple products and add a different value for each product.
3. Now, in order to restrict the system to search for the response across a particular product
you would need to pass that product as a filter in the custom question answering REST
API.
The REST API prediction URL can be retrieved from the Deploy project pane:

\n![Image](images/page1183_image1.png)
\nIn the JSON body for the API call, we have passed surface_pen as value for the metadata
product. So, the system will only look for the response among the QnA pairs with the
same metadata.
JSON

    {
      "question": "What is the price?",
      "top": 3
    },
    "answerSpanRequest": {
      "enable": true,
      "confidenceScoreThreshold": 0.3,
      "topAnswersWithSpan": 1
    },
    "filters": {
      "metadataFilter": {
        "metadata": [
          {
            "key": "product",
            "value": "surface_pen"
\n![Image](images/page1184_image1.png)
\nYou can obtain metadata value based on user input in the following ways:
Explicitly take the domain as input from the user through the bot client. For instance
as shown below, you can take product category as input from the user when the
conversation is initiated.
Implicitly identify domain based on bot context. For instance, in case the previous
question was on a particular Surface product, it can be saved as context by the
client. If the user doesn't specify the product in the next query, you could pass on
the bot context as metadata to the Generate Answer API.
          }
        ]
      }
    }
\n![Image](images/page1185_image1.png)
\nExtract entity from user query to identify domain to be used for metadata filter. You
can use other Azure AI services such as Named Entity Recognition (NER) and
conversational language understanding for entity extraction.
You can add up to 50000 question answer pairs to a single project. If your data exceeds 50,000
question answer pairs, you should consider splitting the project.
You can also create a separate project for each domain and maintain the projects separately.
All APIs require for the user to pass on the project name to make any update to the project or
How large can our projects be?
Create a separate project for each domain
\n![Image](images/page1186_image1.png)

![Image](images/page1186_image2.png)
\nfetch an answer to the user's question.
When the user question is received by the service, you would need to pass on the projectName
in the REST API endpoint shown to fetch a response from the relevant project. You can locate
the URL in the Deploy project page under Get prediction URL:
https://southcentralus.api.cognitive.microsoft.com/language/:query-knowledgebases?
projectName=Test-Project-English&api-version=2021-10-01&deploymentName=production
Let's say the marketing team at Microsoft wants to build a customer support bot that answers
user queries on Surface and Xbox products. They plan to assign distinct teams to access
projects on Surface and Xbox. In this case, it is advised to create two custom question
answering resources - one for Surface and another for Xbox. You can however define distinct
roles for users accessing the same resource.
Create a separate language resource for each
domain
\nAdd your custom question answering
project to Power Virtual Agents
06/21/2025
Create and extend a Power Virtual Agents
 bot to provide answers from your project.
In this tutorial, you learn how to:
1. Follow the quickstart to create a custom question answering project. Once you have
deployed your project.
2. After deploying your project from Language Studio, select “Get Prediction URL”.
3. Get your Site URL from the hostname of Prediction URL and your Account key which
would be the Ocp-Apim-Subscription-Key.
７ Note
The integration demonstrated in this tutorial is in preview and is not intended for
deployment to production environments.
Create a Power Virtual Agents bot
＂
Create a system fallback topic
＂
Add custom question answering as an action to a topic as a Power Automate flow
＂
Create a Power Automate solution
＂
Add a Power Automate flow to your solution
＂
Publish Power Virtual Agents
＂
Test Power Virtual Agents, and receive an answer from your custom question answering
project
＂
７ Note
The QnA Maker service is being retired on the 31st of March, 2025. A newer version of the
question and answering capability is now available as part of Azure AI Language. For
custom question answering capabilities within the Language Service, see custom question
answering. Starting 1st October, 2022 you won’t be able to create new QnA Maker
resources. For information on migrating existing QnA Maker knowledge bases to custom
question answering, consult the migration guide.
Create and publish a project
\n4. Create a custom question answering connector: Follow the connector documentation to
create a connection to question answering.
5. Use this tutorial to create a Bot with Power Virtual Agents instead of creating a bot from
Language Studio.
Power Virtual Agents
 allows teams to create powerful bots by using a guided, no-code
graphical interface. You don't need data scientists or developers.
Create a bot by following the steps in Create and delete Power Virtual Agents bots.
In Power Virtual Agents, you create a bot with a series of topics (subject areas), in order to
answer user questions by performing actions.
Although the bot can connect to your project from any topic, this tutorial uses the system
fallback topic. The fallback topic is used when the bot can't find an answer. The bot passes the

Create a bot in Power Virtual Agents
Create the system fallback topic
\n![Image](images/page1189_image1.png)
\nuser's text to custom question answering Query knowledgebase API, receives the answer from
your project, and displays it to the user as a message.
Create a fallback topic by following the steps in Configure the system fallback topic in Power
Virtual Agents.
Use the Power Virtual Agents authoring canvas to connect the fallback topic to your project.
The topic starts with the unrecognized user text. Add an action that passes that text to custom
question answering, and then shows the answer as a message. The last step of displaying an
answer is handled as a separate step, later in this tutorial.
This section creates the fallback topic conversation flow.
The new fallback action might already have conversation flow elements. Delete the Escalate
item by selecting the Options menu.
Below the Message node, select the (+) icon, then select Call an action.
Use the authoring canvas to add an action

\n![Image](images/page1190_image1.png)