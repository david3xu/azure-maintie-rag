"What is the market value of Microsoft stock?"
"What is the market value of a Microsoft share?"
However, please note that the confidence score with which the system returns the correct
response will vary based on the input query and how different it is from the original
question answer pair.
There are certain scenarios which require the customer to add an alternate question.
When a query does not return the correct answer despite it being present in the project,
we advise adding that query as an alternate question to the intended QnA pair.
Users can add up to 10 alternate questions depending on their scenario. Alternate
questions beyond the first 10 aren’t considered by our core ranker. However, they are
evaluated in the other processing layers resulting in better output overall. All the alternate
questions will be considered in the preprocessing step to look for an exact match.
Semantic understanding in custom question answering should be able to take care of
similar alternate questions.
The return on investment will start diminishing once you exceed 10 questions. Even if
you’re adding more than 10 alternate questions, try to make the initial 10 questions as
semantically dissimilar as possible so that all intents for the answer are captured by these
10 questions. For the project above, in QNA #1, adding alternate questions such as "How
can I buy a car?", "I wanna buy a car." are not required. Whereas adding alternate
questions such as "How to purchase a car.", "What are the options for buying a vehicle?"
can be useful.
Custom question answering provides the flexibility to use synonyms at the project level,
unlike QnA Maker where synonyms are shared across projects for the entire service.
For better relevance, the customer needs to provide a list of acronyms that the end user
intends to use interchangeably. For instance, the following is a list of acceptable
acronyms:
MSFT – Microsoft
ID – Identification
How many alternate questions per QnA is optimal?
When to add synonyms to a project
\nETA – Estimated time of Arrival
Apart from acronyms, if you think your words are similar in context of a particular domain
and generic language models won’t consider them similar, it’s better to add them as
synonyms. For instance, if an auto company producing a car model X receives queries
such as "my car’s audio isn’t working" and the project has questions on "fixing audio for
car X", then we need to add "X" and "car" as synonyms.
The Transformer based model already takes care of most of the common synonym cases,
for e.g.- Purchase – Buy, Sell - Auction, Price – Value. For example, consider the following
QnA pair: Q: "What is the price of Microsoft Stock?" A: "$200".
If we receive user queries like "Microsoft stock value", "Microsoft share value", "Microsoft stock
worth", "Microsoft share worth", "stock value", etc., they should be able to get correct answer
even though these queries have words like share, value, worth which are not originally present
in the knowledge base.
Custom question answering takes casing into account but it's intelligent enough to understand
when it is to be ignored. You should not be seeing any perceivable difference due to wrong
casing.
When a KB has hierarchical relationships (either added manually or via extraction) and the
previous response was an answer related to other QnAs, for the next query we give slight
preference to all the children QnAs, sibling QnAs and grandchildren QnAs in that order. Along
with any query, the [custom question Answering API]
(/rest/api/cognitiveservices/questionanswering/question-answering/get-answers) expects a
"context" object with the property "previousQnAId" which denotes the last top answer. Based
on this previous QnA ID, all the related QnAs are boosted.
Accents are supported for all major European languages. If the query has an incorrect accent,
confidence score might be slightly different, but the service still returns the relevant answer
and takes care of minor errors by leveraging fuzzy search.
How are lowercase/uppercase characters treated?
How are QnAs prioritized for multi-turn questions?
How are accents treated?
How is punctuation in a user query treated?
\nPunctuation is ignored in user query before sending it to the ranking stack. Ideally it should not
impact the relevance scores. Punctuation that are ignored are as follows: ,?:;"'(){}[]-+。./!*؟
Next steps
\nTroubleshooting for custom question
answering
06/21/2025
The curated list of the most frequently asked questions regarding custom question answering
will help you adopt the feature faster and with better results.
How can I improve the throughput performance for query predictions?
Answer: Throughput performance issues indicate you need to scale up your Azure AI Search.
Consider adding a replica to your Azure AI Search to improve performance.
Learn more about pricing tiers.
Why is my URL(s)/file(s) not extracting question-answer pairs?
Answer: It's possible that custom question answering can't auto-extract some question-and-
answer (QnA) content from valid FAQ URLs. In such cases, you can paste the QnA content in a
.txt file and see if the tool can ingest it. Alternately, you can editorially add content to your
project through the Language Studio portal
.
How large a project can I create?
Answer: The size of the project depends on the SKU of Azure search you choose when creating
the QnA Maker service. Read here for more details.
How do I share a project with others?
Answer: Sharing works at the level of the language resource, that is, all projects associated a
language resource can be shared.
Can you share a project with a contributor that is not in the same Microsoft Entra tenant,
to modify a project?
Answer: Sharing is based on Azure role-based access control (Azure Role-base access control).
If you can share any resource in Azure with another user, you can also share custom question
answering.
Can you assign read/write rights to 5 different users so each of them can access only 1
custom question answering project?
Manage predictions
Manage your project
\nAnswer: You can share an entire language resource, not individual projects.
The updates that I made to my project are not reflected in production. Why not?
Answer: Every edit operation, whether in a table update, test, or setting, needs to be saved
before it can be deployed. Be sure to select Save after making changes and then re-deploy
your project for those changes to be reflected in production.
Does the project support rich data or multimedia?
Answer:
Multimedia auto-extraction for files and URLs
URLS - limited HTML-to-Markdown conversion capability.
Files - not supported
Answer text in markdown
Once QnA pairs are in the project, you can edit an answer's markdown text to include links to
media available from public URLs.
Does custom question answering support non-English languages?
Answer: See more details about supported languages.
If you have content from multiple languages, be sure to create a separate project for each
language.
I deleted my existing Search service. How can I fix this?
Answer: If you delete an Azure AI Search index, the operation is final and the index cannot be
recovered.
I deleted my `testkbv2` index in my Search service. How can I fix this?
Answer: In case you deleted the testkbv2  index in your Search service, you can restore the
data from the last published KB. Use the recovery tool RestoreTestKBIndex
 available on
GitHub.
Can I use the same Azure AI Search resource for projects using multiple languages?
Answer: To use multiple language and multiple projects, the user has to create a project for
each language and the first project created for the language resource has to select the option I
want to select the language when I create a project in this resource. This will create a
separate Azure search service per language.
Manage service
\nDo I need to use Bot Framework in order to use custom question answering?
Answer: No, you do not need to use the Bot Framework
 with custom question answering.
However, custom question answering is offered as one of several templates in Azure AI Bot
Service. Bot Service enables rapid intelligent bot development through Microsoft Bot
Framework, and it runs in a server-less environment.
How can I create a new bot with custom question answering?
Answer: Follow the instructions in this documentation to create your Bot with Azure AI Bot
Service.
How do I use a different project with an existing Azure AI Bot Service?
Answer: You need to have the following information about your project:
Project ID.
Project's published endpoint custom subdomain name, known as host , found on
Settings page after you publish.
Project's published endpoint key - found on Settings page after you publish.
With this information, go to your bot's app service in the Azure portal. Under Settings ->
Configuration -> Application settings, change those values.
The project's endpoint key is labeled QnAAuthkey  in the ABS service.
Can two or more client applications share a project?
Answer: Yes, the project can be queried from any number of clients.
How do I embed custom question answering in my website?
Answer: Follow these steps to embed the custom question answering service as a web-chat
control in your website:
1. Create your FAQ bot by following the instructions here.
2. Enable the web chat by following the steps here
Data storage
What data is stored and where is it stored?
Answer:
When you create your language resource for custom question answering, you selected an
Azure region. Your projects and log files are stored in this region.
Integrate with other services including Bots
\nWhat is custom question answering?
Article • 05/19/2025
Custom question answering provides cloud-based Natural Language Processing (NLP) that
allows you to create a natural conversational layer over your data. It is used to find appropriate
answers from customer input or from a project.
Custom question answering is commonly used to build conversational client applications,
which include social media applications, chat bots, and speech-enabled desktop applications.
This offering includes features like enhanced relevance using a deep learning ranker, precise
answers, and end-to-end region support.
Custom question answering comprises two capabilities:
Custom question answering: Using this capability users can customize different aspects
like edit question and answer pairs extracted from the content source, define synonyms
and metadata, accept question suggestions etc.
QnA Maker: This capability allows users to get a response by querying a text passage
without having the need to manage knowledge bases.
This documentation contains the following article types:
The quickstarts are step-by-step instructions that let you make calls to the service and get
results in a short period of time.
The how-to guides contain instructions for using the service in more specific or
customized ways.
The conceptual articles provide in-depth explanations of the service's functionality and
features.
Tutorials are longer guides that show you how to use the service as a component in
broader business solutions.
When you have static information - Use custom question answering when you have
static information in your project. This project is custom to your needs, which you've built
with documents such as PDFs and URLs.
When you want to provide the same answer to a request, question, or command -
when different users submit the same question, the same answer is returned.
When you want to filter static information based on meta-information - add metadata
tags to provide additional filtering options relevant to your client application's users and
the information. Common metadata information includes chit-chat, content type or
format, content purpose, and content freshness.
When to use custom question answering
\nWhen you want to manage a bot conversation that includes static information - your
project takes a user's conversational text or command and answers it. If the answer is part
of a pre-determined conversation flow, represented in your project with multi-turn
context, the bot can easily provide this flow.
When you want to use an agent to get an exact answer - Use the exact question
answering
 agent template answers high-value predefined questions deterministically to
ensure consistent and accurate responses or the intent routing
 agent template, which
detects user intent and provides exact answering. Perfect for deterministically intent
routing and exact question answering with human control.
Custom question answering imports your content into a project full of question and answer
pairs. The import process extracts information about the relationship between the parts of your
structured and semi-structured content to imply relationships between the question and
answer pairs. You can edit these question and answer pairs or add new pairs.
The content of the question and answer pair includes:
All the alternate forms of the question
Metadata tags used to filter answer choices during the search
Follow-up prompts to continue the search refinement
After you publish your project, a client application sends a user's question to your endpoint.
Your custom question answering service processes the question and responds with the best
answer.
Once a custom question answering project is published, a client application sends a question
to your project endpoint and receives the results as a JSON response. A common client
application for custom question answering is a chat bot.
What is a project?
Create a chat bot programmatically
\nStep
Action
1
The client application sends the user's question (text in their own words), "How do I
programmatically update my project?" to your project endpoint.
2
Custom question answering uses the trained project to provide the correct answer and any follow-
up prompts that can be used to refine the search for the best answer. Custom question answering
returns a JSON-formatted response.
3
The client application uses the JSON response to make decisions about how to continue the
conversation. These decisions can include showing the top answer and presenting more choices to
refine the search for the best answer.
The Language Studio
 portal provides the complete project authoring experience. You can
import documents, in their current form, to your project. These documents (such as an FAQ,
product manual, spreadsheet, or web page) are converted into question and answer pairs. Each
pair is analyzed for follow-up prompts and connected to other pairs. The final markdown
format supports rich presentation including images and links.
Once your project is edited, publish the project to a working Azure Web App bot
 without
writing any code. Test your bot in the Azure portal
 or download it and continue
development.
ﾉ
Expand table
Build low code chat bots
High quality responses with layered ranking
\n![Image](images/page1119_image1.png)
\nThe custom question answering system uses a layered ranking approach. The data is stored in
Azure search, which also serves as the first ranking layer. The top results from Azure search are
then passed through custom question answering's NLP re-ranking model to produce the final
results and confidence score.
Custom question answering provides multi-turn prompts and active learning to help you
improve your basic question and answer pairs.
Multi-turn prompts give you the opportunity to connect question and answer pairs. This
connection allows the client application to provide a top answer and provides more questions
to refine the search for a final answer.
After the project receives questions from users at the published endpoint, custom question
answering applies active learning to these real-world questions to suggest changes to your
project to improve the quality.
Custom question answering provides authoring, training, and publishing along with
collaboration permissions to integrate into the full development life cycle.
Multi-turn conversations
Development lifecycle
\n![Image](images/page1120_image1.png)