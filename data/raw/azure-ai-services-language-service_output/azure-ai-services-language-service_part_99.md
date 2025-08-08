en , es , fr , de , it , pt-pt , pt-br , zh , ja , ko , nl , sv , tr , hi , da , nl , no , ro , ar , bg , hr ,
ms , ru , sl , cs , et , fi , he , hu , lv , sk , th , uk
Azure SAS
Connection string for Azure software as a service (SaaS).
To get this entity type, add AzureSAS  to the piiCategories  parameter. AzureSAS  is
returned in the API response if detected.
en , es , fr , de , it , pt-pt , pt-br , zh , ja , ko , nl , sv , tr , hi , da , nl , no , ro , ar , bg , hr ,
ms , ru , sl , cs , et , fi , he , hu , lv , sk , th , uk
Azure Service Bus Connection String
Connection string for an Azure service bus.
To get this entity type, add AzureServiceBusString  to the piiCategories  parameter.
AzureServiceBusString  is returned in the API response if detected.
en , es , fr , de , it , pt-pt , pt-br , zh , ja , ko , nl , sv , tr , hi , da , nl , no , ro , ar , bg , hr ,
ms , ru , sl , cs , et , fi , he , hu , lv , sk , th , uk
Azure Storage Account Key
Account key for an Azure storage account.
To get this entity type, add AzureStorageAccountKey  to the piiCategories  parameter.
AzureStorageAccountKey  is returned in the API response if detected.
en , es , fr , de , it , pt-pt , pt-br , zh , ja , ko , nl , sv , tr , hi , da , nl , no , ro , ar , bg , hr ,
ms , ru , sl , cs , et , fi , he , hu , lv , sk , th , uk
Azure Storage Account Key (Generic)
Generic account key for an Azure storage account.
To get this entity type, add AzureStorageAccountGeneric  to the piiCategories  parameter.
AzureStorageAccountGeneric  is returned in the API response if detected.
en , es , fr , de , it , pt-pt , pt-br , zh , ja , ko , nl , sv , tr , hi , da , nl , no , ro , ar , bg , hr ,
ms , ru , sl , cs , et , fi , he , hu , lv , sk , th , uk
\nSQL Server Connection String
Connection string for a computer running SQL Server.
To get this entity type, add SQLServerConnectionString  to the piiCategories  parameter.
SQLServerConnectionString  is returned in the API response if detected.
en
PII overview
Next steps
\nSupported customer content (PII) entity
categories in conversations
06/21/2025
Use this article to find the entity categories that can be returned by the conversational PII
detection feature. This feature runs a predictive model to identify, categorize, and redact
sensitive information from an input conversation.
The following entity categories are returned when you're sending API requests PII feature.
This category contains the following entity:
Entity
Name
Details
All first, middle, last or full name is considered PII regardless of whether it is the speaker’s
name, the agent’s name, someone else’s name or a different version of the speaker’s full name
(Chris vs. Christopher).
To get this entity category, add Person  to the pii-categories  parameter. Person  will be
returned in the API response if detected.
Supported document languages
en
This category contains the following entity:
Entity categories
Category: Person
７ Note
As of the 2023-04-15-preview API onwards, this category is 'Person' instead of 'Name'.
Category: Phone
\nEntity
Phone
Details
All telephone numbers (including toll-free numbers or numbers that may be easily found or
considered public knowledge) are considered PII
To get this entity category, add Phone  to the pii-categories  parameter. Phone  will be returned
in the API response if detected.
Supported document languages
en
This category contains the following entity:
Entity
Address
Details
Complete or partial addresses are considered PII. All addresses regardless of what residence or
institution the address belongs to (such as: personal residence, business, medical center,
government agency, etc.) are covered under this category.
Note:
If information is limited to City & State only, it will not be considered PII.
If information contains street, zip code or house number, all information is considered as
Address PII , including the city and state
To get this entity category, add Address  to the pii-categories  parameter. Address  will be
returned in the API response if detected.
Supported document languages
en
Category: Address
Category: Email
\nThis category contains the following entity:
Entity
Email
Details
All email addresses are considered PII.
To get this entity category, add Email  to the pii-categories  parameter. Email  will be returned
in the API response if detected.
Supported document languages
en
This category contains the following entities:
Entity
NumericIdentifier
Details
Any numeric or alphanumeric identifier that could contain any PII information. Examples:
Case Number
Member Number
Ticket number
Bank account number
Installation ID
IP Addresses
Product Keys
Serial Numbers (1:1 relationship with a specific item/product)
Shipping tracking numbers, etc.
To get this entity category, add NumericIdentifier  to the pii-categories  parameter.
NumericIdentifier  will be returned in the API response if detected.
Supported document languages
en
Category: NumericIdentifier
\nThis category contains the following entity:
Entity
Credit card
Details
Any credit card number, any security code on the back, or the expiration date is considered as
PII.
To get this entity category, add CreditCard  to the pii-categories  parameter. CreditCard  will
be returned in the API response if detected.
Supported document languages
en
How to detect PII in conversations
Category: Credit card
Next steps
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
\n![Image](images/page989_image1.png)
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
\n![Image](images/page990_image1.png)