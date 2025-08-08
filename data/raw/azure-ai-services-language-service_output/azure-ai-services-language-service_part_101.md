Croatian
Czech
Danish
Dutch
English
Estonian
Finnish
French
Galician
German
Greek
Gujarati
Hebrew
Hindi
Hungarian
Icelandic
Indonesian
Irish
Italian
Japanese
Kannada
Korean
Latvian
Lithuanian
Malayalam
Malay
Norwegian
Polish
Portuguese
Punjabi
Romanian
Russian
Serbian_Cyrillic
Serbian_Latin
Slovak
Slovenian
Spanish
Swedish
Tamil
Telugu
Thai
\nTurkish
Ukrainian
Urdu
Vietnamese
Custom question answering depends on Azure AI Search language analyzers for providing
results.
While the Azure AI Search capabilities are on par for supported languages, custom question
answering has an additional ranker that sits above the Azure search results. In this ranker
model, we use some special semantic and word-based features in the following languages.
Chinese
Czech
Dutch
English
French
German
Hungarian
Italian
Japanese
Korean
Polish
Portuguese
Spanish
Swedish
This ranking is an internal working of the custom question answering's ranker.
Question answering quickstart
Query matching and relevance
Next steps
\nTransparency note and use cases for
question answering
06/24/2025
An AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Creating a system that
is fit for its intended purpose requires an understanding of how the technology works, its
capabilities and limitations, and how to achieve the best performance.
Microsoft provides transparency notes to help you understand how our AI technology works.
This includes the choices system owners can make that influence system performance and
behavior, and the importance of thinking about the whole system, including the technology,
the people, and the environment. You can use transparency notes when developing or
deploying your own system, or share them with the people who will use or be affected by your
system.
Transparency notes are part of a broader effort at Microsoft to put our AI principles into
practice. To find out more, see Microsoft's AI principles
.
Question answering is a cloud-based, natural language processing service that easily creates a
natural conversational layer over your data. It can be used to find the most appropriate answer
for a specified natural language input, from your custom knowledge base of information. See
the list of supported languages here.
Question answering is commonly used to build conversational client applications, which
include social media applications, chat bots, and speech-enabled desktop applications. A client
application based on question answering can be any conversational application that
communicates with a user in natural language to answer a question.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a transparency note?
Introduction to question answering
\nQuestion Answering uses several Azure resources, each for a different purpose: Azure Cognitive
Search, and Azure Monitor. All customer data (question answers and chatlogs) is stored in the
region where the customer deploys the dependent service instances. For more details on
dependent services see here.
The first step in using question answering is training and preparing the QnA service to
recognize the questions and answers that may be developed from your content. Question
answering imports your content into a knowledge base of question and answer pairs. The
import process extracts information about the relationship between the parts of your
structured and semi-structured content to infer relationships between the question and answer
pairs.
The extracted QnA pairs are displayed in the following way:
You can edit these question and answer pairs, and add new pairs yourself. When you're
satisfied with the content of your knowledge base, you can publish it, which will make it ready
to be used to respond to questions sent to your client applications. At the second step, your
client application sends the user's question to your question answering service API. Your
question answering service processes the question and responds with the best answer.
The basics of question answering
\n![Image](images/page1004_image1.png)

![Image](images/page1004_image2.png)
\nFor more details, see the question answering documentation.
Term
Definition
Knowledge
base
A collection of questions, answers, and metadata that have been extracted from content
sources or added manually. The collection is then used to develop question and answer
pairs. Queries to the QnA service are matched against the contents of the knowledge base.
Active
learning
Consumes the feedback from use of the system to provide suggestions (in the form of
new questions) to the knowledge base owner to improve the contents of their knowledge
base. Learn more here.
Multi-turn
Sometimes additional information is needed for question answering to determine the best
answer to a user question. Question answering asks a follow-up question to the user.
Metadata
Additional information in the form of a name and value that you can associate with each
QnA pair in your knowledge base. Metadata can be used to pass context and filter results.
Synonyms
Alternate terms that can be used interchangeably in the knowledge base.
You can use question answering in multiple scenarios and across a variety of industries.
Typically information retrieval use cases are best suited for question answering where there are
usually one or only a few correct responses to a user question. Scenarios or topics that have a
wide variety of viewpoints, worldviews, geopolitical views, controversial content, etc. will be
more difficult to answer correctly. Customers should be aware that providing this type of
content via question answering can create negative sentiment and reactions, and result in
negative publicity. If you do provide this type of content, consider adding source attribution to
allow your users to evaluate the answers for themselves.
Some typical scenarios where question answering is recommended are:
Customer support: In most customer support scenarios, common questions get asked
frequently. Question Answering lets you instantly create a chat bot from existing support
content, and this bot can act as the front line system for handling customer queries. If the
questions can't be answered by the bot, then additional components can help identify
and flag the question for human intervention.
Enterprise FAQ bot: Information retrieval is a challenge for enterprise employees. Internal
FAQ bots are a great tool for helping employees get answers to their common questions.
Terms and definitions
ﾉ
Expand table
Example use cases
\nQuestion answering enables various departments, such as human resources or payroll, to
build FAQ chat bots to help employees.
Instant answers over search: Many search systems augment their search results with
instant answers, which provide the user with immediate access to information relevant to
their query. Answers from question answering can be combined with the results from
document search to offer an instant answer experience to the end user.
Avoid high-risk scenarios: The machine learnt algorithm used by question answering
optimizes the performance based on the data it is trained on, however there will always
be edge cases where the correct answer isn't returned for a user query which the system
doesn't understand well. When you design your scenarios with question answering, be
aware of the possibility of false positive results. It is advisable to create a dataset of the
top queries asked in your scenario and the corresponding expected answers, and
periodically test the service for the correctness of the responses. For example:
Healthcare: This often requires high precision, and wrong information can have life-
threatening consequences. Consider the example of a Doctor Assistant bot that uses
question answering to understand the patient's symptoms and match it to common
illnesses. Likewise, any bots that are designed to converse with patients with mental
health issues, such as depression or anxiety, must be very careful of the responses
returned. question answering can be helpful in parsing through clinical terminology
and deriving useful question and answer pairs, but is not designed, intended or made
available to create medical devices, and is not designed or intended and should not be
used as a substitute for professional medical advice, diagnosis, treatment, or judgment.
Customer is solely responsible for displaying and/or obtaining appropriate consents,
warnings, disclaimers and acknowledgements to end users of their implementation.
Avoid open domain scenarios: question answering is meant to answer questions from a
particular domain knowledge base, not open-ended questions, or out-of-domain
questions. Using out-of-domain questions with question answering runs the risk of
returning incorrect responses. For example:
Social bots: Bots that are meant for generic chit-chat, not related to a particular
domain, are difficult to design with question answering. In these scenarios, the user
intents and viewpoints can range widely (for example, sports, fashion, politics, and
religion). Building a question answering knowledge base is best used for facts and/or
discovery of content. Using question answering for diverse worldview topics may be
challenging and we recommend customers consider more careful review or curating of
such content.
Considerations when choosing other use cases
\nHandling inappropriate conversations: It's possible that users will initiate
inappropriate conversations with the bot, including expletives or hate speech. The bot
designer must be very careful about how to handle these conversations, and make
sure that these intents are detected with high accuracy and the appropriate response
given. It's difficult to build a comprehensive knowledge base in question answering
containing every variation of inappropriate utterances possible. It is therefore better to
handle such cases with a rule based system, for example the user utterances can be
quickly checked for the presence of any words from a pre-processed blocklist of
inappropriate keywords. This is not part of the question answering service and would
need to be developed on top of the question answering service.
Legal and regulatory considerations: Organizations need to evaluate potential specific
legal and regulatory obligations when using any AI services and solutions, which may not
be appropriate for use in every industry or scenario. Additionally, AI services or solutions
are not designed for and may not be used in ways prohibited in applicable terms of
service and relevant codes of conduct.
Microsoft AI principles
Microsoft responsible AI resources
Microsoft principles for developing and deploying facial recognition technology
Identify principles and practices for responsible AI
Building responsible bots
Next steps
\nGuidance for integration and responsible
use of question answering
06/24/2025
Microsoft wants to help you responsibly develop and deploy solutions that use question
answering. We're taking a principled approach to upholding personal agency and dignity by
considering the AI systems' fairness, reliability and safety, privacy and security, inclusiveness,
transparency, and human accountability. These considerations reflect our commitment to
developing Responsible AI.
When you're getting ready to deploy AI-powered products or features, the following activities
help to set you up for success:
Understand what it can do: Fully assess the capabilities of any AI model you are using to
understand its capabilities and limitations. Understand how it will perform in your
particular scenario and context, by thoroughly testing it with real-life conditions and data.
Synthetic data and tests that don't reflect your end-to-end scenario won't be sufficient.
Respect an individual's right to privacy: Only collect data and information from
individuals for lawful and justifiable purposes. Only use data and information that you
have consent to use for this purpose.
Legal review: Obtain appropriate legal advice to review your solution, particularly if you
will use it in sensitive or high-risk applications. Understand what restrictions you might
need to work within, and your responsibility to resolve any issues that might come up in
the future. Do not provide any legal advice or guidance.
System review: If you're planning to deploy an AI-powered product or feature into an
existing system of software, customers, and organizational processes, take the time to
understand how each part of your system will be affected. Consider how your AI solution
aligns with Microsoft's AI principles
.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
General deployment principles
\nHuman in the loop: Keep a human in the loop. This means ensuring constant human
oversight of the AI-powered product or feature, and maintaining the role of humans in
decision making. Ensure that you can have real-time human intervention in the solution
to prevent harm. This enables you to manage situations when the AI model does not
perform as required.
Security: Ensure your solution is secure, and that it has adequate controls to preserve the
integrity of your content and prevent any unauthorized access.
Customer feedback loop: Provide a feedback channel that allows users and individuals to
report issues with the service after it has been deployed. Monitor and improve the AI-
powered product or feature on an ongoing basis.
Common use cases of question answering include customer support chat bots and internal
enterprise FAQ chat bots. When you're deploying an application that uses question answering,
you should ask the following questions:
How is the data processed? All the customer data is stored in Azure Cognitive Search and
Azure Monitor in the customer's Azure subscription. Question answering processes the
data when extracting questions and answers from sources and serving the correct
answers for a particular query. The question answering service doesn't retain customer
data after responding to a client application's query.
Where is the data stored? Some countries or domains might have restrictions on the data
being stored in a particular geographic area. Choose the appropriate regions for Azure
Cognitive Search, and Azure Monitor, keeping in mind the data residency requirements of
your scenario.
How is user privacy handled? In some scenarios, users can be asked for additional
information before the response is returned from question answering. These scenarios are
called multi-turn conversations. Some of the information collected from the users can
include personal information or other sensitive information. For more information about
best practices for data privacy, see the Responsible bots guidelines
 for developers.
Inform users up front about the data that is collected and how it is used, and obtain
their consent beforehand. Provide easy access to a valid privacy statement and
applicable service agreement, and include a "profile page" for users to obtain
information about the bot, with links to relevant privacy and legal information.
Specific deployment guidance for question
answering
\nCollect no more personal data than you need, limit access to it, and store it for no
longer than needed. Collect only the personal data that is essential for your bot to
operate effectively. If your bot will share data (such as with another bot), be sure only
to share the minimum amount of user data necessary in order to complete the
requested function on behalf of the user. If you enable access by other agents to your
bot's user data, do so only for the time necessary in order to complete the requested
function. Always give users the opportunity to choose which agents your bot will share
data with, and what data is suitable for sharing. Consider whether you can purge
stored user data from time to time, while still enabling your bot to learn. Shorter
retention periods minimize security risks for users.
Provide privacy-protecting user controls. For bots that store personal information,
such as authenticated bots, consider providing easily discoverable buttons to protect
privacy. For example: Show me all you know about me, Forget my last interaction
and Delete all you know about me. In some cases, such controls might be legally
required.
Obtain legal and privacy review. The privacy aspects of bot design are subject to
important and increasingly stringent legal requirements. Be sure to obtain both a legal
and a privacy review of your bot's privacy practices through the appropriate channels
in your organization.
Microsoft AI principles
Microsoft responsible AI resources
Microsoft principles for developing and deploying facial recognition technology
Identify principles and practices for responsible AI
Building responsible bots
Next steps