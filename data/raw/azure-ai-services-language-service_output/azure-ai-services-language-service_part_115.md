specific project. If the test query results are negatively impacting your ability to validate the
project, you can:
Organize your project using one of the following:
One resource restricted to one project: restrict your single language resource (and the
resulting Azure AI Search test index) to a project.
Two resources - one for test, one for production: have two language resources, using
one for testing (with its own test and production indexes) and one for production (also
having its own test and production indexes)
Always use the same parameters when querying both your test and production projects.
When you deploy a project, the question and answer contents of your project moves from the
test index to a production index in Azure search.
If you have a project in different regions, each region uses its own Azure AI Search index.
Because different indexes are used, the scores will not be exactly the same.
When no good match is found by the ranker, the confidence score of 0.0 or "None" is returned
and the default response is returned. You can change the default response.
No match found
Next steps
\nCustom question answering best practices
06/21/2025
Use these best practices to improve your project and provide better results to your client
application or chat bot's end users.
Custom question answering is continually improving the algorithms that extract question
answer pairs from content and expanding the list of supported file and HTML formats. In
general, FAQ pages should be stand-alone and not combined with other information. Product
manuals should have clear headings and preferably an index page.
We’ve used the following list of question and answer pairs as representation of a project to
highlight best practices when authoring projects for custom question answering.
Question
Answer
I want to buy a car
There are three options to buy a car.
I want to purchase software license
Software license can be purchased online at no cost.
What is the price of Microsoft stock?
$200.
How to buy Microsoft Services
Microsoft services can be bought online.
Want to sell car
Please send car pics and document.
How to get access to identification card?
Apply via company portal to get identification card.
Custom question answering employs a transformer-based ranker that takes care of user
queries that are semantically similar to the question in the project. For example, consider the
following question answer pair:
Question: What is the price of Microsoft Stock? Answer: $200.
Extraction
Creating good questions and answers
ﾉ
Expand table
When should you add alternate questions to question and
answer pairs?
\nThe service can return the expected response for semantically similar queries such as:
“How much is Microsoft stock worth? “How much is Microsoft share value?” “How much does a
Microsoft share cost?” “What is the market value of a Microsoft stock?” “What is the market
value of a Microsoft share?”
However, it’s important to understand that the confidence score with which the system returns
the correct response will vary based on the input query and how different it is from the original
question answer pair.
There are certain scenarios that require the customer to add an alternate question. When it’s
already verified that for a particular query the correct answer isn’t returned despite being
present in the project, we advise adding that query as an alternate question to the intended
question answer pair.
Users can add as many alternate questions as they want, but only first 5 will be considered for
core ranking. However, the rest will be useful for exact match scenarios. It is also recommended
to keep the different intent/distinct alternate questions at the top for better relevance and
score.
Semantic understanding in custom question answering should be able to take care of similar
alternate questions.
The return on investment will start diminishing once you exceed 10 questions. Even if you’re
adding more than 10 alternate questions, try to make the initial 10 questions as semantically
dissimilar as possible so that all kinds of intents for the answer are captured by these 10
questions. For the project at the beginning of this section, in question answer pair #1, adding
alternate questions such as “How can I buy a car”, “I wanna buy a car” aren’t required. Whereas
adding alternate questions such as “How to purchase a car”, “What are the options of buying a
vehicle” can be useful.
Custom question answering provides the flexibility to use synonyms at the project level, unlike
QnA Maker where synonyms are shared across projects for the entire service.
For better relevance, you need to provide a list of acronyms that the end user intends to use
interchangeably. The following is a list of acceptable acronyms:
MSFT  – Microsoft
How many alternate questions per question answer pair is
optimal?
When to add synonyms to a project?
\nID  – Identification
ETA  – Estimated time of Arrival
Other than acronyms, if you think your words are similar in context of a particular domain and
generic language models won’t consider them similar, it’s better to add them as synonyms. For
instance, if an auto company producing a car model X receives queries such as “my car’s audio
isn’t working” and the project has questions on “fixing audio for car X”, then we need to add ‘X’
and ‘car’ as synonyms.
The transformer-based model already takes care of most of the common synonym cases, for
example: Purchase – Buy , Sell - Auction , Price – Value . For another example, consider the
following question answer pair: Q: “What is the price of Microsoft Stock?” A: “$200”.
If we receive user queries like “Microsoft stock value”,” Microsoft share value”, “Microsoft stock
worth”, “Microsoft share worth”, “stock value”, etc., you should be able to get the correct
answer even though these queries have words like "share", "value", and "worth", which aren’t
originally present in the project.
Special characters are not allowed in synonyms.
Question answering takes casing into account but it's intelligent enough to understand when
it’s to be ignored. You shouldn’t be seeing any perceivable difference due to wrong casing.
When a project has hierarchical relationships (either added manually or via extraction) and the
previous response was an answer related to other question answer pairs, for the next query we
give slight preference to all the children question answer pairs, sibling question answer pairs,
and grandchildren question answer pairs in that order. Along with any query, the custom
question answering REST API expects a context  object with the property previousQnAId , which
denotes the last top answer. Based on this previous QnAID , all the related QnAs  are boosted.
Accents are supported for all major European languages. If the query has an incorrect accent,
the confidence score might be slightly different, but the service still returns the relevant answer
and takes care of minor errors by leveraging fuzzy search.
How are lowercase/uppercase characters treated?
How are question answer pairs prioritized for multi-turn
questions?
How are accents treated?
\nPunctuation is ignored in a user query before sending it to the ranking stack. Ideally it
shouldn’t impact the relevance scores. Punctuation that is ignored is as follows: ,?:;\"'(){}[]-
+。./!*؟
Add chit-chat to your bot, to make your bot more conversational and engaging, with low
effort. You can easily add chit-chat data sources from pre-defined personalities when creating
your project, and change them at any time. Learn how to add chit-chat to your KB.
Chit-chat is supported in many languages.
Chit-chat is supported for several predefined personalities:
Personality
Custom question answering dataset file
Professional
qna_chitchat_professional.tsv
Friendly
qna_chitchat_friendly.tsv
Witty
qna_chitchat_witty.tsv
Caring
qna_chitchat_caring.tsv
Enthusiastic
qna_chitchat_enthusiastic.tsv
The responses range from formal to informal and irreverent. You should select the personality
that is closest aligned with the tone you want for your bot. You can view the datasets
, and
choose one that serves as a base for your bot, and then customize the responses.
There are some bot-specific questions that are part of the chit-chat data set, and have been
filled in with generic answers. Change these answers to best reflect your bot details.
We recommend making the following chit-chat question answer pairs more specific:
Who are you?
How is punctuation in a user query treated?
Chit-Chat
Choosing a personality
ﾉ
Expand table
Edit bot-specific questions
\nWhat can you do?
How old are you?
Who created you?
Hello
If you add your own chit-chat question answer pairs, make sure to add metadata so these
answers are returned. The metadata name/value pair is editorial:chitchat .
The custom question answering REST API uses both questions and the answer to search for
best answers to a user's query.
Use the RankerType=QuestionOnly if you don't want to search answers.
An example of this is when the project is a catalog of acronyms as questions with their full form
as the answer. The value of the answer won’t help to search for the appropriate answer.
Make sure you’re making the best use of the supported ranking features. Doing so will improve
the likelihood that a given user query is answered with an appropriate response.
The default confidence score that is used as a threshold is 0, however you can change the
threshold for your project based on your needs. Since every project is different, you should test
and choose the threshold that is best suited for your project.
By default, custom question answering searches through questions and answers. If you want to
search through questions only, to generate an answer, use the RankerType=QuestionOnly  in the
POST body of the REST API request.
Adding custom chit-chat with a metadata tag
Searching for answers
Searching questions only when answer isn’t relevant
Ranking/Scoring
Choosing a threshold
Choosing Ranker type
\nAlternate questions to improve the likelihood of a match with a user query. Alternate questions
are useful when there are multiple ways in which the same question may be asked. This can
include changes in the sentence structure and word-style.
Original query
Alternate queries
Change
Is parking available?
Do you have a car park?
sentence structure
Hi
Yo
Hey there
word-style or slang
Metadata adds the ability for a client application to know it shouldn’t take all answers but
instead to narrow down the results of a user query based on metadata tags. The project answer
can differ based on the metadata tag, even if the query is the same. For example, "where is
parking located" can have a different answer if the location of the restaurant branch is different
- that is, the metadata is Location: Seattle versus Location: Redmond.
While there’s some support for synonyms in the English language, use case-insensitive word
alterations to add synonyms to keywords that take different forms.
Original word
Synonyms
buy
purchase
net-banking
net banking
The ranking algorithm, which matches a user query with a question in the project, works best if
each question addresses a different need. Repetition of the same word set between questions
reduces the likelihood that the right answer is chosen for a given user query with those words.
Add alternate questions
ﾉ
Expand table
Use metadata tags to filter questions and answers
Use synonyms
ﾉ
Expand table
Use distinct words to differentiate questions
\nFor example, you might have two separate question answer pairs with the following questions:
Questions
where is the parking location
where is the ATM location
Since these two questions are phrased with very similar words, this similarity could cause very
similar scores for many user queries that are phrased like "where is the <x>  location". Instead,
try to clearly differentiate with queries like "where is the parking lot" and "where is the ATM", by
avoiding words like "location" that could be in many questions in your project.
Custom question answering allows users to collaborate on a project. Users need access to the
associated Azure resource group in order to access the projects. Some organizations may want
to outsource the project editing and maintenance, and still be able to protect access to their
Azure resources. This editor-approver model is done by setting up two identical language
resources with identical custom question answering projects in different subscriptions and
selecting one for the edit-testing cycle. Once testing is finished, the project contents are
exported and transferred with an import-export process to the language resource of the
approver that will finally deploy the project and update the endpoint.
Active learning does the best job of suggesting alternative questions when it has a wide range
of quality and quantity of user-based queries. It’s important to allow client-applications' user
queries to participate in the active learning feedback loop without censorship. Once questions
are suggested in Language Studio, you can review and accept or reject those suggestions.
ﾉ
Expand table
Collaborate
Active learning
Next steps
\nProject limits and boundaries
06/21/2025
Custom question answering limits provided below are a combination of the Azure AI Search
pricing tier limits and custom question answering limits. Both sets of limits affect how many
projects you can create per resource and how large each project can grow.
The maximum number of projects is based on Azure AI Search tier limits.
Choose the appropriate Azure search SKU
 for your scenario. Typically, you decide the
number of projects you need based on number of different subject domains. One subject
domain (for a single language) should be in one project.
With custom question answering, you have a choice to set up your language resource in a
single language or multiple languages. You can make this selection when you create your first
project in the Language Studio
.
For example, if your tier has 15 allowed indexes, you can publish 14 projects of the same
language (one index per published project). The 15th index is used for all the projects for
authoring and testing. If you choose to have projects in different languages, then you can only
publish seven projects.
File names may not include the following characters:
Projects
） Important
You can publish N-1 projects of a single language or N/2 projects of different languages in
a particular tier, where N is the maximum indexes allowed in the tier. Also check the
maximum size and the number of documents allowed per tier.
Extraction limits
File naming constraints
ﾉ
Expand table
\nDo not use character
Single quote '
Double quote "
Format
Max file size (MB)
.docx
10
.pdf
25
.tsv
10
.txt
10
.xlsx
3
The maximum number of deep-links that can be crawled for extraction of question answer
pairs from a URL page is 20.
Metadata is presented as a text-based key:value  pair, such as product:windows 10 . It is stored
and compared in lower case. Maximum number of metadata fields is based on your Azure AI
Search tier limits.
Maximum file size
ﾉ
Expand table
Maximum number of files
７ Note
Custom question answering currently has no limits on the number of sources that can be
added. Throughput is currently capped at 10 text records per second for both
management APIs and prediction APIs. When using the F0 tier, upload is limited to 3 files.
Maximum number of deep-links from URL
Metadata limits