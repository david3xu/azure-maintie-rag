Custom question answering limits
The project query prediction request is controlled by the web app plan and web app. Refer to
recommended settings to plan your pricing tier.
Proper resource selection means your project answers query predictions successfully.
If your project isn't functioning properly, it's typically an issue of improper resource
management.
Improper resource selection requires investigation to determine which resource needs to
change.
A project is directly tied its language resource. It holds the question and answer (QnA) pairs
that are used to answer query prediction requests.
You can now have projects in different languages within the same language resource where the
custom question answering feature is enabled. When you create the first project, you can
choose whether you want to use the resource for projects in a single language that will apply
to all subsequent projects or make a language selection each time a project is created.
Custom question answering also supports unstructured content. You can upload a file that has
unstructured content.
Currently we do not support URLs for unstructured content.
The ingestion process converts supported content types to markdown. All further editing of the
answer is done with markdown. After you create a project, you can edit QnA pairs in Language
Studio with rich text authoring.
Because the final format of a QnA pair is markdown, it's important to understand markdown
support.
Understand the impact of resource selection
Project
Language considerations
Ingest data sources
Data format considerations
\nAdd a bot personality to your project with chit-chat. This personality comes through with
answers provided in a certain conversational tone such as professional and friendly. This chit-
chat is provided as a conversational set, which you have total control to add, edit, and remove.
A bot personality is recommended if your bot connects to your project. You can choose to use
chit-chat in your project even if you also connect to other services, but you should review how
the bot service interacts to know if that is the correct architectural design for your use.
Conversation flow usually begins with a salutation from a user, such as Hi  or Hello . Your
project can answer with a general answer, such as Hi, how can I help you , and it can also
provide a selection of follow-up prompts to continue the conversation.
You should design your conversational flow with a loop in mind so that a user knows how to
use your bot and isn't abandoned by the bot in the conversation. Follow-up prompts provide
linking between QnA pairs, which allow for the conversational flow.
Collaborators may be other developers who share the full development stack of the project
application or may be limited to just authoring the project.
project authoring supports several role-based access permissions you apply in the Azure portal
to limit the scope of a collaborator's abilities.
Integration with client applications is accomplished by sending a query to the prediction
runtime endpoint. A query is sent to your specific project with an SDK or REST-based request
to your custom question answering web app endpoint.
To authenticate a client request correctly, the client application must send the correct
credentials and project ID. If you're using an Azure AI Bot Service, configure these settings as
part of the bot configuration in the Azure portal.
Bot personality
Conversation flow with a project
Authoring with collaborators
Integration with client applications
Conversation flow in a client application
\nConversation flow in a client application, such as an Azure bot, may require functionality before
and after interacting with the project.
Does your client application support conversation flow, either by providing alternate means to
handle follow-up prompts or including chit-chit? If so, design these early and make sure the
client application query is handled correctly by another service or when sent to your project.
Custom question answering uses active learning to improve your project by suggesting
alternate questions to an answer. The client application is responsible for a part of this active
learning. Through conversational prompts, the client application can determine that the project
returned an answer that's not useful to the user, and it can determine a better answer. The
client application needs to send that information back to the project to improve the prediction
quality.
If your project doesn't find an answer, it returns the default answer. This answer is configurable
on the Settings page.
This default answer is different from the Azure bot default answer. You configure the default
answer for your Azure bot in the Azure portal as part of configuration settings. It's returned
when the score threshold isn't met.
The prediction is the response from your project, and it includes more information than just the
answer. To get a query prediction response, use the custom question answering API.
A score can change based on several factors:
Number of answers you requested in response with the top  property
Variety of available alternate questions
Filtering for metadata
Query sent to test  or production  project.
Active learning from a client application
Providing a default answer
Prediction
Prediction score fluctuations
Analytics with Azure Monitor
\nIn custom question answering, telemetry is offered through the Azure Monitor service. Use our
top queries to understand your metrics.
The development lifecycle of a project is ongoing: editing, testing, and publishing your project.
Your QnA pairs should be designed and developed based on your client application usage.
Each pair can contain:
Metadata - filterable when querying to allow you to tag your QnA pairs with additional
information about the source, content, format, and purpose of your data.
Follow-up prompts - helps to determine a path through your project so the user arrives at
the correct answer.
Alternate questions - important to allow search to match to your answer from different
forms of the question. Active learning suggestions turn into alternate questions.
Developing a project to insert into a DevOps pipeline requires that the project is isolated
during batch testing.
A project shares the Azure AI Search index with all other projects on the language resource.
While the project is isolated by partition, sharing the index can cause a difference in the score
when compared to the published project.
To have the same score on the test  and production  projects, isolate a language resource to a
single project. In this architecture, the resource only needs to live as long as the isolated batch
test.
Azure resources
Development lifecycle
Project development of question answer pairs
DevOps development
Next steps
\nPrecise answering
06/21/2025
The precise answering feature introduced, allows you to get the precise short answer from the
best candidate answer passage present in the project for any user query. This feature uses a
deep learning model at runtime, which understands the intent of the user query and detects
the precise short answer from the answer passage, if there is a short answer present as a fact in
the answer passage.
This feature is beneficial for both content developers as well as end users. Now, content
developers don't need to manually curate specific question answer pairs for every fact present
in the project, and the end user doesn't need to look through the whole answer passage
returned from the service to find the actual fact that answers the user's query.
In the Language Studio portal
, when you open the test pane, you will see an option to
Include short answer response on the top above show advanced options.
When you enter a query in the test pane, you will see a short-answer along with the answer
passage, if there is a short answer present in the answer passage.
Precise answering via the portal
\n![Image](images/page1136_image1.png)
\nYou can unselect the Include short answer response option, if you want to see only the Long
answer passage in the test pane.
The service also returns back the confidence score of the precise answer as an Answer-span
confidence score which you can check by selecting the Inspect option and then selection
additional information.

\n![Image](images/page1137_image1.png)
\nWhen you publish a bot, you get the precise answer enabled experience by default in your
application, where you will see short answer along with the answer passage. Refer to the API
reference for REST API to see how to use the precise answer (called AnswerSpan) in the
response. User has the flexibility to choose other experiences by updating the template
through the Bot app service.

Deploying a bot
\n![Image](images/page1138_image1.png)
\nConfidence score
06/04/2025
When a user query is matched against a project (also known as a knowledge base), Custom
question answering returns relevant answers, along with a confidence score. This score
indicates the confidence that the answer is the right match for the given user query.
The confidence score is a number between 0 and 100. A score of 100 is likely an exact match,
while a score of 0 means, that no matching answer was found. The higher the score- the
greater the confidence in the answer. For a given query, there could be multiple answers
returned. In that case, the answers are returned in order of decreasing confidence score.
The following table indicates typical confidence associated for a given score.
Score
Value
Score Meaning
Example
Query
0.90 -
1.00
A near exact match of user query and a KB question
> 0.70
High confidence - typically a good answer that completely answers the
user's query
0.50 -
0.70
Medium confidence - typically a fairly good answer that should answer the
main intent of the user query
0.30 -
0.50
Low confidence - typically a related answer, that partially answers the user's
intent
< 0.30
Very low confidence - typically does not answer the user's query, but has
some matching words or phrases
0
No match, so the answer is not returned.
The table above shows the range of scores that can occur when querying with Custom
question answering. However, since every project is different, and has different types of words,
intents, and goals- we recommend you test and choose the threshold that best works for you.
By default the threshold is set to 0 , so that all possible answers are returned. The
recommended threshold that should work for most projects, is 50.
ﾉ
Expand table
Choose a score threshold
\nWhen choosing your threshold, keep in mind the balance between Accuracy and Coverage,
and adjust your threshold based on your requirements.
If Accuracy (or precision) is more important for your scenario, then increase your
threshold. This way, every time you return an answer, it will be a much more CONFIDENT
case, and much more likely to be the answer users are looking for. In this case, you might
end up leaving more questions unanswered.
If Coverage (or recall) is more important- and you want to answer as many questions as
possible, even if there is only a partial relation to the user's question- then LOWER the
threshold. This means there could be more cases where the answer does not answer the
user's actual query, but gives some other somewhat related answer.
Set the threshold score as a property of the REST API JSON body. This means you set it for
each call to REST API.
To improve the confidence score of a particular response to a user query, you can add the user
query to the project as an alternate question on that response. You can also use case-
insensitive synonyms to add synonyms to keywords in your project.
When multiple responses have a similar confidence score, it is likely that the query was too
generic and therefore matched with equal likelihood with multiple answers. Try to structure
your QnAs better so that every QnA entity has a distinct intent.
The confidence score of an answer may change negligibly between the test and deployed
version of the project even if the content is the same. This is because the content of the test
and the deployed project are located in different Azure AI Search indexes.
The test index holds all the question and answer pairs of your project. When querying the test
index, the query applies to the entire index then results are restricted to the partition for that
Set threshold
Improve confidence scores
Similar confidence scores
Confidence score differences between test and
production