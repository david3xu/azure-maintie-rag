4. Review your Language resource and LUIS applications selections. Select Finish to
migrate your applications.
5. A popup window lets you track the migration status of your applications.
Applications that haven't started migrating have a status of Not started. Applications
that have begun migrating have a status of In progress, and once they have finished
migrating their status is Succeeded. A Failed application means that you must repeat
the migration process. Once the migration has completed for all applications, select
Done.
your new migrated applications.

\n![Image](images/page311_image1.png)
\n6. After your applications have migrated, you can perform the following steps:
Train your model
Deploy your model
Call your deployed model
CLU supports the model JSON version 7.0.0. If the JSON format is older, it would need to be
imported into LUIS first, then exported from LUIS with the most recent version.
In CLU, a single entity can have multiple entity components, which are different methods for
extraction. Those components are then combined together using rules you can define. The
available components are:

Frequently asked questions
Which LUIS JSON version is supported by CLU?
How are entities different in CLU?
\n![Image](images/page312_image1.png)
\nLearned: Equivalent to ML entities in LUIS, labels are used to train a machine-learned
model to predict an entity based on the content and context of the provided labels.
List: Just like list entities in LUIS, list components exact match a set of synonyms and maps
them back to a normalized value called a list key.
Prebuilt: Prebuilt components allow you to define an entity with the prebuilt extractors for
common types available in both LUIS and CLU.
Regex: Regex components use regular expressions to capture custom defined patterns,
exactly like regex entities in LUIS.
Entities in LUIS are transferred over as entities of the same name in CLU with the equivalent
components transferred.
After migrating, your structured machine-learned leaf nodes and bottom-level subentities are
transferred to the new CLU model while all the parent entities and higher-level entities are
ignored. The name of the entity is the bottom-level entity’s name concatenated with its parent
entity.
LUIS entity:
Pizza Order
Topping
Size
Migrated LUIS entity in CLU:
Pizza Order.Topping
Pizza Order.Size
You also can't label 2 different entities in CLU for the same span of characters. Learned
components in CLU are mutually exclusive and don't provide overlapping predictions for
learned components only. When migrating your LUIS application, entity labels that overlapped
preserved the longest label and ignored any others.
For more information on entity components, see Entity components.
Your roles are transferred as distinct entities along with their labeled utterances. Each role’s
entity type determines which entity component is populated. For example, a list entity role is
transferred as an entity with the same name as the role, with a populated list component.
Example:
How are entity roles transferred to CLU?
\nEntities used as features for intents aren't transferred. Entities used as features for other entities
populate the relevant component of the entity. For example, if a list entity named SizeList was
used as a feature to a machine-learned entity named Size, then the Size entity is transferred to
CLU with the list values from SizeList added to its list component. The same is applied for
prebuilt and regex entities.
Any extracted entity has a 100% confidence score and therefore entity confidence scores
shouldn't be used to make decisions between entities.
Conversational language understanding projects accept utterances in different languages.
Furthermore, you can train your model in one language and extend it to predict in other
languages.
Training utterance (English): How are you?
Labeled intent: Greeting
Runtime utterance (French): Comment ça va?
Predicted intent: Greeting
CLU uses state-of-the-art models to enhance machine learning performance of different
models of intent classification and entity extraction.
These models are insensitive to minor variations, removing the need for the following settings:
Normalize punctuation, normalize diacritics, normalize word form, and use all training data.
Additionally, the new models don't support phrase list features as they no longer require
supplementary information from the user to provide semantically similar words for better
accuracy. Patterns were also used to provide improved intent classification using rule-based
matching techniques that aren't necessary in the new model paradigm. The question below
explains this in more detail.
How do entity features get transferred in CLU?
How are entity confidence scores different in CLU?
How is conversational language understanding multilingual?
Example:
How is the accuracy of CLU better than LUIS?
\nThere are several features that were present in LUIS that are no longer available in CLU. This
includes the ability to do feature engineering, having patterns and pattern.any entities, and
structured entities. If you had dependencies on these features in LUIS, use the following
guidance:
Patterns: Patterns were added in LUIS to assist the intent classification through defining
regular expression template utterances. This included the ability to define Pattern only
intents (without utterance examples). CLU is capable of generalizing by using the state-of-
the-art models. You can provide a few utterances to that matched a specific pattern to the
intent in CLU, and it likely classifies the different patterns as the top intent without the
need of the pattern template utterance. This simplifies the requirement to formulate
these patterns, which was limited in LUIS, and provides a better intent classification
experience.
Phrase list features: The ability to associate features mainly occurred to assist the
classification of intents by highlighting the key elements/features to use. This is no longer
required since the deep models used in CLU already possess the ability to identify the
elements that are inherent in the language. In turn removing these features has no effect
on the classification ability of the model.
Structured entities: The ability to define structured entities was mainly to enable
multilevel parsing of utterances. With the different possibilities of the subentities, LUIS
needed all the different combinations of entities to be defined and presented to the
model as examples. In CLU, these structured entities are no longer supported, since
overlapping learned components aren't supported. There are a few possible approaches
to handling these structured extractions:
Non-ambiguous extractions: In most cases the detection of the leaf entities is enough
to understand the required items within a full span. For example, structured entity such
as Trip that fully spanned a source and destination (London to New York or Home to
work) can be identified with the individual spans predicted for source and destination.
Their presence as individual predictions would inform you of the Trip entity.
Ambiguous extractions: When the boundaries of different subentities aren't clear. To
illustrate, take the example "I want to order a pepperoni pizza and an extra cheese
vegetarian pizza". While the different pizza types and the topping modifications can be
extracted, having them extracted without context would have a degree of ambiguity of
where the extra cheese is added. In this case, the extent of the span is context based
and would require ML to determine this. For ambiguous extractions you can use one of
the following approaches:
What do I do if the features I'm using in LUIS are no longer
present?
\n1. Combine subentities into different entity components within the same entity.
LUIS Implementation:
Pizza Order (entity)
Size (subentity)
Quantity (subentity)
CLU Implementation:
Pizza Order (entity)
Size (list entity component: small, medium, large)
Quantity (prebuilt entity component: number)
In CLU, you would label the entire span for Pizza Order inclusive of the size and quantity, which
would return the pizza order with a list key for size, and a number value for quantity in the
same entity object.
2. For more complex problems where entities contain several levels of depth, you can create
a project for each level of depth in the entity structure. This gives you the option to:
Pass the utterance to each project.
Combine the analyses of each project in the stage proceeding CLU.
For a detailed example on this concept, check out the pizza sample projects available on
GitHub
.
CLU saves the data assets used to train your model. You can export a model's assets or load
them back into the project at any point. So models act as different versions of your project.
You can export your CLU projects using Language Studio
 or programmatically and store
different versions of the assets locally.
CLU presents a different approach to training models by using multi-classification as opposed
to binary classification. As a result, the interpretation of scores is different and also differs
across training options. While you're likely to achieve better results, you have to observe the
Example:
How do I manage versions in CLU?
Why is CLU classification different from LUIS? How does None
classification work?
\ndifference in scores and determine a new threshold for accepting intent predictions. You can
easily add a confidence score threshold for the None intent in your project settings. This
returns None as the top intent if the top intent didn't exceed the confidence score threshold
provided.
The new CLU models have better semantic understanding of language than in LUIS, and in turn
help make models generalize with a significant reduction of data. While you shouldn’t aim to
reduce the amount of data that you have, you should expect better performance and resilience
to variations and synonyms in CLU compared to LUIS.
Your existing LUIS applications are available until October 1, 2025. After that time you'll no
longer be able to use those applications, the service endpoints will no longer function, and the
applications will be permanently deleted.
Only JSON format is supported by CLU. You can import your .LU files to LUIS and export them
in JSON format, or you can follow the migration steps above for your application.
See the service limits article for more information.
The API objects of CLU applications are different from LUIS and therefore code refactoring is
necessary.
If you're using the LUIS programmatic
 and runtime
 APIs, you can replace them with their
equivalent APIs.
CLU authoring APIs
: Instead of LUIS's specific CRUD APIs for individual actions such as add
utterance, delete entity, and rename intent, CLU offers an import API that replaces the full
content of a project using the same name. If your service used LUIS programmatic APIs to
provide a platform for other customers, you must consider this new design paradigm. All other
APIs such as: listing projects, training, deploying, and deleting are available. APIs for actions such
Do I need more data for CLU models than LUIS?
If I don’t migrate my LUIS apps, are they deleted?
Are .LU files supported on CLU?
What are the service limits of CLU?
Do I have to refactor my code if I migrate my applications
from LUIS to CLU?
\nas importing and deploying are asynchronous operations instead of synchronous as they were
in LUIS.
CLU runtime APIs
: The new API request and response includes many of the same parameters
such as: query, prediction, top intent, intents, entities, and their values. The CLU response object
offers a more straightforward approach. Entity predictions are provided as they are within the
utterance text, and any additional information such as resolution or list keys are provided in
extra parameters called extraInformation  and resolution .
You can use the .NET
 or Python
 CLU runtime SDK to replace the LUIS runtime SDK. There's
currently no authoring SDK available for CLU.
CLU offers standard training, which trains and learns in English and is comparable to the
training time of LUIS. It also offers advanced training, which takes a considerably longer
duration as it extends the training to all other supported languages. The train API continues to
be an asynchronous process, and you need to assess the change in the DevOps process you
employ for your solution.
In LUIS you would Build-Train-Test-Publish, whereas in CLU you Build-Train-Evaluate-Deploy-
Test.
1. Build: In CLU, you can define your intents, entities, and utterances before you train. CLU
additionally offers you the ability to specify test data as you build your application to be
used for model evaluation. Evaluation assesses how well your model is performing on
your test data and provides you with precision, recall, and F1 metrics.
2. Train: You create a model with a name each time you train. You can overwrite an already
trained model. You can specify either standard or advanced training, and determine if you
would like to use your test data for evaluation, or a percentage of your training data to be
left out from training and used as testing data. After training is complete, you can
evaluate how well your model is doing on the outside.
3. Deploy: After training is complete and you have a model with a name, it can be deployed
for predictions. A deployment is also named and has an assigned model. You could have
multiple deployments for the same model. A deployment can be overwritten with a
different model, or you can swap models with other deployments in the project.
How are the training times different in CLU? How is standard
training different from advanced training?
How has the experience changed in CLU compared to LUIS?
How is the development lifecycle different?
\n4. Test: Once deployment is complete, you can use it for predictions through the
deployment endpoint. You can also test it in the studio in the Test deployment page.
This process is in contrast to LUIS, where the application ID was attached to everything, and
you deployed a version of the application in either the staging or production slots.
This influences the DevOps processes you use.
No, you can't export CLU to containers.
Any special characters in the LUIS application name are removed. If the cleared name length is
greater than 50 characters, the extra characters are removed. If the name after removing
special characters is empty (for example, if the LUIS application name was @@ ), the new name is
untitled. If there's already a conversational language understanding project with the same
name, the migrated LUIS application is appended with _1  for the first duplicate and increase
by 1 for each subsequent duplicate. In case the new name’s length is 50 characters and it needs
to be renamed, the last 1 or 2 characters are removed to be able to concatenate the number
and still be within the 50 characters limit.
If you have any questions that were unanswered in this article, consider leaving your questions
at our Microsoft Q&A thread
.
Quickstart: create a CLU project
CLU language support
CLU FAQ
Does CLU have container support?
How are my LUIS applications be named in CLU after
migration?
Migration from LUIS Q&A
Next steps
\nBest practices for conversational language
understanding
06/04/2025
Use the following guidelines to create the best possible projects in conversational language
understanding.
Schema is the definition of your intents and entities. There are different approaches you could
take when you define what you should create as an intent versus an entity. Ask yourself these
questions:
What actions or queries am I trying to capture from my user?
What pieces of information are relevant in each action?
You can typically think of actions and queries as intents, while the information required to fulfill
those queries are entities.
For example, assume that you want your customers to cancel subscriptions for various
products that you offer through your chatbot. You can create a cancel intent with various
examples like "Cancel the Contoso service" or "Stop charging me for the Fabrikam
subscription." The user's intent here is to cancel, and the Contoso service or Fabrikam
subscription are the subscriptions they want to cancel.
To proceed, you create an entity for subscriptions. Then you can model your entire project to
capture actions as intents and use entities to fill in those actions. This approach allows you to
cancel anything you define as an entity, such as other products. You can then have intents for
signing up, renewing, and upgrading that all make use of the subscriptions and other entities.
The preceding schema design makes it easy for you to extend existing capabilities (canceling,
upgrading, or signing up) to new targets by creating a new entity.
Another approach is to model the information as intents and the actions as entities. Let's take
the same example of allowing your customers to cancel subscriptions through your chatbot.
You can create an intent for each subscription available, such as Contoso, with utterances like
"Cancel Contoso," "Stop charging me for Contoso services," and "Cancel the Contoso
subscription." You then create an entity to capture the cancel action. You can define different
entities for each action or consolidate actions as one entity with a list component to
differentiate between actions with different keys.
Choose a consistent schema