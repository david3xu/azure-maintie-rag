Multilingual projects
06/30/2025
Conversational language understanding makes it easy for you to extend your project to several
languages at once. When you enable multiple languages in projects, you can add language-
specific utterances and synonyms to your project. You can get multilingual predictions for your
intents and entities.
When you enable multiple languages in a project, you can train the project primarily in one
language and immediately get predictions in other languages.
For example, you can train your project entirely with English utterances and query it in French,
German, Mandarin, Japanese, Korean, and others. Conversational language understanding
makes it easy for you to scale your projects to multiple languages by using multilingual
technology to train your models.
Whenever you identify that a particular language isn't performing as well as other languages,
you can add utterances for that language in your project. In the tag utterances page in
Language Studio, you can select the language of the utterance you're adding. When you
introduce examples for that language to the model, it's introduced to more of the syntax of
that language and learns to predict it better.
You aren't expected to add the same number of utterances for every language. You should
build most of your project in one language and only add a few utterances in languages that
you observe aren't performing well. If you create a project that's primarily in English and start
testing it in French, German, and Spanish, you might observe that German doesn't perform as
well as the other two languages. In that case, consider adding 5% of your original English
examples in German, train a new model, and test in German again. You should see better
results for German queries. The more utterances you add, the more likely the results are going
to get better.
When you add data in another language, you shouldn't expect it to negatively affect other
languages.
Multilingual intent and learned entity components
List and prebuilt components in multiple
languages
\nProjects with multiple languages enabled allow you to specify synonyms per language for every
list key. Depending on the language you query your project with, you only get matches for the
list component with synonyms of that language. When you query your project, you can specify
the language in the request body:
JSON
If you don't provide a language, it falls back to the default language of your project. For a list
of different language codes, see Language support.
Prebuilt components are similar, where you should expect to get predictions for prebuilt
components that are available in specific languages. The request's language again determines
which components are attempting to be predicted. For information on the language support of
each prebuilt component, see the Supported prebuilt entity components.
Tag utterances
Train a model
"query": "{query}"
"language": "{language code}"
Related content
\nEntity components
06/04/2025
In conversational language understanding, entities are relevant pieces of information that are
extracted from your utterances. An entity can be extracted by different methods. They can be
learned through context, matched from a list, or detected by a prebuilt recognized entity. Every
entity in your project is composed of one or more of these methods, which are defined as your
entity's components.
When an entity is defined by more than one component, their predictions can overlap. You can
determine the behavior of an entity prediction when its components overlap by using a fixed
set of options in the entity options.
An entity component determines a way that you can extract the entity. An entity can contain
one component, which determines the only method to be used to extract the entity. An entity
can also contain multiple components to expand the ways in which the entity is defined and
extracted.
The learned component uses the entity tags you label your utterances with to train a machine-
learned model. The model learns to predict where the entity is based on the context within the
utterance. Your labels provide examples of where the entity is expected to be present in an
utterance, based on the meaning of the words around it and as the words that were labeled.
This component is only defined if you add labels by tagging utterances for the entity. If you
don't tag any utterances with the entity, it doesn't have a learned component.
Component types
Learned component

List component
\n![Image](images/page333_image1.png)
\nThe list component represents a fixed, closed set of related words along with their synonyms.
The component performs an exact text match against the list of values you provide as
synonyms. Each synonym belongs to a list key, which can be used as the normalized, standard
value for the synonym that returns in the output if the list component is matched. List keys
aren't used for matching.
In multilingual projects, you can specify a different set of synonyms for each language. When
you use the prediction API, you can specify the language in the input request, which only
matches the synonyms associated to that language.
The prebuilt component allows you to select from a library of common types such as numbers,
datetimes, and names. When added, a prebuilt component is automatically detected. You can
have up to five prebuilt components per entity. For more information, see the list of supported
prebuilt components.
The regex component matches regular expressions to capture consistent patterns. When
added, any text that matches the regular expression is extracted. You can have multiple regular
expressions within the same entity, each with a different key identifier. A matched expression
returns the key as part of the prediction response.
In multilingual projects, you can specify a different expression for each language. When you
use the prediction API, you can specify the language in the input request, which only matches

Prebuilt component

Regex component
\n![Image](images/page334_image1.png)

![Image](images/page334_image2.png)
\nthe regular expression associated to that language.
When multiple components are defined for an entity, their predictions might overlap. When an
overlap occurs, each entity's final prediction is determined by one of the following options.
Combine components as one entity when they overlap by taking the union of all the
components.
Use this option to combine all components when they overlap. When components are
combined, you get all the extra information that's tied to a list or prebuilt component when
they're present.
Suppose you have an entity called Software that has a list component, which contains
"Proseware OS" as an entry. In your utterance data, you have "I want to buy Proseware OS 9"
with "Proseware OS 9" tagged as Software:
By using combined components, the entity returns with the full context as "Proseware OS 9"
along with the key from the list component:

Entity options
Combine components
Example

\n![Image](images/page335_image1.png)
\nSuppose you had the same utterance, but only "OS 9" was predicted by the learned
component:
With combined components, the entity still returns as "Proseware OS 9" with the key from the
list component:
Each overlapping component returns as a separate instance of the entity. Apply your own logic
after prediction with this option.
Suppose you have an entity called Software that has a list component, which contains
"Proseware Desktop" as an entry. In your utterance data, you have "I want to buy Proseware
Desktop Pro" with "Proseware Desktop Pro" tagged as Software:



Don't combine components
Example

\nWhen you don't combine components, the entity returns twice:
Sometimes an entity can be defined by multiple components but requires one or more of them
to be present. Every component can be set as required, which means the entity won't be
returned if that component wasn't present. For example, if you have an entity with a list
component and a required learned component, it's guaranteed that any returned entity
includes a learned component. If it doesn't, the entity isn't returned.
Required components are most frequently used with learned components because they can
restrict the other component types to a specific context, which is commonly associated to roles.
You can also require all components to make sure that every component is present for an
entity.
In Language Studio, every component in an entity has a toggle next to it that allows you to set
it as required.
Suppose you have an entity called Ticket Quantity that attempts to extract the number of
tickets you want to reserve for flights, for utterances such as "Book two tickets tomorrow to
Cairo."
Typically, you add a prebuilt component for Quantity.Number  that already extracts all numbers.
If your entity was only defined with the prebuilt component, it also extracts other numbers as
part of the Ticket Quantity entity, such as "Book two tickets tomorrow to Cairo at 3 PM."
To resolve this scenario, you label a learned component in your training data for all the
numbers that are meant to be Ticket Quantity. The entity now has two components: the
prebuilt component that knows all numbers, and the learned one that predicts where the ticket
quantity is in a sentence. If you require the learned component, you make sure that Ticket
Quantity only returns when the learned component predicts it in the right context. If you also

Required components
Example
\nrequire the prebuilt component, you can then guarantee that the returned Ticket Quantity
entity is both a number and in the correct position.
Components give you the flexibility to define your entity in more than one way. When you
combine components, you make sure that each component is represented and you reduce the
number of entities returned in your predictions.
A common practice is to extend a prebuilt component with a list of values that the prebuilt
might not support. For example, if you have an Organization entity, which has a
General.Organization  prebuilt component added to it, the entity might not predict all the
organizations specific to your domain. You can use a list component to extend the values of the
Organization entity and extend the prebuilt component with your own organizations.
Other times, you might be interested in extracting an entity through context, such as a Product
in a retail project. You label the learned component of the product to learn where a product is
based on its position within the sentence. You might also have a list of products that you
already know beforehand that you want to always extract. Combining both components in one
entity allows you to get both options for the entity.
When you don't combine components, you allow every component to act as an independent
entity extractor. One way of using this option is to separate the entities extracted from a list to
the ones extracted through the learned or prebuilt components to handle and treat them
differently.
Supported prebuilt components
Use components and options
７ Note
Previously during the public preview of the service, there were four available options:
Longest overlap, Exact overlap, Union overlap, and Return all separately. Longest
overlap and Exact overlap are deprecated and are only supported for projects that
previously had those options selected. Union overlap has been renamed to Combine
components, while Return all separately has been renamed to Do not combine
components.
Related content
\nEvaluation metrics for conversational
language understanding models
06/04/2025
Your dataset is split into two parts: a set for training and a set for testing. The training set is
used to train the model, while the testing set is used as a test for model after training to
calculate the model performance and evaluation. The testing set isn't introduced to the model
through the training process to make sure that the model is tested on new data.
Model evaluation is triggered automatically after training is completed successfully. The
evaluation process starts by using the trained model to predict user-defined intents and
entities for utterances in the test set. Then the process compares them with the provided tags
to establish a baseline of truth. The results are returned so that you can review the model's
performance. For evaluation, conversational language understanding uses the following
metrics:
Precision: Measures how precise or accurate your model is. It's the ratio between the
correctly identified positives (true positives) and all identified positives. The precision
metric reveals how many of the predicted classes are correctly labeled.
Precision = #True_Positive / (#True_Positive + #False_Positive)
Recall: Measures the model's ability to predict actual positive classes. It's the ratio
between the predicted true positives and what was tagged. The recall metric reveals how
many of the predicted classes are correct.
Recall = #True_Positive / (#True_Positive + #False_Negatives)
F1 score: The F1 score is a function of precision and recall. It's needed when you seek a
balance between precision and recall.
F1 Score = 2 * Precision * Recall / (Precision + Recall)
Precision, recall, and the F1 score are calculated for:
Each entity separately (entity-level evaluation).
Each intent separately (intent-level evaluation).
For the model collectively (model-level evaluation).
The definitions of precision, recall, and evaluation are the same for entity-level, intent-level,
and model-level evaluations. However, the counts for true positives, false positives, and false
negatives can differ. For example, consider the following text.
\nMake a response with "thank you very much."
Reply with saying "yes."
Check my email please.
Email to Cynthia that dinner last week was splendid.
Send an email to Mike.
The intents used are Reply , sendEmail , and readEmail . The entities are contactName  and
message .
The model could make the following predictions:
Utterance
Predicted
intent
Actual
intent
Predicted entity
Actual entity
Make a response with
"thank you very
much"
Reply
Reply
thank you very
much  as message
thank you very much  as
message
Reply with saying
"yes"
sendEmail
Reply
--
yes  as message
Check my email
please
readEmail
readEmail
--
--
Email to Cynthia that
dinner last week was
splendid
Reply
sendEmail
dinner last week
was splendid  as
message
cynthia  as contactName ,
dinner last week was
splendid  as message
Send an email to Mike
sendEmail
sendEmail
mike  as message
mike  as contactName
Key
Count
Explanation
True positive
1
Utterance 1 was correctly predicted as Reply .
False positive
1
Utterance 4 was mistakenly predicted as Reply .
False negative
1
Utterance 2 was mistakenly predicted as sendEmail .
Example
ﾉ
Expand table
Intent-level evaluation for Reply intent
ﾉ
Expand table