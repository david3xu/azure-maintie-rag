Precision = #True_Positive / (#True_Positive + #False_Positive) = 1 / (1 + 1) = 0.5
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 1 / (1 + 1) = 0.5
F1 score = 2 * Precision * Recall / (Precision + Recall) = (2 * 0.5 * 0.5) / (0.5 + 0.5) =
0.5
Key
Count
Explanation
True positive
1
Utterance 5 was correctly predicted as sendEmail .
False positive
1
Utterance 2 was mistakenly predicted as sendEmail .
False negative
1
Utterance 4 was mistakenly predicted as Reply .
Precision = #True_Positive / (#True_Positive + #False_Positive) = 1 / (1 + 1) = 0.5
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 1 / (1 + 1) = 0.5
F1 score = 2 * Precision * Recall / (Precision + Recall) = (2 * 0.5 * 0.5) / (0.5 + 0.5) =
0.5
Key
Count
Explanation
True positive
1
Utterance 3 was correctly predicted as readEmail .
False positive
0
--
False negative
0
--
Precision = #True_Positive / (#True_Positive + #False_Positive) = 1 / (1 + 0) = 1
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 1 / (1 + 0) = 1
F1 score = 2 * Precision * Recall / (Precision + Recall) = (2 * 1 * 1) / (1 + 1) = 1
Intent-level evaluation for sendEmail intent
ﾉ
Expand table
Intent-level evaluation for readEmail intent
ﾉ
Expand table
\nKey
Count
Explanation
True positive
1
cynthia  was correctly predicted as contactName  in utterance 4.
False positive
0
--
False negative
1
mike  was mistakenly predicted as message  in utterance 5.
Precision = #True_Positive / (#True_Positive + #False_Positive) = 1 / (1 + 0) = 1
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 1 / (1 + 1) = 0.5
F1 score = 2 * Precision * Recall / (Precision + Recall) = (2 * 1 * 0.5) / (1 + 0.5) =
0.67
Key
Count
Explanation
True
positive
2
thank you very much  was correctly predicted as message  in utterance 1 and dinner
last week was splendid  was correctly predicted as message  in utterance 4.
False
positive
1
mike  was mistakenly predicted as message  in utterance 5.
False
negative
1
yes  wasn't predicted as message  in utterance 2.
Precision = #True_Positive / (#True_Positive + #False_Positive) = 2 / (2 + 1) = 0.67
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 2 / (2 + 1) = 0.67
F1 Score = 2 * Precision * Recall / (Precision + Recall) = (2 * 0.67 * 0.67) / (0.67 +
0.67) = 0.67
Entity-level evaluation for contactName entity
ﾉ
Expand table
Entity-level evaluation for message entity
ﾉ
Expand table
Model-level evaluation for the collective model
ﾉ
Expand table
\nKey
Count
Explanation
True positive
6
Sum of true positives for all intents and entities.
False positive
3
Sum of false positives for all intents and entities.
False negative
4
Sum of false negatives for all intents and entities.
Precision = #True_Positive / (#True_Positive + #False_Positive) = 6 / (6 + 3) = 0.67
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 6 / (6 + 4) = 0.60
F1 score = 2 * Precision * Recall / (Precision + Recall) = (2 * 0.67 * 0.60) / (0.67 +
0.60) = 0.63
A confusion matrix is an N x N matrix used for model performance evaluation, where N is the
number of entities or intents. The matrix compares the expected labels with the ones predicted
by the model. The matrix gives a holistic view of how well the model is performing and what
kinds of errors it's making.
You can use the confusion matrix to identify intents or entities that are too close to each other
and often get mistaken (ambiguity). In this case, consider merging these intents or entities
together. If merging isn't possible, consider adding more tagged examples of both intents or
entities to help the model differentiate between them.
The highlighted diagonal in the following image shows the correctly predicted entities, where
the predicted tag is the same as the actual tag.
Confusion matrix
\nYou can calculate the intent-level or entity-level and model-level evaluation metrics from the
confusion matrix:
The values in the diagonal are the true positive values of each intent or entity.
The sum of the values in the intent or entities rows (excluding the diagonal) is the false
positive of the model.
The sum of the values in the intent or entities columns (excluding the diagonal) is the
false negative of the model.
Similarly:
The true positive of the model is the sum of true positives for all intents or entities.
The false positive of the model is the sum of false positives for all intents or entities.
The false negative of the model is the sum of false negatives for all intents or entities.
After you train your model, you see some guidance and recommendations on how to improve
the model. We recommend that you have a model covering every point in the guidance
section.
Training set has enough data: When an intent or entity has fewer than 15 labeled
instances in the training data, it can lead to lower accuracy because the model isn't
adequately trained on that intent. In this case, consider adding more labeled data in the
training set. You should only consider adding more labeled data to your entity if your

Guidance
\n![Image](images/page344_image1.png)
\nentity has a learned component. If your entity is defined only by list, prebuilt, and regex
components, this recommendation doesn't apply.
All intents or entities are present in test set: When the testing data lacks labeled
instances for an intent or entity, the model evaluation is less comprehensive because of
untested scenarios. Consider having test data for every intent and entity in your model to
ensure that everything is being tested.
Unclear distinction between intents or entities: When data is similar for different intents
or entities, it can lead to lower accuracy because they might be frequently misclassified as
each other. Review the following intents and entities and consider merging them if they're
similar. Otherwise, add more examples to better distinguish them from each other. You
can check the Confusion matrix tab for more guidance. If you're seeing two entities
constantly being predicted for the same spans because they share the same list, prebuilt,
or regex components, make sure to add a learned component for each entity and make it
required. Learn more about entity components.
Train a model in Language Studio
Related content
\nData formats accepted by conversational
language understanding
06/05/2025
If you're uploading your data into conversational language understanding, it must follow a
specific format. Use this article to learn more about accepted data formats.
If you're importing a project into conversational language understanding, the file uploaded
must be in the following format:
JSON
Import project file format
{
  "projectFileVersion": "2022-10-01-preview",
  "stringIndexType": "Utf16CodeUnit",
  "metadata": {
    "projectKind": "Conversation",
    "projectName": "{PROJECT-NAME}",
    "multilingual": true,
    "description": "DESCRIPTION",
    "language": "{LANGUAGE-CODE}",
    "settings": {
            "confidenceThreshold": 0
        }
  },
  "assets": {
    "projectKind": "Conversation",
    "intents": [
      {
        "category": "intent1"
      }
    ],
    "entities": [
      {
        "category": "entity1",
        "compositionSetting": "{COMPOSITION-SETTING}",
        "list": {
          "sublists": [
            {
              "listKey": "list1",
              "synonyms": [
                {
                  "language": "{LANGUAGE-CODE}",
                  "values": [
                    "{VALUES-FOR-LIST}"
                  ]
\nKey
Placeholder
Value
Example
{API-VERSION}
The version of
the API you're
calling.
2023-04-01
                }
              ]
            }            
          ]
        },
        "prebuilts": [
          {
            "category": "{PREBUILT-COMPONENTS}"
          }
        ],
        "regex": {
          "expressions": [
              {
                  "regexKey": "regex1",
                  "language": "{LANGUAGE-CODE}",
                  "regexPattern": "{REGEX-PATTERN}"
              }
          ]
        },
        "requiredComponents": [
            "{REQUIRED-COMPONENTS}"
        ]
      }
    ],
    "utterances": [
      {
        "text": "utterance1",
        "intent": "intent1",
        "language": "{LANGUAGE-CODE}",
        "dataset": "{DATASET}",
        "entities": [
          {
            "category": "ENTITY1",
            "offset": 6,
            "length": 4
          }
        ]
      }
    ]
  }
}
ﾉ
Expand table
\nKey
Placeholder
Value
Example
confidenceThreshold
{CONFIDENCE-
THRESHOLD}
This is the threshold score below which
the intent is predicted as None intent.
Values are from 0  to 1 .
0.7
projectName
{PROJECT-NAME}
The name of your project. This value is
case sensitive.
EmailApp
multilingual
true
A Boolean value that enables you to have
utterances in multiple languages in your
dataset. When your model is deployed,
you can query the model in any
supported language (not necessarily
included in your training documents. For
more information about supported
language codes, see Language support.
true
sublists
[]
Array that contains sublists. Each sublist is
a key and its associated values.
[]
compositionSetting
{COMPOSITION-
SETTING}
Rule that defines how to manage multiple
components in your entity. Options are
combineComponents  or
separateComponents .
combineComponents
synonyms
[]
Array that contains all the synonyms.
synonym
language
{LANGUAGE-
CODE}
A string specifying the language code for
the utterances, synonyms, and regular
expressions used in your project. If your
project is a multilingual project, choose
the language code of most the
utterances.
en-us
intents
[]
Array that contains all the intents you
have in the project. These intents are
classified from your utterances.
[]
entities
[]
Array that contains all the entities in your
project. These entities are extracted from
your utterances. Every entity can have
other optional components defined with
them: list, prebuilt, or regex.
[]
dataset
{DATASET}
The test set to which this utterance goes
to when it's split before training. To learn
more about data splitting, see Train your
conversational language understanding
Train
\nKey
Placeholder
Value
Example
model. Possible values for this field are
Train  and Test .
category
The type of entity associated with the
span of text specified.
Entity1
offset
The inclusive character position of the
start of the entity.
5
length
The character length of the entity.
5
listKey
A normalized value for the list of
synonyms to map back to in prediction.
Microsoft
values
{VALUES-FOR-
LIST}
A list of comma-separated strings that
are matched exactly for extraction and
map to the list key.
"msft",
"microsoft", "MS"
regexKey
{REGEX-
PATTERN}
A normalized value for the regular
expression to map back to in prediction.
ProductPattern1
regexPattern
{REGEX-
PATTERN}
A regular expression.
^pre
prebuilts
{PREBUILT-
COMPONENTS}
The prebuilt components that can extract
common types. For the list of prebuilts
you can add, see Supported prebuilt
entity components.
Quantity.Number
requiredComponents
{REQUIRED-
COMPONENTS}
A setting that specifies a requirement that
a specific component must be present to
return the entity. To learn more, see Entity
components. The possible values are
learned , regex , list , or prebuilts .
"learned",
"prebuilt"
Conversational language understanding offers the option to upload your utterances directly to
the project rather than typing them in one by one. You can find this option on the data labeling
page for your project.
JSON
Utterance file format
[
    {
        "text": "{Utterance-Text}",
        "language": "{LANGUAGE-CODE}",
\nKey
Placeholder
Value
Example
text
{Utterance-
Text}
Your utterance text.
Testing
language
{LANGUAGE-
CODE}
A string that specifies the language code for the utterances used
in your project. If your project is a multilingual project, choose the
language code of most of the utterances. For more information
about supported language codes, see Language support.
en-us
dataset
{DATASET}
The test set to which this utterance goes to when it's split before
training. To learn more about data splitting, see Train your
conversational language understanding model. Possible values for
this field are Train  and Test .
Train
intent
{intent}
The assigned intent.
intent1
entity
{entity}
The entity to be extracted.
entity1
        "dataset": "{DATASET}",
        "intent": "{intent}",
        "entities": [
            {
                "category": "{entity}",
                "offset": 19,
                "length": 10
            }
        ]
    },
    {
        "text": "{Utterance-Text}",
        "language": "{LANGUAGE-CODE}",
        "dataset": "{DATASET}",
        "intent": "{intent}",
        "entities": [
            {
                "category": "{entity}",
                "offset": 20,
                "length": 10
            },
            {
                "category": "{entity}",
                "offset": 31,
                "length": 5
            }
        ]
    }
]
ﾉ
Expand table