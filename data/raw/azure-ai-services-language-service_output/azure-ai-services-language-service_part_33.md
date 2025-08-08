This schema design makes it easy for you to extend new actions to existing targets by adding
new action entities or entity components.
Make sure to avoid trying to funnel all the concepts into intents. For example, don't try to
create a Cancel Contoso intent that only has the purpose of that one specific action. Intents and
entities should work together to capture all the required information from the customer.
You also want to avoid mixing different schema designs. Don't build half of your application
with actions as intents and the other half with information as intents. To get the possible
results, ensure that it's consistent.
When it comes to training data, try to keep your schema well balanced. Including large
quantities of one intent and very few of another results in a model that's biased toward
particular intents.
To address this scenario, you might need to downsample your training set. Or you might need
to add to it. To downsample, you can:
Get rid of a certain percentage of the training data randomly.
Analyze the dataset and remove overrepresented duplicate entries, which is a more
systematic manner.
To add to the training set, in Language Studio, on the Data labeling tab, select Suggest
utterances. Conversational Language Understanding sends a call to Azure OpenAI to generate
similar utterances.
Balance training data
\nYou should also look for unintended "patterns" in the training set. For example, look to see if
the training set for a particular intent is all lowercase or starts with a particular phrase. In such
cases, the model you train might learn these unintended biases in the training set instead of
being able to generalize.
We recommend that you introduce casing and punctuation diversity in the training set. If your
model is expected to handle variations, be sure to have a training set that also reflects that
diversity. For example, include some utterances in proper casing and some in all lowercase.
Ensure that the concepts that your entities refer to are well defined and separable. Check
if you can easily determine the differences reliably. If you can't, this lack of distinction
might indicate that the learned component will also have difficulty.
If there's a similarity between entities, ensure that there's some aspect of your data that
provides a signal for the difference between them.
For example, if you built a model to book flights, a user might use an utterance like "I
want a flight from Boston to Seattle." The origin city and destination city for such
utterances would be expected to be similar. A signal to differentiate origin city might be
that the word from often precedes it.
Ensure that you label all instances of each entity in both your training and testing data.
One approach is to use the search function to find all instances of a word or phrase in

Clearly label utterances
\n![Image](images/page322_image1.png)
\nyour data to check if they're correctly labeled.
Label test data for entities that have no learned component and also for the entities that
do. This practice helps to ensure that your evaluation metrics are accurate.
Standard training is free and faster than advanced training. It can help you quickly understand
the effect of changing your training set or schema while you build the model. After you're
satisfied with the schema, consider using advanced training to get the best model quality.
When you build an app, it's often helpful to catch errors early. It's usually a good practice to
add a test set when you build the app. Training and evaluation results are useful in identifying
errors or issues in your schema.
For more information, see Component types.
If you see too many false positives, such as out-of-context utterances being marked as valid
intents, see Confidence threshold for information on how it affects inference.
Non-machine-learned entity components, like lists and regex, are by definition not
contextual. If you see list or regex entities in unintended places, try labeling the list
synonyms as the machine-learned component.
For entities, you can use learned component as the Required component, to restrict when
a composed entity should fire.
For example, suppose you have an entity called Ticket Quantity that attempts to extract the
number of tickets you want to reserve for booking flights, for utterances such as "Book two
tickets tomorrow to Cairo."
Typically, you add a prebuilt component for Quantity.Number  that already extracts all numbers
in utterances. However, if your entity was only defined with the prebuilt component, it also
extracts other numbers as part of the Ticket Quantity entity, such as "Book two tickets
tomorrow to Cairo at 3 PM."
Use standard training before advanced training
Use the evaluation feature
Machine-learning components and composition
Use the None score threshold
\nTo resolve this issue, you label a learned component in your training data for all the numbers
that are meant to be a ticket quantity. The entity now has two components:
The prebuilt component that can interpret all numbers.
The learned component that predicts where the ticket quantity is located in a sentence.
If you require the learned component, make sure that Ticket Quantity is only returned when
the learned component predicts it in the right context. If you also require the prebuilt
component, you can then guarantee that the returned Ticket Quantity entity is both a number
and in the correct position.
If your model is overly sensitive to small grammatical changes, like casing or diacritics, you can
systematically manipulate your dataset directly in Language Studio. To use these features,
select the Settings tab on the left pane and locate the Advanced project settings section.
First, you can enable the setting for Enable data transformation for casing, which normalizes
the casing of utterances when training, testing, and implementing your model. If you migrated
from LUIS, you might recognize that LUIS did this normalization by default. To access this
feature via the API, set the normalizeCasing  parameter to true . See the following example:
JSON
Second, you can also enable the setting for Enable data augmentation for diacritics to
generate variations of your training data for possible diacritic variations used in natural
language. This feature is available for all languages. It's especially useful for Germanic and
Slavic languages, where users often write words by using classic English characters instead of
Address model inconsistencies

{
  "projectFileVersion": "2022-10-01-preview",
    ...
    "settings": {
      ...
      "normalizeCasing": true
      ...
    }
...
\n![Image](images/page324_image1.png)
\nthe correct characters. For example, the phrase "Navigate to the sports channel" in French is
"Accédez à la chaîne sportive." When this feature is enabled, the phrase "Accedez a la chaine
sportive" (without diacritic characters) is also included in the training dataset.
If you enable this feature, the utterance count of your training set increases. For this reason,
you might need to adjust your training data size accordingly. The current maximum utterance
count after augmentation is 25,000. To access this feature via the API, set the
augmentDiacritics  parameter to true . See the following example:
JSON
Customers can use the LoraNorm training configuration version if the model is being
incorrectly overconfident. An example of this behavior can be like the following scenario where
the model predicts the incorrect intent with 100% confidence. This score makes the confidence
threshold project setting unusable.
Text
Predicted intent
Confidence score
"Who built the Eiffel Tower?"
Sports
1.00
"Do I look good to you today?"
QueryWeather
1.00
"I hope you have a good evening."
Alarm
1.00
To address this scenario, use the 2023-04-15  configuration version that normalizes confidence
scores. The confidence threshold project setting can then be adjusted to achieve the desired
result.
Console
{
  "projectFileVersion": "2022-10-01-preview",
    ...
    "settings": {
      ...
      "augmentDiacritics": true
      ...
    }
...
Address model overconfidence
ﾉ
Expand table
\nAfter the request is sent, you can track the progress of the training job in Language Studio as
usual.
With model version 2023-04-15, conversational language understanding provides
normalization in the inference layer that doesn't affect training.
The normalization layer normalizes the classification confidence scores to a confined range.
The range selected currently is from [-a,a]  where "a" is the square root of the number of
intents. As a result, the normalization depends on the number of intents in the app. If the
number of intents is low, the normalization layer has a small range to work with. With a large
number of intents, the normalization is more effective.
If this normalization doesn't seem to help intents that are out of scope to the extent that the
confidence threshold can be used to filter out-of-scope utterances, it might be related to the
number of intents in the app. Consider adding more intents to the app. Or, if you're using an
orchestrated architecture, consider merging apps that belong to the same domain together.
curl --location 'https://<your-
resource>.cognitiveservices.azure.com/language/authoring/analyze-
conversations/projects/<your-project>/:train?api-version=2022-10-01-preview' \
--header 'Ocp-Apim-Subscription-Key: <your subscription key>' \
--header 'Content-Type: application/json' \
--data '{
      "modelLabel": "<modelLabel>",
      "trainingMode": "advanced",
      "trainingConfigVersion": "2023-04-15",
      "evaluationOptions": {
            "kind": "percentage",
            "testingSplitPercentage": 0,
            "trainingSplitPercentage": 100
      }
}
７ Note
You have to retrain your model after you update the confidenceThreshold  project setting.
Afterward, you need to republish the app for the new threshold to take effect.
Normalization in model version 2023-04-15
Debug composed entities
\nEntities are functions that emit spans in your input with an associated type. One or more
components define the function. You can mark components as needed, and you can decide
whether to enable the Combine components setting. When you combine components, all
spans that overlap are merged into a single span. If the setting isn't used, each individual
component span is emitted.
To better understand how individual components are performing, you can disable the setting
and set each component to Not required. This setting lets you inspect the individual spans that
are emitted and experiment with removing components so that only problematic components
are generated.
Data in a conversational language understanding project can have two datasets: a testing set
and a training set. If you want to use multiple test sets to evaluate your model, you can:
Give your test sets different names (for example, "test1" and "test2").
Export your project to get a JSON file with its parameters and configuration.
Use the JSON to import a new project. Rename your second desired test set to "test."
Train the model to run the evaluation by using your second test set.
If you're using orchestrated apps, you might want to send custom parameter overrides for
various child apps. The targetProjectParameters  field allows users to send a dictionary
representing the parameters for each target project. For example, consider an orchestrator app
named Orchestrator  orchestrating between a conversational language understanding app
named CLU1  and a custom question answering app named CQA1 . If you want to send a
parameter named "top" to the question answering app, you can use the preceding parameter.
Console
Evaluate a model by using multiple test sets
Custom parameters for target apps and child apps
curl --request POST \
   --url 'https://<your-language-
resource>.cognitiveservices.azure.com/language/:analyze-conversations?api-
version=2022-10-01-preview' \
   --header 'ocp-apim-subscription-key: <your subscription key>' \
   --data '{
     "kind": "Conversation",
     "analysisInput": {
         "conversationItem": {
             "id": "1",
             "text": "Turn down the volume",
             "modality": "text",
\nOften you can copy conversational language understanding projects from one resource to
another by using the Copy button in Language Studio. In some cases, it might be easier to
copy projects by using the API.
First, identify the:
Source project name.
Target project name.
Source language resource.
Target language resource, which is where you want to copy it to.
Call the API to authorize the copy action and get accessTokens  for the actual copy operation
later.
Console
Call the API to complete the copy operation. Use the response you got earlier as the payload.
             "language": "en-us",
             "participantId": "1"
         }
     },
     "parameters": {
         "projectName": "Orchestrator",
         "verbose": true,
         "deploymentName": "std",
         "stringIndexType": "TextElement_V8",
"targetProjectParameters": {
            "CQA1": {
                "targetProjectKind": "QuestionAnswering",
                "callingOptions": {
                    "top": 1
                }
             }
         }
     }
 }'
Copy projects across language resources
curl --request POST \ 
  --url 'https://<target-language-
resource>.cognitiveservices.azure.com//language/authoring/analyze-
conversations/projects/<source-project-name>/:authorize-copy?api-version=2023-04-
15-preview' \ 
  --header 'Content-Type: application/json' \ 
  --header 'Ocp-Apim-Subscription-Key: <Your-Subscription-Key>' \ 
  --data '{"projectKind":"Conversation","allowOverwrite":false}' 
\nConsole
Customers can use the newly updated training configuration version 2024-08-01-preview
(previously 2024-06-01-preview ) if the model has poor quality on out-of-domain utterances. An
example of this scenario with the default training configuration can be like the following
example where the model has three intents: Sports , QueryWeather , and Alarm . The test
utterances are out-of-domain utterances and the model classifies them as InDomain  with a
relatively high confidence score.
Text
Predicted intent
Confidence score
"Who built the Eiffel Tower?"
Sports
0.90
"Do I look good to you today?"
QueryWeather
1.00
"I hope you have a good evening."
Alarm
0.80
To address this scenario, use the 2024-08-01-preview  configuration version that's built
specifically to address this issue while also maintaining reasonably good quality on InDomain
utterances.
Console
curl --request POST \ 
  --url 'https://<source-language-
resource>.cognitiveservices.azure.com/language/authoring/analyze-
conversations/projects/<source-project-name>/:copy?api-version=2023-04-15-preview' 
\ 
  --header 'Content-Type: application/json' \ 
  --header 'Ocp-Apim-Subscription-Key: <Your-Subscription-Key>\ 
  --data '{ 
"projectKind": "Conversation", 
"targetProjectName": "<target-project-name>", 
"accessToken": "<access-token>", 
"expiresAt": "<expiry-date>", 
"targetResourceId": "<target-resource-id>", 
"targetResourceRegion": "<target-region>" 
}'
Address out-of-domain utterances
ﾉ
Expand table
curl --location 'https://<your-
resource>.cognitiveservices.azure.com/language/authoring/analyze-
conversations/projects/<your-project>/:train?api-version=2022-10-01-preview' \
\nAfter the request is sent, you can track the progress of the training job in Language Studio as
usual.
Caveats:
The None score threshold for the app (confidence threshold below which topIntent  is
marked as None ) when you use this training configuration should be set to 0. This setting
is used because this new training configuration attributes a certain portion of the in-
domain probabilities to out of domain so that the model isn't incorrectly overconfident
about in-domain utterances. As a result, users might see slightly reduced confidence
scores for in-domain utterances as compared to the production training configuration.
We don't recommend this training configuration for apps with only two intents, such as
IntentA  and None , for example.
We don't recommend this training configuration for apps with a low number of
utterances per intent. We highly recommend a minimum of 25 utterances per intent.
--header 'Ocp-Apim-Subscription-Key: <your subscription key>' \
--header 'Content-Type: application/json' \
--data '{
      "modelLabel": "<modelLabel>",
      "trainingMode": "advanced",
      "trainingConfigVersion": "2024-08-01-preview",
      "evaluationOptions": {
            "kind": "percentage",
            "testingSplitPercentage": 0,
            "trainingSplitPercentage": 100
      }
}