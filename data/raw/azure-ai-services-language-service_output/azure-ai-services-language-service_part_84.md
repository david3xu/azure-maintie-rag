Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating
your API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-NAME}
The name for your project. This
value is case-sensitive.
myProject
{DEPLOYMENT-
NAME}
The name for your deployment.
This value is case-sensitive.
staging
{JOB-ID}
The ID for locating your model's
training status.
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxxx
{API-VERSION}
The version of the API you are
calling.
2023-04-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
Once you send the request, you'll get the following response. Keep polling this endpoint until
the status parameter changes to "succeeded".
JSON
ﾉ
Expand table
Headers
ﾉ
Expand table
Response Body
{
    "jobId":"{JOB-ID}",
    "createdDateTime":"{CREATED-TIME}",
    "lastUpdatedDateTime":"{UPDATED-TIME}",
    "expirationDateTime":"{EXPIRATION-TIME}",
    "status":"running"
}
Changes in calling the runtime
\nWithin your system, at the step where you call runtime API
 check for the response code
returned from the submit task API. If you observe a consistent failure in submitting the request,
this could indicate an outage in your primary region. Failure once doesn't mean an outage, it
may be transient issue. Retry submitting the job through the secondary resource you have
created. For the second request use your {YOUR-SECONDARY-ENDPOINT}  and secondary key, if you
have followed the steps above, {PROJECT-NAME}  and {DEPLOYMENT-NAME}  would be the same so
no changes are required to the request body.
In case you revert to using your secondary resource you will observe slight increase in latency
because of the difference in regions where your model is deployed.
Maintaining the freshness of both projects is an important part of process. You need to
frequently check if any updates were made to your primary project so that you move them
over to your secondary project. This way if your primary region fail and you move into the
secondary region you should expect similar model performance since it already contains the
latest updates. Setting the frequency of checking if your projects are in sync is an important
choice, we recommend that you do this check daily in order to guarantee the freshness of data
in your secondary model.
Use the following url to get your project details, one of the keys returned in the body indicates
the last modified date of the project. Repeat the following step twice, one for your primary
project and another for your secondary project and compare the timestamp returned for both
of them to check if they are out of sync.
Use the following GET request to get your project details. You can use the URL you received
from the previous step, or replace the placeholder values below with your own values.
rest
Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating
your API request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
Check if your projects are out of sync
Get project details
{ENDPOINT}/language/authoring/analyze-conversations/projects/{PROJECT-NAME}?api-
version={API-VERSION}
ﾉ
Expand table
\nPlaceholder
Value
Example
{PROJECT-
NAME}
The name for your project. This
value is case-sensitive.
myProject
{API-
VERSION}
The version of the API you are
calling.
2023-04-01
Use the following header to authenticate your request.
Key
Description
Value
Ocp-Apim-
Subscription-Key
The key to your resource. Used for authenticating
your API requests.
{YOUR-PRIMARY-
RESOURCE-KEY}
JSON
Repeat the same steps for your replicated project using {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY} . Compare the returned lastModifiedDateTime  from both projects. If your
primary project was modified sooner than your secondary one, you need to repeat the steps of
exporting, importing, training and deploying your model.
In this article, you have learned how to use the export and import APIs to replicate your project
to a secondary Language resource in other region. Next, explore the API reference docs to see
Headers
ﾉ
Expand table
Response body
{
  "createdDateTime": "2022-04-18T13:53:03Z",
  "lastModifiedDateTime": "2022-04-18T13:53:03Z",
  "lastTrainedDateTime": "2022-04-18T14:14:28Z",
  "lastDeployedDateTime": "2022-04-18T14:49:01Z",
  "projectKind": "Conversation",
  "projectName": "{PROJECT-NAME}",
  "multilingual": true,
  "description": "This is a sample conversation project.",
  "language": "{LANGUAGE-CODE}"
}
Next steps
\nwhat else you can do with authoring APIs.
Authoring REST API reference
Runtime prediction REST API reference
\nEvaluation metrics for orchestration
workflow models
06/21/2025
Your dataset is split into two parts: a set for training, and a set for testing. The training set is
used to train the model, while the testing set is used as a test for model after training to
calculate the model performance and evaluation. The testing set isn't introduced to the model
through the training process, to make sure that the model is tested on new data.
Model evaluation is triggered automatically after training is completed successfully. The
evaluation process starts by using the trained model to predict user defined intents for
utterances in the test set, and compares them with the provided tags (which establishes a
baseline of truth). The results are returned so you can review the model’s performance. For
evaluation, orchestration workflow uses the following metrics:
Precision: Measures how precise/accurate your model is. It's the ratio between the
correctly identified positives (true positives) and all identified positives. The precision
metric reveals how many of the predicted classes are correctly labeled.
Precision = #True_Positive / (#True_Positive + #False_Positive)
Recall: Measures the model's ability to predict actual positive classes. It's the ratio
between the predicted true positives and what was actually tagged. The recall metric
reveals how many of the predicted classes are correct.
Recall = #True_Positive / (#True_Positive + #False_Negatives)
F1 score: The F1 score is a function of Precision and Recall. It's needed when you seek a
balance between Precision and Recall.
F1 Score = 2 * Precision * Recall / (Precision + Recall)
Precision, recall, and F1 score are calculated for:
Each intent separately (intent-level evaluation)
For the model collectively (model-level evaluation).
The definitions of precision, recall, and evaluation are the same for intent-level and model-level
evaluations. However, the counts for True Positives, False Positives, and False Negatives can
differ. For example, consider the following text.
Example
\nMake a response with thank you very much
Call my friend
Hello
Good morning
These are the intents used: CLUEmail and Greeting
The model could make the following predictions:
Utterance
Predicted intent
Actual intent
Make a response with thank you very much
CLUEmail
CLUEmail
Call my friend
Greeting
CLUEmail
Hello
CLUEmail
Greeting
Goodmorning
Greeting
Greeting
Key
Count
Explanation
True Positive
1
Utterance 1 was correctly predicted as CLUEmail.
False Positive
1
Utterance 3 was mistakenly predicted as CLUEmail.
False Negative
1
Utterance 2 was mistakenly predicted as Greeting.
Precision = #True_Positive / (#True_Positive + #False_Positive) = 1 / (1 + 1) = 0.5
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 1 / (1 + 1) = 0.5
F1 Score = 2 * Precision * Recall / (Precision + Recall) = (2 * 0.5 * 0.5) / (0.5 + 0.5)
= 0.5
ﾉ
Expand table
Intent level evaluation for CLUEmail intent
ﾉ
Expand table
Intent level evaluation for Greeting intent
ﾉ
Expand table
\nKey
Count
Explanation
True Positive
1
Utterance 4 was correctly predicted as Greeting.
False Positive
1
Utterance 2 was mistakenly predicted as Greeting.
False Negative
1
Utterance 3 was mistakenly predicted as CLUEmail.
Precision = #True_Positive / (#True_Positive + #False_Positive) = 1 / (1 + 1) = 0.5
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 1 / (1 + 1) = 0.5
F1 Score = 2 * Precision * Recall / (Precision + Recall) = (2 * 0.5 * 0.5) / (0.5 + 0.5)
= 0.5
Key
Count
Explanation
True Positive
2
Sum of TP for all intents
False Positive
2
Sum of FP for all intents
False Negative
2
Sum of FN for all intents
Precision = #True_Positive / (#True_Positive + #False_Positive) = 2 / (2 + 2) = 0.5
Recall = #True_Positive / (#True_Positive + #False_Negatives) = 2 / (2 + 2) = 0.5
F1 Score = 2 * Precision * Recall / (Precision + Recall) = (2 * 0.5 * 0.5) / (0.5 + 0.5)
= 0.5
A Confusion matrix is an N x N matrix used for model performance evaluation, where N is the
number of intents. The matrix compares the actual tags with the tags predicted by the model.
This gives a holistic view of how well the model is performing and what kinds of errors it is
making.
You can use the Confusion matrix to identify intents that are too close to each other and often
get mistaken (ambiguity). In this case consider merging these intents together. If that isn't
Model-level evaluation for the collective model
ﾉ
Expand table
Confusion matrix
\npossible, consider adding more tagged examples of both intents to help the model
differentiate between them.
You can calculate the model-level evaluation metrics from the confusion matrix:
The true positive of the model is the sum of true Positives for all intents.
The false positive of the model is the sum of false positives for all intents.
The false Negative of the model is the sum of false negatives for all intents.
Train a model in Language Studio
Next steps
\nBack up and recover your orchestration
workflow models
06/21/2025
When you create a Language resource in the Azure portal, you specify a region for it to be
created in. From then on, your resource and all of the operations related to it take place in the
specified Azure server region. It's rare, but not impossible, to encounter a network issue that
hits an entire region. If your solution needs to always be available, then you should design it to
either fail-over into another region. This requires two Azure AI Language resources in different
regions and the ability to sync your orchestration workflow models across regions.
If your app or business depends on the use of an orchestration workflow model, we
recommend that you create a replica of your project into another supported region. So that if a
regional outage occurs, you can then access your model in the other fail-over region where
you replicated your project.
Replicating a project means that you export your project metadata and assets and import them
into a new project. This only makes a copy of your project settings, intents and utterances. You
still need to train and deploy the models before you can query them with the runtime APIs
.
In this article, you will learn to how to use the export and import APIs to replicate your project
from one resource to another existing in different supported geographical regions, guidance
on keeping your projects in sync and changes needed to your runtime consumption.
Two Azure AI Language resources in different Azure regions, each of them in a different
region.
Use the following steps to get the keys and endpoint of your primary and secondary resources.
These will be used in the following steps.
Go to your resource overview page in the Azure portal
. From the menu on the left side,
select Keys and Endpoint. You will use the endpoint and key for API requests.
Prerequisites
Get your resource keys endpoint
\nStart by exporting the project assets from the project in your primary resource.
Replace the placeholders in the following request with your {PRIMARY-ENDPOINT}  and {PRIMARY-
RESOURCE-KEY}  that you obtained in the first step.
Create a POST request using the following URL, headers, and JSON body to export your
project.
Use the following URL when creating your API request. Replace the placeholder values below
with your own values.
rest

 Tip
Keep a note of keys and endpoints for both primary and secondary resources. Use these
values to replace the following placeholders: {PRIMARY-ENDPOINT} , {PRIMARY-RESOURCE-
KEY} , {SECONDARY-ENDPOINT}  and {SECONDARY-RESOURCE-KEY} . Also take note of your project
name, your model name and your deployment name. Use these values to replace the
following placeholders: {PROJECT-NAME} , {MODEL-NAME}  and {DEPLOYMENT-NAME} .
Export your primary project assets
Submit export job
Request URL
\n![Image](images/page840_image1.png)