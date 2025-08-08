To add a regex component, select Add expression. Name the regex key, and enter a regular
expression that matches the entity to be extracted.
Select the Entity Options tab on the entity details page. When multiple components are
defined for an entity, their predictions might overlap. When an overlap occurs, each entity's
final prediction is determined based on the entity option that you select in this step. Select the
option that you want to apply to this entity, and then select Save.
After you create your entities, you can come back and edit them. You can edit entity
components or delete them by selecting Edit or Delete.
Add utterances and label your data
Define entity options
Related content
\nLabel your utterances in Azure AI Foundry
06/13/2025
After you build a schema for your fine-tuning task, you add training utterances to your project.
The utterances should be similar to what your users use when they interact with the project.
When you add an utterance, you have to assign which intent it belongs to. After the utterance
is added, label the words within your utterance that you want to extract as entities.
Data labeling is a crucial step in the conversational language understanding (CLU) trained
development lifecycle. This data is used in the next step when you train your model so that
your model can learn from the labeled data. If you already labeled utterances, you can directly
import them into your project, if your data follows the accepted data format. To learn more
about importing labeled data, see Create a CLU fine-tuning task. Labeled data informs the
model about how to interpret text and is used for training and evaluation.
A successfully created project.
For more information, see the CLU development lifecycle.
After you build your schema and create your project, you need to label your data. Labeling
your data is important so that your model knows which sentences and words are associated
with the intents and entities in your project. Spend time labeling your utterances to introduce
and refine the data that's used in training your models.
As you add utterances and label them, keep in mind:
The machine learning models generalize based on the labeled examples that you provide.
The more examples that you provide, the more data points the model has to make better
generalizations.
 Tip
Use the Quick Deploy option to implement custom CLU intent routing, which is powered
by your own large language model deployment without adding or labeling any training
data.
Prerequisites
Data labeling guidelines
\nThe precision, consistency, and completeness of your labeled data are key factors to
determining model performance:
Label precisely: Label each intent and entity to its right type always. Only include what
you want classified and extracted. Avoid unnecessary data in your labels.
Label consistently: The same entity should have the same label across all the
utterances.
Label completely: Provide varied utterances for every intent. Label all the instances of
the entity in all your utterances.
Ensure that the concepts that your entities refer to are well defined and separable. Check
if you can easily determine the differences reliably. If you can't, this lack of distinction
might indicate difficulty for the learned component.
Ensure that some aspect of your data can provide a signal for differences when there's a
similarity between entities.
For example, if you built a model to book flights, a user might use an utterance like "I
want a flight from Boston to Seattle." The origin city and destination city for such
utterances would be expected to be similar. A signal to differentiate origin city might be
that the word from often precedes it.
Ensure that you label all instances of each entity in both your training and testing data.
One approach is to use the search function to find all instances of a word or phrase in
your data to check if they're correctly labeled.
Ensure that you label test data for entities without learned components and also for the
entities with them. This practice helps to ensure that your evaluation metrics are accurate.
For multilingual projects, adding utterances in other languages increases the model's
performance in these languages. Avoid duplicating your data across all the languages
that you want to support. For example, to improve a calender bot's performance with
users, a developer might add examples mostly in English and a few in Spanish or French.
They might add utterances such as:
"Set a meeting with Matt and Kevin tomorrow at 12 PM." (English)
"Reply as tentative to the weekly update meeting." (English)
"Cancelar mi próxima reunión." (Spanish)
Use the following steps to label your utterances:
Clearly label utterances
Label your utterances
\n1. Go to your project page in Azure AI Foundry
.
2. On the left pane, select Manage data. On this page, you can add your utterances and
label them. You can also upload your utterances directly by selecting Upload utterance
file from the top menu. Make sure to follow the accepted format.
3. By using the top tabs, you can change the view to Training set or Testing set. Learn more
about training and testing sets and how they're used for model training and evaluation.
4. From the Select intent dropdown menu, select one of the intents, the language of the
utterance (for multilingual projects), and the utterance itself. Press the Enter key in the
utterance's text box and add the utterance.
5. You have two options to label entities in an utterance:
Option
Description
Label by using a
brush
Select the brush icon next to an entity in the pane on the right, and then
highlight the text in the utterance that you want to label.

 Tip
If you plan to use Automatically split the testing set from training data splitting,
add all your utterances to the training set.
ﾉ
Expand table
\n![Image](images/page254_image1.png)
\nOption
Description
Label by using
inline menu
Highlight the word that you want to label as an entity, and a menu appears.
Select the entity that you want to label these words with.
6. In the pane on the right, on the Labels tab, you can find all the entity types in your
project and the count of labeled instances per each one.
7. On the Distribution tab, you can view the distribution across training and testing sets.
You have these options for viewing:
Total instances per labeled entity: You can view the count of all labeled instances of
a specific entity.
Unique utterances per labeled entity: Each utterance is counted if it contains at
least one labeled instance of this entity.
Utterances per intent: You can view the count of utterances per intent.
To remove a label:
1. From within your utterance, select the entity from which you want to remove a label.
2. Scroll through the menu that appears, and select Remove label.

７ Note
List, regex, and prebuilt components aren't shown on the data labeling page. All labels
here apply to the learned component only.
\n![Image](images/page255_image1.png)
\nTo delete an entity:
1. Select the garbage bin icon next to the entity that you want to edit in the pane on the
right.
2. Select Delete to confirm.
In CLU, use Azure OpenAI to suggest utterances to add to your project by using generative
language models. We recommend that you use an Azure AI Foundry resource while you use
CLU so that you don't need to connect multiple resources.
To use the Azure AI Foundry resource, you need to provide your Azure AI Foundry resource
with elevated access. To do so, access the Azure portal. Within your Azure AI resource, provide
access as a Cognitive Services User to itself. This step ensures that all parts of your resource
are communicating correctly.
You first need to get access and create a resource in Azure OpenAI. Next, create a connection
to the Azure OpenAI resource within the same Azure AI Foundry project in the Management
center on the left pane of the Azure AI Foundry page. You then need to create a deployment
for the Azure OpenAI models within the connected Azure OpenAI resource. To create a new
resource, follow the steps in Create and deploy an Azure OpenAI in Azure AI Foundry Models
resource.
Before you get started, the suggested utterances feature is available only if your Language
resource is in the following regions:
East US
South Central US
West Europe
On the Data labeling page:
1. Select Suggest utterances. A pane opens on the right and prompts you to select your
Azure OpenAI resource and deployment.
2. After you select an Azure OpenAI resource, select Connect so that your Language
resource has direct access to your Azure OpenAI resource. It assigns your Language
resource the Cognitive Services User role to your Azure OpenAI resource. Now your
current Language resource has access to Azure OpenAI. If the connection fails, follow
these steps to manually add the correct role to your Azure OpenAI resource.
Suggest utterances with Azure OpenAI
Connect with separate Language and Azure OpenAI resources
\n3. After the resource is connected, select the deployment. The model that we recommend
for the Azure OpenAI deployment is gpt-35-turbo-instruct .
4. Select the intent for which you want to get suggestions. Make sure the intent that you
selected has at least five saved utterances to be enabled for utterance suggestions. The
suggestions provided by Azure OpenAI are based on the most recent utterances that you
added for that intent.
5. Select Generate utterances.
The suggested utterances show up with a dotted line around them and the note
Generated by AI. Those suggestions must be accepted or rejected. Accepting a
suggestion adds it to your project, as if you had added it yourself. Rejecting a suggestion
deletes it entirely. Only accepted utterances are part of your project and used for training
or testing.
To accept or reject, select the green check mark or red cancel buttons beside each
utterance. You can also use Accept all and Reject all on the toolbar.
Use of this feature entails a charge to your Azure OpenAI resource for a similar number of
tokens to the suggested utterances that are generated. For information on Azure OpenAI
pricing, see Azure OpenAI Service pricing
.
Enable identity management for your Language resource by using the following options.

Add required configurations to Azure OpenAI resource
\n![Image](images/page257_image1.png)
\nYour Language resource must have identity management. To enable it by using the Azure
portal
:
1. Go to your Language resource.
2. On the left pane, under the Resource Management section, select Identity.
3. On the System assigned tab, set Status to On.
After you enable managed identity, assign the Cognitive Services User role to your Azure
OpenAI resource by using the managed identity of your Language resource.
1. Sign in to the Azure portal
 and go to your Azure OpenAI resource.
2. Select the Access Control (IAM) tab.
3. Select Add > Add role assignment.
4. Select Job function roles and select Next.
5. Select Cognitive Services User from the list of roles, and select Next.
6. Select Assign access to: Managed identity and choose Select members.
7. Under Managed identity, select Language.
8. Search for your resource and select it. Then select Next and complete the process.
9. Review the details and select Review + assign.
Azure portal
\nAfter a few minutes, refresh Azure AI Foundry, and you can successfully connect to Azure
OpenAI.
Train your conversational language understanding model

Related content
\n![Image](images/page259_image1.png)
\nTrain a conversational language
understanding model
06/30/2025
After you complete labeling your utterances, you can start training a model. Training is the
process where the model learns from your labeled utterances.
To train a model, start a training job. Only successfully completed jobs create a model. Training
jobs expire after seven days, then you can no longer retrieve the job details. If your training job
completed successfully and a model was created, the job doesn't expire. You can only have one
training job running at a time, and you can't start other jobs in the same fine tuning task.
The training times can be anywhere from a few seconds for simple projects, up to several hours
when you reach the maximum limit of utterances.
Model evaluation is triggered automatically after training is completed successfully. The
evaluation process starts by using the trained model to run predictions on the utterances in the
testing set, and compares the predicted results with the provided labels (which establishes a
baseline of truth).
An active Azure subscription. If you don't have one, you can create one for free
.
Requisite permissions. Make sure the person establishing the account and project is
assigned as the Azure AI Account Owner role at the subscription level. Alternatively,
having either the Contributor or Cognitive Services Contributor role at the subscription
scope also meets this requirement. For more information, see Role based access control
(RBAC)
A project created in the Azure AI Foundry. For more information, see Create an AI
Foundry project
Your labeled utterances tagged for your fine tuning task.
７ Note
When using the Quick Deploy option, Conversational Language Understanding (CLU)
automatically creates an instant training job to set up your CLU intent router using your
selected LLM  deployment.
Prerequisites
Balance training data