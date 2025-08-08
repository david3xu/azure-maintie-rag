Once you have a Language resource created, create an orchestration workflow project. A
project is a work area for building your custom ML models based on your data. Your project
can only be accessed by you and others who have access to the Language resource being
used.
For this quickstart, complete the conversational language understanding quickstart to create a
conversational language understanding project that will be used later.
1. In Language Studio
, find the section labeled Understand questions and conversational
language and select Orchestration Workflow.

Create an orchestration workflow project
\n![Image](images/page781_image1.png)
\n2. This will bring you to the Orchestration workflow project page. Select Create new
project. To create a project, you will need to provide the following details:
Value
Description
Name
A name for your project.
Description
Optional project description.
Utterances primary
language
The primary language of your project. Your training data should primarily be
in this language.
Once you're done, select Next and review the details. Select create project to complete the
process. You should now see the Build Schema screen in your project.
After you complete the conversational language understanding quickstart and create an
orchestration project, the next step is to add intents.
To connect to the previously created conversational language understanding project:
In the build schema page in your orchestration project, select Add, to add an intent.
In the window that appears, give your intent a name.
Select Yes, I want to connect it to an existing project.
From the connected services dropdown, select Conversational Language Understanding.
From the project name dropdown, select your conversational language understanding
project.
Select Add intent to create your intent.

ﾉ
Expand table
Build schema
\n![Image](images/page782_image1.png)
\nTo train a model, you need to start a training job. The output of a successful training job is your
trained model.
To start training your model from within the Language Studio
:
1. Select Training jobs from the left side menu.
2. Select Start a training job from the top menu.
3. Select Train a new model and type in the model name in the text box. You can also
overwrite an existing model by selecting this option and choosing the model you want
to overwrite from the dropdown menu. Overwriting a trained model is irreversible, but it
won't affect your deployed models until you deploy the new model.
If you have enabled your project to manually split your data when tagging your
utterances, you will see two data splitting options:
Automatically splitting the testing set from training data: Your tagged utterances
will be randomly split between the training and testing sets, according to the
percentages you choose. The default percentage split is 80% for training and 20%
for testing. To change these values, choose which set you want to change and type
in the new value.
Use a manual split of training and testing data: Assign each utterance to either the
training or testing set during the tagging step of the project.
Train your model
７ Note
If you choose the Automatically splitting the testing set from training data option,
only the utterances in your training set will be split according to the percentages
provided.
７ Note
Use a manual split of training and testing data option will only be enabled if you
add utterances to the testing set in the tag data page. Otherwise, it will be disabled.
\n4. Select the Train button.
Generally after training a model you would review its evaluation details. In this quickstart, you
will just deploy your model, and make it available for you to try in Language Studio, or you can
call the prediction API
.
To deploy your model from within the Language Studio
:
1. Select Deploying a model from the left side menu.
2. Select Add deployment to start a new deployment job.

７ Note
Only successfully completed training jobs will generate models.
Training can take some time between a couple of minutes and couple of hours based
on the size of your tagged data.
You can only have one training job running at a time. You cannot start other training
job wihtin the same project until the running job is completed.
Deploy your model
\n![Image](images/page784_image1.png)
\n3. Select Create new deployment to create a new deployment and assign a trained model
from the dropdown below. You can also Overwrite an existing deployment by selecting
this option and select the trained model you want to assign to it from the dropdown
below.
4. If you're connecting one or more LUIS
 applications or conversational language
understanding
 projects, you have to specify the deployment name.

７ Note
Overwriting an existing deployment doesn't require changes to your prediction
API
 call, but the results you get will be based on the newly assigned model.

\n![Image](images/page785_image1.png)

![Image](images/page785_image2.png)
\nNo configurations are required for custom question answering or unlinked intents.
LUIS projects must be published to the slot configured during the Orchestration
deployment, and custom question answering KBs must also be published to their
Production slots.
5. Select Deploy to submit your deployment job
6. After deployment is successful, an expiration date will appear next to it. Deployment
expiration is when your deployed model will be unavailable to be used for prediction,
which typically happens twelve months after a training configuration expires.
After your model is deployed, you can start using it to make predictions through Prediction
API
. For this quickstart, you will use the Language Studio
 to submit an utterance, get
predictions and visualize the results.
To test your model from Language Studio
1. Select Testing deployments from the left side menu.
2. Select the model you want to test. You can only test models that are assigned to
deployments.
3. From deployment name dropdown, select your deployment name.
4. In the text box, enter an utterance to test.
5. From the top menu, select Run the test.
6. After you run the test, you should see the response of the model in the result. You can
view the results in entities cards view, or view it in JSON format.
Test model
\nWhen you don't need your project anymore, you can delete your project using Language
Studio. Select Projects from the left pane, select the project you want to delete, and then select
Delete from the top menu.
Learn about orchestration workflows

Clean up resources
Next steps
\n![Image](images/page787_image1.png)
\nFrequently asked questions for
orchestration workflows
06/21/2025
Use this article to quickly get the answers to common questions about orchestration workflows
See the quickstart to quickly create your first project, or the how-to article for more details.
See How to create projects and build schemas for information on connecting another project
as an intent.
LUIS applications that use the Language resource as their authoring resource will be available
for connection. You can only connect to LUIS applications that are owned by the same
resource. This option will only be available for resources in West Europe, as it's the only
common available region between LUIS and CLU.
Question answering projects that use the Language resource will be available for connection.
You can only connect to question answering projects that are in the same Language resource.
For orchestration projects, long training times are expected. Based on the number of examples
you have your training times may vary from 5 minutes to 1 hour or more.
How do I create a project?
How do I connect other service applications in
orchestration workflow projects?
Which LUIS applications can I connect to in
orchestration workflow projects?
Which question answering project can I connect to
in orchestration workflow projects?
Training is taking a long time, is this expected?
\nNo. Orchestration projects are only enabled for intents that can be connected to other projects
for routing.
See evaluation metrics for information on how models are evaluated, and metrics you can use
to improve accuracy.
Unlike LUIS, you cannot label the same text as 2 different entities. Learned components across
different entities are mutually exclusive, and only one learned span is predicted for each set of
characters.
Yes, only for predictions, and samples are available
. There is currently no authoring support
for the SDK.
Yes, all the APIs are available.
Authoring APIs
Prediction API
Orchestration workflow overview
Can I add entities to orchestration workflow
projects?
How do I get more accurate results for my project?
Can I label the same word as 2 different entities?
Is there any SDK support?
Are there APIs for this feature?
Next steps
\nLanguage support for orchestration
workflow projects
06/21/2025
Use this article to learn about the languages currently supported by orchestration workflow
projects.
Orchestration workflow projects do not support the multi-lingual option.
Orchestration workflow projects support the following languages:
Language
Language code
German
de
English
en-us
Spanish
es
French
fr
Italian
it
Portuguese (Brazil)
pt-br
Orchestration workflow overview
Service limits
Multilingual options
Language support
ﾉ
Expand table
Next steps