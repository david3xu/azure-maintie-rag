Train your orchestration workflow model
06/21/2025
Training is the process where the model learns from your labeled utterances. After training is
completed, you will be able to view model performance.
To train a model, start a training job. Only successfully completed jobs create a model. Training
jobs expire after seven days, after this time you will no longer be able to retrieve the job
details. If your training job completed successfully and a model was created, it won't be
affected by the job expiring. You can only have one training job running at a time, and you
can't start other jobs in the same project.
The training times can be anywhere from a few seconds when dealing with simple projects, up
to a couple of hours when you reach the maximum limit of utterances.
Model evaluation is triggered automatically after training is completed successfully. The
evaluation process starts by using the trained model to run predictions on the utterances in the
testing set, and compares the predicted results with the provided labels (which establishes a
baseline of truth). The results are returned so you can review the model’s performance.
A successfully created project with a configured Azure blob storage account
See the project development lifecycle for more information.
Before you start the training process, labeled utterances in your project are divided into a
training set and a testing set. Each one of them serves a different function. The training set is
used in training the model, this is the set from which the model learns the labeled utterances.
The testing set is a blind set that isn't introduced to the model during training but only during
evaluation.
After the model is trained successfully, the model can be used to make predictions from the
utterances in the testing set. These predictions are used to calculate evaluation metrics.
It is recommended to make sure that all your intents are adequately represented in both the
training and testing set.
Orchestration workflow supports two methods for data splitting:
Prerequisites
Data splitting
\nAutomatically splitting the testing set from training data: The system will split your
tagged data between the training and testing sets, according to the percentages you
choose. The recommended percentage split is 80% for training and 20% for testing.
Use a manual split of training and testing data: This method enables users to define
which utterances should belong to which set. This step is only enabled if you have added
utterances to your testing set during labeling.
To start training your model from within the Language Studio
:
1. Select Training jobs from the left side menu.
2. Select Start a training job from the top menu.
3. Select Train a new model and type in the model name in the text box. You can also
overwrite an existing model by selecting this option and choosing the model you
want to overwrite from the dropdown menu. Overwriting a trained model is
irreversible, but it won't affect your deployed models until you deploy the new
model.
If you have enabled your project to manually split your data when tagging your
utterances, you will see two data splitting options:
Automatically splitting the testing set from training data: Your tagged
utterances will be randomly split between the training and testing sets,
according to the percentages you choose. The default percentage split is 80%
７ Note
If you choose the Automatically splitting the testing set from training data option, only
the data assigned to training set will be split according to the percentages provided.
７ Note
You can only add utterances in the training dataset for non-connected intents only.
Train model
Start training job
Language Studio
\nfor training and 20% for testing. To change these values, choose which set you
want to change and type in the new value.
Use a manual split of training and testing data: Assign each utterance to either
the training or testing set during the tagging step of the project.
4. Select the Train button.
７ Note
If you choose the Automatically splitting the testing set from training data
option, only the utterances in your training set will be split according to the
percentages provided.
７ Note
Use a manual split of training and testing data option will only be enabled if
you add utterances to the testing set in the tag data page. Otherwise, it will be
disabled.

７ Note
Only successfully completed training jobs will generate models.
Training can take some time between a couple of minutes and couple of hours
based on the size of your tagged data.
\n![Image](images/page803_image1.png)
\nSelect the training job ID from the list, a side pane will appear where you can check the
Training progress, Job status, and other details for this job.
To cancel a training job from within Language Studio
, go to the Train model page.
Select the training job you want to cancel, and select Cancel from the top menu.
Model evaluation metrics concepts
How to deploy a model
You can only have one training job running at a time. You cannot start other
training job wihtin the same project until the running job is completed.
Get training job status
Language Studio
Cancel training job
Language Studio
Next steps
\nView orchestration workflow model details
06/21/2025
After model training is completed, you can view your model details and see how well it
performs against the test set. Observing how well your model performed is called evaluation.
The test set consists of data that wasn't introduced to the model during the training process.
Before viewing a model's evaluation, you need:
An orchestration workflow project.
A successfully trained model
See the project development lifecycle for more information.
In the view model details page, you'll be able to see all your models, with their current
training status, and the date they were last trained.
1. Go to your project page in Language Studio
.
2. Select Model performance from the menu on the left side of the screen.
3. In this page you can only view the successfully trained models, F1 score for each
model and model expiration date. You can select the model name for more details
about its performance.
７ Note
Using the Automatically split the testing set from training data option may result in
different model evaluation result every time you train a new model, as the test set is
selected randomly from your utterances. To make sure that the evaluation is calculated on
the same test set every time you train a model, make sure to use the Use a manual split of
training and testing data option when starting a training job and define your Testing set
when add your utterances.
Prerequisites
Model details
Language studio
\n4. You can find the model-level evaluation metrics under Overview, and the intent-level
evaluation metrics. See Evaluation metrics for more information.
5. The confusion matrix for the model is located under Test set confusion matrix. You
can see the confusion matrix for intents.
To load your model data:
1. Select any model in the model evaluation page.
2. Select the Load model data button.
3. Confirm that you do not have any unsaved changes you need to capture in window
that appears, and select Load data.

７ Note
If you don't see any of the intents you have in your model displayed here, it is
because they weren't in any of the utterances that were used for the test set.
Load or export model data
Language studio
\n![Image](images/page806_image1.png)
\n4. Wait until your model data finishes loading back into your project. On completion,
you'll be redirected back to the Schema design page.
To export your model data:
1. Select any model in the model evaluation page.
2. Select the Export model data button. Wait for the JSON snapshot of your model to
be downloaded locally.
To delete your model from within the Language Studio
:
1. Select Model performance from the left side menu.
2. Select the model name you want to delete and select Delete from the top menu.
3. In the window that appears, select OK to delete the model.
As you review how your model performs, learn about the evaluation metrics that are
used.
If you're happy with your model performance, you can deploy your model
Delete model
Language studio
Next steps
\nDeploy an orchestration workflow model
06/21/2025
Once you are satisfied with how your model performs, it's ready to be deployed, and query it
for predictions from utterances. Deploying a model makes it available for use through the
prediction API
.
A successfully created project
Labeled utterances and successfully trained model
See project development lifecycle for more information.
After you have reviewed the model's performance and decide it's fit to be used in your
environment, you need to assign it to a deployment to be able to query it. Assigning the model
to a deployment makes it available for use through the prediction API
. It is recommended to
create a deployment named production  to which you assign the best model you have built so
far and use it in your system. You can create another deployment called staging  to which you
can assign the model you're currently working on to be able to test it. You can have a
maximum on 10 deployments in your project.
To deploy your model from within the Language Studio
:
1. Select Deploying a model from the left side menu.
2. Select Add deployment to start a new deployment job.
Prerequisites
Deploy model
Language Studio
\n3. Select Create new deployment to create a new deployment and assign a trained
model from the dropdown below. You can also Overwrite an existing deployment
by selecting this option and select the trained model you want to assign to it from
the dropdown below.
4. If you're connecting one or more LUIS
 applications or conversational language
understanding
 projects, you have to specify the deployment name.
No configurations are required for custom question answering or unlinked
intents.

７ Note
Overwriting an existing deployment doesn't require changes to your prediction
API
 call, but the results you get will be based on the newly assigned model.

\n![Image](images/page809_image1.png)

![Image](images/page809_image2.png)
\nLUIS projects must be published to the slot configured during the
Orchestration deployment, and custom question answering KBs must also be
published to their Production slots.
5. Select Deploy to submit your deployment job
6. After deployment is successful, an expiration date will appear next to it. Deployment
expiration is when your deployed model will be unavailable to be used for prediction,
which typically happens twelve months after a training configuration expires.
After you are done testing a model assigned to one deployment, you might want to assign it to
another deployment. Swapping deployments involves:
Taking the model assigned to the first deployment, and assigning it to the second
deployment.
taking the model assigned to second deployment and assign it to the first deployment.
This can be used to swap your production  and staging  deployments when you want to take
the model assigned to staging  and assign it to production .
To swap deployments from within Language Studio
1. In the Deploy model page, select the two deployments you want to swap and select
Swap deployments from the top menu.
2. From the window that appears, select the names of the deployments you want to
swap.
To delete a deployment from within Language Studio
, go to the Deploy model page.
Select the deployment you want to delete, and select Delete deployment from the top
menu.
Swap deployments
Language Studio
Delete deployment
Language Studio