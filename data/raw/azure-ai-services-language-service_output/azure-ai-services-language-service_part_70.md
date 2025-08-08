Before you connect to Azure Machine Learning, you need an Azure Machine Learning account
with a pricing plan that can accommodate the compute needs of your project. See the
prerequisites section to make sure that you have successfully completed all the requirements
to start connecting your Language Studio project to Azure Machine Learning.
1. Use the Azure portal
 to navigate to the Azure Blob Storage account connected to your
language resource.
2. Ensure that the Storage Blob Data Contributor role is assigned to your AML workspace
within the role assignments for your Azure Blob Storage account.
3. Navigate to your project in Language Studio
. From the left pane of your project, select
Data labeling.
4. Select use Azure Machine Learning to label in either the Data labeling description, or
under the Activity pane.
5. Select Connect to Azure Machine Learning to start the connection process.
Connect to Azure Machine Learning

\n![Image](images/page691_image1.png)
\n6. In the window that appears, follow the prompts. Select the Azure Machine Learning
workspace you’ve created previously under the same Azure subscription. Enter a name for
the new Azure Machine Learning project that will be created to enable labeling in Azure
Machine Learning.
7. (Optional) Turn on the vendor labeling toggle to use labeling vendor companies. Before
choosing the vendor labeling companies, contact the vendor labeling companies on the
Azure Marketplace
 to finalize a contract with them. For more information about
working with vendor companies, see How to outsource data labeling.
You can also leave labeling instructions for the human labelers that will help you in the
labeling process. These instructions can help them understand the task by leaving clear
definitions of the labels and including examples for better results.
8. Review the settings for your connection to Azure Machine Learning and make changes if
needed.

 Tip
Make sure your workspace is linked to the same Azure Blob Storage account and
Language resource before continuing. You can create a new workspace and link to
your storage account using the Azure portal
. Ensure that the storage account is
properly linked to the workspace.
） Important
Finalizing the connection is permanent. Attempting to disconnect your established
connection at any point in time will permanently disable your Language Studio
project from connecting to the same Azure Machine Learning project.
\n![Image](images/page692_image1.png)
\n9. After the connection has been initiated, your ability to label data in Language Studio will
be disabled for a few minutes to prepare the new connection.
Once the connection has been established, you can switch to Azure Machine Learning through
the Activity pane in Language Studio at any time.
When you switch, your ability to label data in Language Studio will be disabled, and you will be
able to label data in Azure Machine Learning. You can switch back to labeling in Language
Studio at any time through Azure Machine Learning.
For information on how to label the text, see Azure Machine Learning how to label. For
information about managing and tracking the text labeling project, see Azure Machine
Learning set up and manage a text labeling project.
When you switch to labeling using Azure Machine Learning, you can still train, evaluate, and
deploy your model in Language Studio. To train your model using updated labels from Azure
Machine Learning:
Switch to labeling with Azure Machine Learning
from Language Studio

Train your model using labels from Azure Machine
Learning
\n![Image](images/page693_image1.png)
\n1. Select Training jobs from the navigation menu on the left of the Language studio screen
for your project.
2. Select Import latest labels from Azure Machine Learning from the Choose label origin
section in the training page. This synchronizes the labels from Azure Machine Learning
before starting the training job.
After you've switched to labeling with Azure Machine Learning, You can switch back to labeling
with Language Studio project at any time.
To switch back to labeling with Language Studio:
1. Navigate to your project in Azure Machine Learning and select Data labeling from the left
pane.

Switch to labeling with Language Studio from
Azure Machine Learning
７ Note
Only users with the correct roles in Azure Machine Learning have the ability to
switch labeling.
When you switch to using Language Studio, labeling on Azure Machine learning will
be disabled.
\n![Image](images/page694_image1.png)
\n2. Select the Language Studio tab and select Switch to Language Studio.
3. The process takes a few minutes to complete, and your ability to label in Azure Machine
Learning will be disabled until it's switched back from Language Studio.
Disconnecting your project from Azure Machine Learning is a permanent, irreversible process
and can't be undone. You will no longer be able to access your labels in Azure Machine
Learning, and you won’t be able to reconnect the Azure Machine Learning project to any
Language Studio project in the future. To disconnect from Azure Machine Learning:
1. Ensure that any updated labels you want to maintain are synchronized with Azure
Machine Learning by switching the labeling experience back to the Language Studio.
2. Select Project settings from the navigation menu on the left in Language Studio.
3. Select the Disconnect from Azure Machine Learning button from the Manage Azure
Machine Learning connections section.
Learn more about labeling your data for Custom Text Classification and Custom Named Entity
Recognition.

Disconnecting from Azure Machine Learning
Next steps
\n![Image](images/page695_image1.png)
\nTrain your custom named entity
recognition model
06/30/2025
Training is the process where the model learns from your labeled data. After training is
completed, you'll be able to view the model's performance to determine if you need to
improve your model.
To train a model, you start a training job and only successfully completed jobs create a model.
Training jobs expire after seven days, which means you won't be able to retrieve the job details
after this time. If your training job completed successfully and a model was created, the model
won't be affected. You can only have one training job running at a time, and you can't start
other jobs in the same project.
The training times can be anywhere from a few minutes when dealing with few documents, up
to several hours depending on the dataset size and the complexity of your schema.
A successfully created project with a configured Azure blob storage account
Text data that has been uploaded to your storage account.
Labeled data
See the project development lifecycle for more information.
Before you start the training process, labeled documents in your project are divided into a
training set and a testing set. Each one of them serves a different function. The training set is
used in training the model, this is the set from which the model learns the labeled entities and
what spans of text are to be extracted as entities. The testing set is a blind set that is not
introduced to the model during training but only during evaluation. After model training is
completed successfully, the model is used to make predictions from the documents in the
testing and based on these predictions evaluation metrics are calculated. It's recommended to
make sure that all your entities are adequately represented in both the training and testing set.
Custom NER supports two methods for data splitting:
Automatically splitting the testing set from training data:The system will split your
labeled data between the training and testing sets, according to the percentages you
Prerequisites
Data splitting
\nchoose. The recommended percentage split is 80% for training and 20% for testing.
Use a manual split of training and testing data: This method enables users to define
which labeled documents should belong to which set. This step is only enabled if you
have added documents to your testing set during data labeling.
To start training your model from within the Language Studio
:
1. Select Training jobs from the left side menu.
2. Select Start a training job from the top menu.
3. Select Train a new model and type in the model name in the text box. You can also
overwrite an existing model by selecting this option and choosing the model you
want to overwrite from the dropdown menu. Overwriting a trained model is
irreversible, but it won't affect your deployed models until you deploy the new
model.
７ Note
If you choose the Automatically splitting the testing set from training data option, only
the data assigned to training set will be split according to the percentages provided.
Train model
Language studio
\n4. Select data splitting method. You can choose Automatically splitting the testing set
from training data where the system will split your labeled data between the training
and testing sets, according to the specified percentages. Or you can Use a manual
split of training and testing data, this option is only enabled if you have added
documents to your testing set during data labeling. See How to train a model for
information about data splitting.
5. Select the Train button.
6. If you select the Training Job ID from the list, a side pane will appear where you can
check the Training progress, Job status, and other details for this job.

７ Note
Only successfully completed training jobs will generate models.
Training can take some time between a couple of minutes and several
hours based on the size of your labeled data.
You can only have one training job running at a time. You can't start other
training job within the same project until the running job is completed.
\n![Image](images/page698_image1.png)
\nTo cancel a training job from within Language Studio
, go to the Training jobs page.
Select the training job you want to cancel and select Cancel from the top menu.
After training is completed, you'll be able to view model performance to optionally improve
your model if needed. Once you're satisfied with your model, you can deploy it, making it
available to use for extracting entities from text.
Cancel training job
Language Studio
Next steps
\nView the custom NER model's evaluation
and details
06/30/2025
After your model has finished training, you can view the model performance and see the
extracted entities for the documents in the test set.
Before viewing model evaluation, you need:
A successfully created project with a configured Azure blob storage account.
Text data that has been uploaded to your storage account.
Labeled data
A successfully trained model
See the project development lifecycle for more information.
1. Go to your project page in Language Studio
.
2. Select Model performance from the menu on the left side of the screen.
3. In this page you can only view the successfully trained models, F1 score for each
model and model expiration date. You can select the model name for more details
about its performance.
７ Note
Using the Automatically split the testing set from training data option may result in
different model evaluation result every time you train a new model, as the test set is
selected randomly from the data. To make sure that the evaluation is calculated on the
same test set every time you train a model, make sure to use the Use a manual split of
training and testing data option when starting a training job and define your Test
documents when labeling data.
Prerequisites
Model details
Language studio