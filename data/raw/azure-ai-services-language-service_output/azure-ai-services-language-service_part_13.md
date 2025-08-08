How to train a custom text classification
model
06/30/2025
Training is the process where the model learns from your labeled data. After training is
completed, you will be able to view the model's performance to determine if you need to
improve your model.
To train a model, start a training job. Only successfully completed jobs create a usable model.
Training jobs expire after seven days. After this period, you won't be able to retrieve the job
details. If your training job completed successfully and a model was created, it won't be
affected by the job expiration. You can only have one training job running at a time, and you
can't start other jobs in the same project.
The training times can be anywhere from a few minutes when dealing with few documents, up
to several hours depending on the dataset size and the complexity of your schema.
Before you train your model, you need:
A successfully created project with a configured Azure blob storage account,
Text data that has been uploaded to your storage account.
Labeled data
See the project development lifecycle for more information.
Before you start the training process, labeled documents in your project are divided into a
training set and a testing set. Each one of them serves a different function. The training set is
used in training the model, this is the set from which the model learns the class/classes
assigned to each document. The testing set is a blind set that is not introduced to the model
during training but only during evaluation. After the model is trained successfully, it is used to
make predictions from the documents in the testing set. Based on these predictions, the
model's evaluation metrics will be calculated. It is recommended to make sure that all your
classes are adequately represented in both the training and testing set.
Custom text classification supports two methods for data splitting:
Prerequisites
Data splitting
\nAutomatically splitting the testing set from training data: The system will split your
labeled data between the training and testing sets, according to the percentages you
choose. The system will attempt to have a representation of all classes in your training set.
The recommended percentage split is 80% for training and 20% for testing.
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
more information on data splitting.
5. Select the Train button.
6. If you select the training job ID from the list, a side pane will appear where you can
check the Training progress, Job status, and other details for this job.

７ Note
Only successfully completed training jobs will generate models.
The time to train the model can take anywhere between a few minutes to
several hours based on the size of your labeled data.
You can only have one training job running at a time. You can't start other
training job within the same project until the running job is completed.
Cancel training job
Language Studio
\n![Image](images/page123_image1.png)
\nTo cancel a training job in Language Studio
, go to the Training jobs page. Select the
training job you want to cancel, and select Cancel from the top menu.
After training is completed, you will be able to view the model's performance to optionally
improve your model if needed. Once you're satisfied with your model, you can deploy it,
making it available to use for classifying text.
Next steps
\nView your text classification model's
evaluation and details
06/30/2025
After your model has finished training, you can view the model performance and see the
predicted classes for the documents in the test set.
Before viewing model evaluation you need:
A custom text classification project with a configured Azure blob storage account.
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
\nIn this tab you can view the model's details such as: F1 score, precision, recall,
date and time for the training job, total training time and number of training and
testing documents included in this training job.
You will also see guidance on how to improve the model. When clicking on view
details a side panel will open to give more guidance on how to improve the
model. In this example, there are not enough data in training set for these
classes. Also, there is unclear distinction between class types in training set,
where two classes are confused with each other. By clicking on the confused
classes, you will be taken to the data labeling page to label more data with the
correct class.
７ Note
Classes that are neither labeled nor predicted in the test set will not be part of the
displayed results.
Overview

\n![Image](images/page126_image1.png)
\nLearn more about model guidance and confusion matrix in model performance
concepts.
To load your model data:
1. Select any model in the model evaluation page.
2. Select the Load model data button.
3. Confirm that you do not have any unsaved changes you need to capture in window
that appears, and select Load data.
4. Wait until your model data finishes loading back into your project. On completion,
you'll be redirected back to the Schema design page.
To export your model data:
1. Select any model in the model evaluation page.
2. Select the Export model data button. Wait for the JSON snapshot of your model to
be downloaded locally.

Load or export model data
Language studio
Delete model
\n![Image](images/page127_image1.png)
\nTo delete your model from within the Language Studio
:
1. Select Model performance from the left side menu.
2. Select the model name you want to delete and select Delete from the top menu.
3. In the window that appears, select OK to delete the model.
As you review your how your model performs, learn about the evaluation metrics that are used.
Once you know whether your model performance needs to improve, you can begin improving
the model.
Language studio
Next steps
\nDeploy a model and classify text using the
runtime API
06/30/2025
Once you are satisfied with how your model performs, it is ready to be deployed; and use it to
classify text. Deploying a model makes it available for use through the prediction API
.
A custom text classification project with a configured Azure storage account,
Text data that has been uploaded to your storage account.
Labeled data and successfully trained model
Reviewed the model evaluation details to determine how your model is performing.
See the project development lifecycle for more information.
After you have reviewed your model's performance and decided it can be used in your
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

７ Note
Overwriting an existing deployment doesn't require changes to your Prediction
API
 call but the results you get will be based on the newly assigned model.
\n![Image](images/page130_image1.png)