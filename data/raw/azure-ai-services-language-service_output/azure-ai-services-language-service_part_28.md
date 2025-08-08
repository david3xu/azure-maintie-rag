1. Navigate to the Azure portal
.
2. Go to your Azure OpenAI resource. (select All resources to locate your resource).
3. Next, select Access Control (IAM) on the left panel, then select Add role assignment.
4. Search and select the Cognitive Services User role, then select Next.
5. Navigate to the Members tab and then select Managed Identity.
Step 1: Assign the correct role to the Azure OpenAI resource
\n![Image](images/page271_image1.png)

![Image](images/page271_image2.png)
\n6. Select Select members, then in the right panel, search for and choose your Azure AI
Foundry resource (the one you're using for this project), and choose Select.
7. Finally, select Review + assign to confirm your selection.
Azure AI Foundry offers a unified platform where you can easily build, manage, and deploy AI
solutions using a wide range of models and tools. Connections enable authentication and
access to both Microsoft and external resources within your Azure AI Foundry projects.
1. Sign into Azure AI Foundry
 using your account and required subscription. Then, select
the project containing your desired Azure AI Foundry resource.
2. Next, navigate to the Management Center in the bottom left corner of the page.
3. Scroll to the Connected resources section of the Management center.
Step 2: Configure connections in AI Foundry
\n![Image](images/page272_image1.png)
\n4. Select the + New connection button.
5. In the new window, select Azure AI Language as the resource type, then find your Azure
AI Language resource.
6. Select Add connection in the corner of your selected Azure AI Language resource.
7. Select Azure OpenAI as the resource type, then find your desired Azure OpenAI resource.
8. Ensure Authentication is set to API key.
9. Select Add connection, then select Close.
\n![Image](images/page273_image1.png)

![Image](images/page273_image2.png)
\n10. Your resources are now set up properly. Continue with setting up the fine-tuning task and
customizing your CLU project.
Create a CLU fine-tuning task
Next Steps
\n![Image](images/page274_image1.png)
\nView conversational language
understanding model details
06/30/2025
After model training is completed, you can view your model details and see how well it
performs against the test set.
Before viewing a model's evaluation, you need:
A successfully created project.
A successfully trained model.
See the project development lifecycle for more information.
1. Go to your project page in Language Studio
.
2. Select Model performance from the menu on the left side of the screen.
3. In this page you can only view the successfully trained models, F1 score of each
model and model expiration date. You can select the model name for more details
about its performance. Models only include evaluation details if there was test data
selected while training the model.
７ Note
Using the Automatically split the testing set from training data option may result in
different model evaluation result every time you train a new model, as the test set is
selected randomly from your utterances. To make sure that the evaluation is calculated on
the same test set every time you train a model, make sure to use the Use a manual split of
training and testing data option when starting a training job and define your Testing set
when add your utterances.
Prerequisites
Model details
Azure AI Foundry
\nIn this tab you can view the model's details such as: F1 score, precision, recall,
date and time for the training job, total training time and number of training and
testing utterances included in this training job. You can view details between
intents or entities by selecting Model Type at the top.
You will also see guidance on how to improve the model. When clicking on view
details a side panel will open to give more guidance on how to improve the
model.
Overview

\n![Image](images/page276_image1.png)
\nTo load your model data:
1. Select any model in the model evaluation page.
2. Select the Load model data button.
3. Confirm that you do not have any unsaved changes you need to capture in window
that appears, and select Load data.
4. Wait until your model data finishes loading back into your project. On completion,
you'll be redirected back to the Schema design page.
To export your model data:

Load or export model data
Azure AI Foundry
\n![Image](images/page277_image1.png)
\n1. Select any model in the model evaluation page.
2. Select the Export model data button. Wait for the JSON snapshot of your model to
be downloaded locally.
To delete your model from within the Language Studio
:
1. Select Model performance from the left side menu.
2. Select the model name you want to delete and select Delete from the top menu.
3. In the window that appears, select OK to delete the model.
As you review your how your model performs, learn about the evaluation metrics that are
used.
If you're happy with your model performance, you can deploy your model
Delete model
Azure AI Foundry
Next steps
\nDeploy a model
06/30/2025
Once you're satisfied with how your model performs, it's ready to be deployed, and query it for
predictions from utterances. Deploying a model makes it available for use through the
prediction API.
A created project
Labeled utterances and successfully trained model
Reviewed the model performance to determine how your model is performing.
See project development lifecycle for more information.
After you have reviewed the model's performance and decide it's fit to be used in your
environment, you need to assign it to a deployment to be able to query it. Assigning the model
to a deployment makes it available for use through the prediction API. It is recommended to
create a deployment named production  to which you assign the best model you have built so
far and use it in your system. You can create another deployment called staging  to which you
can assign the model you're currently working on to be able to test it. You can have a
maximum on 10 deployments in your project.
To deploy your model from within the Language Studio
:
1. Select Deploying a model from the left side menu.
2. Select Add deployment to start the Add deployment wizard.
Prerequisites
Deploy model
Language Studio
\n3. Select Create a new deployment name to create a new deployment and assign a
trained model from the dropdown below. You can otherwise select Overwrite an
existing deployment name to effectively replace the model that's used by an
existing deployment.
4. Select a trained model from the Model dropdown.

７ Note
Overwriting an existing deployment doesn't require changes to your Prediction
API
 call but the results you get will be based on the newly assigned model.

\n![Image](images/page280_image1.png)

![Image](images/page280_image2.png)