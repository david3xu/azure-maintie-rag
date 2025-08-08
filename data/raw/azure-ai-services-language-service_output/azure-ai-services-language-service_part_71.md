In this tab you can view the model's details such as: F1 score, precision, recall,
date and time for the training job, total training time and number of training and
testing documents included in this training job.
You will also see guidance on how to improve the model. When clicking on view
details a side panel will open to give more guidance on how to improve the
model. In this example, BorrowerAddress and BorrowerName entities are
confused with $none entity. By clicking on the confused entities, you will be
taken to the data labeling page to label more data with the correct entity.
７ Note
Entities that are neither labeled nor predicted in the test set will not be part of the
displayed results.
Overview


\n![Image](images/page701_image1.png)

![Image](images/page701_image2.png)
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
To delete your model from within the Language Studio
:
1. Select Model performance from the left side menu.
2. Select the model name you want to delete and select Delete from the top menu.
3. In the window that appears, select OK to delete the model.
Load or export model data
Language studio
Delete model
Language studio
\nDeploy your model
Learn about the metrics used in evaluation.
Next steps
\nDeploy a model and extract entities from
text using the runtime API
06/30/2025
Once you are satisfied with how your model performs, it is ready to be deployed and used to
recognize entities in text. Deploying a model makes it available for use through the prediction
API
.
A successfully created project with a configured Azure storage account.
Text data that has been uploaded to your storage account.
Labeled data and successfully trained model
Reviewed the model evaluation details to determine how your model is performing.
See project development lifecycle for more information.
After you've reviewed your model's performance and decided it can be used in your
environment, you need to assign it to a deployment. Assigning the model to a deployment
makes it available for use through the prediction API
. It is recommended to create a
deployment named production to which you assign the best model you have built so far and
use it in your system. You can create another deployment called staging to which you can
assign the model you're currently working on to be able to test it. You can have a maximum of
10 deployments in your project.
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
Overwriting an existing deployment doesn't require changes to your prediction
API
 call but the results you get will be based on the newly assigned model.
\n![Image](images/page705_image1.png)
\n4. Select Deploy to start the deployment job.
5. After deployment is successful, an expiration date will appear next to it. Deployment
expiration is when your deployed model will be unavailable to be used for prediction,
which typically happens twelve months after a training configuration expires.
After you are done testing a model assigned to one deployment and you want to assign this
model to another deployment you can swap these two deployments. Swapping deployments
involves taking the model assigned to the first deployment, and assigning it to the second
deployment. Then taking the model assigned to second deployment, and assigning it to the
first deployment. You can use this process to swap your production and staging deployments
when you want to take the model assigned to staging and assign it to production.
To swap deployments from within Language Studio
:
1. In the Deploying a model page, select the two deployments you want to swap and
select Swap deployments from the top menu.

Swap deployments
Language Studio
\n![Image](images/page706_image1.png)
\n2. From the window that appears, select the names of the deployments you want to
swap.
To delete a deployment from within Language Studio
, go to the Deploying a model
page. Select the deployment you want to delete and select Delete deployment from the
top menu.
You can deploy your project to multiple regions by assigning different Language resources that
exist in different regions.
To assign deployment resources in other regions in Language Studio
:
1. Make sure you've assigned yourself as a Cognitive Services Language Owner
 to the
resource you used to create the project.
2. Go to the Deploying a model page in Language Studio.
3. Select the Regions tab.
4. Select Add deployment resource.
5. Select a Language resource in another region.
You are now ready to deploy your project to the regions where you have assigned
resources.
When unassigning or removing a deployment resource from a project, you will also delete all
the deployments that have been deployed to that resource's region.
To unassign or remove deployment resources in other regions using Language Studio
:
Delete deployment
Language Studio
Assign deployment resources
Language Studio
Unassign deployment resources
Language Studio
\n1. Go to the Regions tab in the Deploy a model page.
2. Select the resource you'd like to unassign.
3. Select the Remove assignment button.
4. In the window that appears, type the name of the resource you want to remove.
After you have a deployment, you can use it to extract entities from text.
Next steps
\nQuery your custom model
06/30/2025
After the deployment is added successfully, you can query the deployment to extract entities
from your text based on the model you assigned to the deployment. You can query the
deployment programmatically using the Prediction API or through the client libraries (Azure
SDK).
You can use Language Studio to submit the custom entity recognition task and visualize the
results.
To test your deployed models from within the Language Studio
:
1. Select Testing deployments from the left side menu.
2. Select the deployment you want to test. You can only test models that are assigned to
deployments.
3. For multilingual projects, from the language dropdown, select the language of the text
you are testing.
4. Select the deployment you want to query/test from the dropdown.
5. You can enter the text you want to submit to the request or upload a .txt  file to use.
6. Select Run the test from the top menu.
7. In the Result tab, you can see the extracted entities from your text and their types. You
can also view the JSON response under the JSON tab.
Test deployed model
\n1. After the deployment job is completed successfully, select the deployment you want
to use and from the top menu select Get prediction URL.

Send an entity recognition request to your model
Language Studio

\n![Image](images/page710_image1.png)

![Image](images/page710_image2.png)