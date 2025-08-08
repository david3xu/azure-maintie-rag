"Mystery"
"Drama"
"Thriller"
"Comedy"
"Action"
2. In the Azure portal
, navigate to the storage account you created, and select it. You can
do this by clicking Storage accounts and typing your storage account name into Filter for
any field.
if your resource group does not show up, make sure the Subscription equals filter is set
to All.
3. In your storage account, select Containers from the left menu, located below Data
storage. On the screen that appears, select + Container. Give the container the name
example-data and leave the default Public access level.
4. After your container has been created, select it. Then select Upload button to select the
.txt  and .json  files you downloaded earlier.


Create a custom text classification project
\n![Image](images/page51_image1.png)

![Image](images/page51_image2.png)
\nOnce your resource and storage container are configured, create a new custom text
classification project. A project is a work area for building your custom ML models based on
your data. Your project can only be accessed by you and others who have access to the
Language resource being used.
1. Sign into the Language Studio
. A window will appear to let you select your subscription
and Language resource. Select your Language resource.
2. Under the Classify text section of Language Studio, select Custom text classification.
3. Select Create new project from the top menu in your projects page. Creating a project
will let you label data, train, evaluate, improve, and deploy your models.
4. After you click, Create new project, a window will appear to let you connect your storage
account. If you've already connected a storage account, you will see the storage
accounted connected. If not, choose your storage account from the dropdown that
appears and select Connect storage account; this will set the required roles for your


\n![Image](images/page52_image1.png)

![Image](images/page52_image2.png)
\nstorage account. This step will possibly return an error if you are not assigned as owner
on the storage account.
5. Select project type. You can either create a Multi label classification project where each
document can belong to one or more classes or Single label classification project where
each document can belong to only one class. The selected type can't be changed later.
Learn more about project types
７ Note
You only need to do this step once for each new language resource you use.
This process is irreversible, if you connect a storage account to your Language
resource you cannot disconnect it later.
You can only connect your Language resource to one storage account.

\n![Image](images/page53_image1.png)
\n6. Enter the project information, including a name, description, and the language of the
documents in your project. If you're using the example dataset
, select English. You
won’t be able to change the name of your project later. Select Next.
7. Select the container where you have uploaded your dataset.

 Tip
Your dataset doesn't have to be entirely in the same language. You can have multiple
documents, each with different supported languages. If your dataset contains
documents of different languages or if you expect text from different languages
during runtime, select enable multi-lingual dataset option when you enter the basic
information for your project. This option can be enabled later from the Project
settings page.
７ Note
If you have already labeled your data make sure it follows the supported format and
select Yes, my documents are already labeled and I have formatted JSON labels file
and select the labels file from the drop-down menu below.
\n![Image](images/page54_image1.png)
\nIf you’re using one of the example datasets, use the included
webOfScience_labelsFile  or movieLabels  json file. Then select Next.
8. Review the data you entered and select Create Project.
Typically after you create a project, you go ahead and start labeling the documents you have in
the container connected to your project. For this quickstart, you have imported a sample
labeled dataset and initialized your project with the sample JSON labels file.
To start training your model from within the Language Studio
:
1. Select Training jobs from the left side menu.
2. Select Start a training job from the top menu.
3. Select Train a new model and type in the model name in the text box. You can also
overwrite an existing model by selecting this option and choosing the model you want
to overwrite from the dropdown menu. Overwriting a trained model is irreversible, but it
won't affect your deployed models until you deploy the new model.
4. Select data splitting method. You can choose Automatically splitting the testing set from
training data where the system will split your labeled data between the training and
testing sets, according to the specified percentages. Or you can Use a manual split of
training and testing data, this option is only enabled if you have added documents to
Train your model

\n![Image](images/page55_image1.png)
\nyour testing set during data labeling. See How to train a model for more information on
data splitting.
5. Select the Train button.
6. If you select the training job ID from the list, a side pane will appear where you can check
the Training progress, Job status, and other details for this job.
Generally after training a model you would review its evaluation details and make
improvements if necessary. In this quickstart, you will just deploy your model, and make it
available for you to try in Language Studio, or you can call the prediction API
.
To deploy your model from within the Language Studio
:
1. Select Deploying a model from the left side menu.
2. Select Add deployment to start a new deployment job.
７ Note
Only successfully completed training jobs will generate models.
The time to train the model can take anywhere between a few minutes to
several hours based on the size of your labeled data.
You can only have one training job running at a time. You can't start other
training job within the same project until the running job is completed.
Deploy your model
\n3. Select Create new deployment to create a new deployment and assign a trained model
from the dropdown below. You can also Overwrite an existing deployment by selecting
this option and select the trained model you want to assign to it from the dropdown
below.

７ Note
Overwriting an existing deployment doesn't require changes to your Prediction
API
 call but the results you get will be based on the newly assigned model.
\n![Image](images/page57_image1.png)
\n4. select Deploy to start the deployment job.
5. After deployment is successful, an expiration date will appear next to it. Deployment
expiration is when your deployed model will be unavailable to be used for prediction,
which typically happens twelve months after a training configuration expires.
After your model is deployed, you can start using it to classify your text via Prediction API
.
For this quickstart, you will use the Language Studio
 to submit the custom text classification
task and visualize the results. In the sample dataset you downloaded earlier you can find some
test documents that you can use in this step.
To test your deployed models within Language Studio
:
1. Select Testing deployments from the menu on the left side of the screen.

Test your model
\n![Image](images/page58_image1.png)
\n2. Select the deployment you want to test. You can only test models that are assigned to
deployments.
3. For multilingual projects, select the language of the text you're testing using the language
dropdown.
4. Select the deployment you want to query/test from the dropdown.
5. Enter the text you want to submit in the request, or upload a .txt  document to use. If
you’re using one of the example datasets, you can use one of the included .txt files.
6. Select Run the test from the top menu.
7. In the Result tab, you can see the predicted classes for your text. You can also view the
JSON response under the JSON tab. The following example is for a single label
classification project. A multi label classification project can return more than one class in
the result.
When you don't need your project anymore, you can delete your project using Language
Studio
. Select Custom text classification in the top, and then select the project you want to

Clean up projects
\n![Image](images/page59_image1.png)
\ndelete. Select Delete from the top menu to delete the project.
After you've created a custom text classification model, you can:
Use the runtime API to classify text
When you start to create your own custom text classification projects, use the how-to articles
to learn more about developing your model in greater detail:
Data selection and schema design
Tag data
Train a model
View model evaluation
Next steps