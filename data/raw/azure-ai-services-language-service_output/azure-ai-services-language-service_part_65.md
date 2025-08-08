1. Sign in to the Azure portal
 to create a new Azure AI Language resource.
2. In the window that appears, select Custom text classification & custom named
entity recognition from the custom features. Select Continue to create your
resource at the bottom of the screen.
3. Create a Language resource with following details.
Name
Description
Subscription
Your Azure subscription.
Resource
group
A resource group that will contain your resource. You can use an existing
one, or create a new one.
Region
The region for your Language resource. For example, "West US 2".
Name
A name for your resource.
Pricing tier
The pricing tier for your Language resource. You can use the Free (F0) tier
to try the service.
If you have a pre-existing resource that you'd like to use, you will need to connect it
to storage account. See guidance to using a pre-existing resource for information.
Create a new resource from the Azure portal

ﾉ
Expand table
\n![Image](images/page641_image1.png)
\n4. In the Custom text classification & custom named entity recognition section,
select an existing storage account or select New storage account. These values are
to help you get started, and not necessarily the storage account values you’ll want
to use in production environments. To avoid latency during building your project
connect to storage accounts in the same region as your Language resource.
Storage account value
Recommended value
Storage account name
Any name
Storage account type
Standard LRS
5. Make sure the Responsible AI Notice is checked. Select Review + create at the
bottom of the page, then select Create.
After you have created an Azure storage account and connected it to your Language
resource, you will need to upload the documents from the sample dataset to the root
directory of your container. These documents will later be used to train your model.
1. Download the sample dataset
 from GitHub.
2. Open the .zip file, and extract the folder containing the documents.
3. In the Azure portal
, navigate to the storage account you created, and select it.
4. In your storage account, select Containers from the left menu, located below Data
storage. On the screen that appears, select + Container. Give the container the
name example-data and leave the default Public access level.
７ Note
If you get a message saying "your login account is not an owner of the selected
storage account's resource group", your account needs to have an owner role
assigned on the resource group before you can create a Language resource.
Contact your Azure subscription owner for assistance.
ﾉ
Expand table
Upload sample data to blob container
\n5. After your container has been created, select it. Then select Upload button to
select the .txt  and .json  files you downloaded earlier.
The provided sample dataset contains 20 loan agreements. Each agreement includes
two parties: a lender and a borrower. You can use the provided sample file to extract
relevant information for: both parties, an agreement date, a loan amount, and an
interest rate.
Once your resource and storage account are configured, create a new custom NER
project. A project is a work area for building your custom ML models based on your
data. Your project can only be accessed by you and others who have access to the
Language resource being used.
1. Sign into the Language Studio
. A window will appear to let you select your
subscription and Language resource. Select the Language resource you created in
the above step.
2. Under the Extract information section of Language Studio, select Custom named
entity recognition.


Create a custom named entity recognition
project
\n![Image](images/page643_image1.png)

![Image](images/page643_image2.png)
\n3. Select Create new project from the top menu in your projects page. Creating a
project will let you tag data, train, evaluate, improve, and deploy your models.
4. After you click, Create new project, a window will appear to let you connect your
storage account. If you've already connected a storage account, you will see the
storage accounted connected. If not, choose your storage account from the
dropdown that appears and select Connect storage account; this will set the
required roles for your storage account. This step will possibly return an error if
you are not assigned as owner on the storage account.


７ Note
You only need to do this step once for each new resource you use.
This process is irreversible, if you connect a storage account to your
Language resource you cannot disconnect it later.
You can only connect your Language resource to one storage account.
\n![Image](images/page644_image1.png)

![Image](images/page644_image2.png)
\n5. Enter the project information, including a name, description, and the language of
the files in your project. If you're using the example dataset
, select English. You
won’t be able to change the name of your project later. Select Next
6. Select the container where you have uploaded your dataset. If you have already
labeled data make sure it follows the supported format and select Yes, my files are
already labeled and I have formatted JSON labels file and select the labels file
from the drop-down menu. Select Next.
7. Review the data you entered and select Create Project.
Typically after you create a project, you go ahead and start tagging the documents you
have in the container connected to your project. For this quickstart, you have imported a
sample tagged dataset and initialized your project with the sample JSON tags file.

 Tip
Your dataset doesn't have to be entirely in the same language. You can have
multiple documents, each with different supported languages. If your dataset
contains documents of different languages or if you expect text from different
languages during runtime, select enable multi-lingual dataset option when
you enter the basic information for your project. This option can be enabled
later from the Project settings page.
Train your model
\n![Image](images/page645_image1.png)
\nTo start training your model from within the Language Studio
:
1. Select Training jobs from the left side menu.
2. Select Start a training job from the top menu.
3. Select Train a new model and type in the model name in the text box. You can also
overwrite an existing model by selecting this option and choosing the model you
want to overwrite from the dropdown menu. Overwriting a trained model is
irreversible, but it won't affect your deployed models until you deploy the new
model.
4. Select data splitting method. You can choose Automatically splitting the testing
set from training data where the system will split your labeled data between the
training and testing sets, according to the specified percentages. Or you can Use a
manual split of training and testing data, this option is only enabled if you have
added documents to your testing set during data labeling. See How to train a
model for information about data splitting.
5. Select the Train button.
6. If you select the Training Job ID from the list, a side pane will appear where you
can check the Training progress, Job status, and other details for this job.

\n![Image](images/page646_image1.png)
\nGenerally after training a model you would review its evaluation details and make
improvements if necessary. In this quickstart, you will just deploy your model, and make
it available for you to try in Language studio, or you can call the prediction API
.
To deploy your model from within the Language Studio
:
1. Select Deploying a model from the left side menu.
2. Select Add deployment to start a new deployment job.
７ Note
Only successfully completed training jobs will generate models.
Training can take some time between a couple of minutes and several
hours based on the size of your labeled data.
You can only have one training job running at a time. You can't start
other training job within the same project until the running job is
completed.
Deploy your model

\n![Image](images/page647_image1.png)
\n3. Select Create new deployment to create a new deployment and assign a trained
model from the dropdown below. You can also Overwrite an existing deployment
by selecting this option and select the trained model you want to assign to it from
the dropdown below.
4. Select Deploy to start the deployment job.
5. After deployment is successful, an expiration date will appear next to it.
Deployment expiration is when your deployed model will be unavailable to be
used for prediction, which typically happens twelve months after a training
configuration expires.
After your model is deployed, you can start using it to extract entities from your text via
Prediction API
. For this quickstart, you will use the Language Studio
 to submit the
custom entity recognition task and visualize the results. In the sample dataset you
downloaded earlier, you can find some test documents that you can use in this step.
７ Note
Overwriting an existing deployment doesn't require changes to your
prediction API
 call but the results you get will be based on the newly
assigned model.

Test your model
\n![Image](images/page648_image1.png)
\nTo test your deployed models from within the Language Studio
:
1. Select Testing deployments from the left side menu.
2. Select the deployment you want to test. You can only test models that are assigned
to deployments.
3. For multilingual projects, from the language dropdown, select the language of the
text you are testing.
4. Select the deployment you want to query/test from the dropdown.
5. You can enter the text you want to submit to the request or upload a .txt  file to
use.
6. Select Run the test from the top menu.
7. In the Result tab, you can see the extracted entities from your text and their types.
You can also view the JSON response under the JSON tab.

Clean up resources
\n![Image](images/page649_image1.png)
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
When you don't need your project anymore, you can delete your project using
Language Studio
. Select Custom named entity recognition (NER) from the top, select
the project you want to delete, and then select Delete from the top menu.
After you've created entity extraction model, you can:
Use the runtime API to extract entities
When you start to create your own custom NER projects, use the how-to articles to learn
more about tagging, training and consuming your model in greater detail:
Data selection and schema design
Tag data
Train a model
Model evaluation
Next steps
Yes
No