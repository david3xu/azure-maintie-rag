1. Go to your storage account page in the Azure portal
.
2. Select Access Control (IAM) in the left pane.
3. Select Add to Add Role Assignments, and choose the Storage blob data contributor role
on the storage account.
4. Within Assign access to, select Managed identity.
5. Select Select members
6. Select your subscription, and Language as the managed identity. You can search for user
names in the Select field.
Make sure to allow (GET, PUT, DELETE) methods when enabling Cross-Origin Resource Sharing
(CORS). Set allowed origins field to https://language.cognitive.azure.com . Allow all header by
adding *  to the allowed header values, and set the maximum age to 500 .
Roles for your storage account
） Important
If you have a virtual network or private endpoint, be sure to select Allow Azure services
on the trusted services list to access this storage account in the Azure portal.
Enable CORS for your storage account

Create a custom text classification project
\n![Image](images/page91_image1.png)
\nOnce your resource and storage container are configured, create a new custom text
classification project. A project is a work area for building your custom AI models based on
your data. Your project can only be accessed by you and others who have access to the Azure
resource being used. If you have labeled data, you can import it to get started.
1. Sign into the Language Studio
. A window will appear to let you select your
subscription and Language resource. Select your Language resource.
2. Under the Classify text section of Language Studio, select Custom text classification.
3. Select Create new project from the top menu in your projects page. Creating a
project will let you label data, train, evaluate, improve, and deploy your models.
4. After you click, Create new project, a window will appear to let you connect your
storage account. If you've already connected a storage account, you will see the
Language Studio


\n![Image](images/page92_image1.png)

![Image](images/page92_image2.png)
\nstorage accounted connected. If not, choose your storage account from the
dropdown that appears and select Connect storage account; this will set the
required roles for your storage account. This step will possibly return an error if you
are not assigned as owner on the storage account.
5. Select project type. You can either create a Multi label classification project where
each document can belong to one or more classes or Single label classification
project where each document can belong to only one class. The selected type can't
be changed later. Learn more about project types
７ Note
You only need to do this step once for each new language resource you
use.
This process is irreversible, if you connect a storage account to your
Language resource you cannot disconnect it later.
You can only connect your Language resource to one storage account.

\n![Image](images/page93_image1.png)
\n6. Enter the project information, including a name, description, and the language of the
documents in your project. If you're using the example dataset
, select English. You
won’t be able to change the name of your project later. Select Next.
7. Select the container where you have uploaded your dataset.

 Tip
Your dataset doesn't have to be entirely in the same language. You can have
multiple documents, each with different supported languages. If your dataset
contains documents of different languages or if you expect text from different
languages during runtime, select enable multi-lingual dataset option when you
enter the basic information for your project. This option can be enabled later
from the Project settings page.
７ Note
If you have already labeled your data make sure it follows the supported format
and select Yes, my documents are already labeled and I have formatted JSON
labels file and select the labels file from the drop-down menu below.
\n![Image](images/page94_image1.png)
\nIf you’re using one of the example datasets, use the included
webOfScience_labelsFile  or movieLabels  json file. Then select Next.
8. Review the data you entered and select Create Project.
If you have already labeled data, you can use it to get started with the service. Make sure that
your labeled data follows the accepted data formats.
1. Sign into the Language Studio
. A window will appear to let you select your
subscription and Language resource. Select your Language resource.
2. Under the Classify text section of Language Studio, select Custom text classification.
3. Select Create new project from the top menu in your projects page. Creating a
project will let you label data, train, evaluate, improve, and deploy your models.
Import a custom text classification project
Language Studio

\n![Image](images/page95_image1.png)
\n4. After you select Create new project, a screen will appear to let you connect your
storage account. If you can’t find your storage account, make sure you created a
resource using the recommended steps. If you've already connected a storage
account to your Language resource, you will see your storage account connected.

７ Note
You only need to do this step once for each new language resource you
use.
This process is irreversible, if you connect a storage account to your
Language resource you cannot disconnect it later.
You can only connect your Language resource to one storage account.
\n![Image](images/page96_image1.png)
\n5. Select project type. You can either create a Multi label classification project where
each document can belong to one or more classes or Single label classification
project where each document can belong to only one class. The selected type can't
be changed later.
6. Enter the project information, including a name, description, and the language of the
documents in your project. You won’t be able to change the name of your project


\n![Image](images/page97_image1.png)

![Image](images/page97_image2.png)
\nlater. Select Next.
7. Select the container where you have uploaded your dataset.
8. Select Yes, my documents are already labeled and I have formatted JSON labels file
and select the labels file from the drop-down menu below to import your JSON
labels file. Make sure it follows the supported format.
9. Select Next.
10. Review the data you entered and select Create Project.
1. Go to your project settings page in Language Studio
.
2. You can see project details.
3. In this page, you can update project description and enable/disable Multi-lingual
dataset in project settings.
4. You can also view the connected storage account and container to your Language
resource.
5. You can also retrieve your resource primary key from this page.
 Tip
Your dataset doesn't have to be entirely in the same language. You can have
multiple documents, each with different supported languages. If your dataset
contains documents of different languages or if you expect text from different
languages during runtime, select enable multi-lingual dataset option when you
enter the basic information for your project. This option can be enabled later
from the Project settings page.
Get project details
Language Studio
\nWhen you don't need your project anymore, you can delete your project using Language
Studio
. Select Custom text classification in the top, and then select the project you want
to delete. Select Delete from the top menu to delete the project.

Delete project
Language Studio
\n![Image](images/page99_image1.png)
\nYou should have an idea of the project schema you will use to label your data.
After your project is created, you can start labeling your data, which will inform your text
classification model how to interpret text, and is used for training and evaluation.
Next steps