1. Sign into the Language Studio
. A window will appear to let you select your
subscription and Language resource. Select your Language resource.
2. Under the Extract information section of Language Studio, select Custom named
entity recognition.
3. Select Create new project from the top menu in your projects page. Creating a
project will let you tag data, train, evaluate, improve, and deploy your models.
4. After you select Create new project, a screen will appear to let you connect your
storage account. If you can’t find your storage account, make sure you created a
resource using the recommended steps. If you've already connected a storage
account to your Language resource, you will see your storage account connected.
Language Studio


７ Note
\n![Image](images/page671_image1.png)

![Image](images/page671_image2.png)
\n5. Enter the project information, including a name, description, and the language of the
files in your project. You won’t be able to change the name of your project later.
Select Next.
6. Select the container where you have uploaded your dataset.
7. Select Yes, my files are already labeled and I have formatted JSON labels file and
select the labels file from the drop-down menu below to import your JSON labels
file. Make sure it follows the supported format.
You only need to do this step once for each new language resource you
use.
This process is irreversible, if you connect a storage account to your
Language resource you cannot disconnect it later.
You can only connect your Language resource to one storage account.

 Tip
Your dataset doesn't have to be entirely in the same language. You can have
multiple documents, each with different supported languages. If your dataset
contains documents of different languages or if you expect text from different
languages during runtime, select enable multi-lingual dataset option when you
enter the basic information for your project. This option can be enabled later
from the Project settings page.
\n![Image](images/page672_image1.png)
\n8. Select Next.
9. Review the data you entered and select Create Project.
1. Go to your project settings page in Language Studio
.
2. You can see project details.
3. In this page you can update project description and enable/disable Multi-lingual
dataset in project settings.
4. You can also view the connected storage account and container to your Language
resource.
5. You can also retrieve your primary resource key from this page.
Get project details
Language Studio
\nWhen you don't need your project anymore, you can delete your project using Language
Studio
. Select Custom named entity recognition (NER) from the top, select the project
you want to delete, and then select Delete from the top menu.
You should have an idea of the project schema you will use to label your data.

Delete project
Language Studio
Next steps
\n![Image](images/page674_image1.png)
\nAfter your project is created, you can start labeling your data, which will inform your
entity extraction model how to interpret text, and is used for training and evaluation.
\nHow to prepare data and define a schema
for custom NER
06/30/2025
In order to create a custom NER model, you will need quality data to train it. This article covers
how you should select and prepare your data, along with defining a schema. Defining the
schema is the first step in project development lifecycle, and it defines the entity
types/categories that you need your model to extract from the text at runtime.
The schema defines the entity types/categories that you need your model to extract from text
at runtime.
Review documents in your dataset to be familiar with their format and structure.
Identify the entities you want to extract from the data.
For example, if you are extracting entities from support emails, you might need to extract
"Customer name", "Product name", "Request date", and "Contact information".
Avoid entity types ambiguity.
Ambiguity happens when entity types you select are similar to each other. The more
ambiguous your schema the more labeled data you will need to differentiate between
different entity types.
For example, if you are extracting data from a legal contract, to extract "Name of first
party" and "Name of second party" you will need to add more examples to overcome
ambiguity since the names of both parties look similar. Avoid ambiguity as it saves time,
effort, and yields better results.
Avoid complex entities. Complex entities can be difficult to pick out precisely from text,
consider breaking it down into multiple entities.
For example, extracting "Address" would be challenging if it's not broken down to smaller
entities. There are so many variations of how addresses appear, it would take large
number of labeled entities to teach the model to extract an address, as a whole, without
breaking it down. However, if you replace "Address" with "Street Name", "PO Box", "City",
"State" and "Zip", the model will require fewer labels per entity.
Schema design
\nThe quality of data you train your model with affects model performance greatly.
Use real-life data that reflects your domain's problem space to effectively train your
model. You can use synthetic data to accelerate the initial model training process, but it
will likely differ from your real-life data and make your model less effective when used.
Balance your data distribution as much as possible without deviating far from the
distribution in real-life. For example, if you are training your model to extract entities from
legal documents that may come in many different formats and languages, you should
provide examples that exemplify the diversity as you would expect to see in real life.
Use diverse data whenever possible to avoid overfitting your model. Less diversity in
training data may lead to your model learning spurious correlations that may not exist in
real-life data.
Avoid duplicate documents in your data. Duplicate data has a negative effect on the
training process, model metrics, and model performance.
Consider where your data comes from. If you are collecting data from one person,
department, or part of your scenario, you are likely missing diversity that may be
important for your model to learn about.
As a prerequisite for creating a project, your training data needs to be uploaded to a blob
container in your storage account. You can create and upload training documents from Azure
directly, or through using the Azure Storage Explorer tool. Using the Azure Storage Explorer
tool allows you to upload more data quickly.
Create and upload documents from Azure
Create and upload documents using Azure Storage Explorer
You can only use .txt  documents. If your data is in other format, you can use CLUtils parse
command
 to change your document format.
Data selection
７ Note
If your documents are in multiple languages, select the enable multi-lingual option
during project creation and set the language option to the language of the majority of
your documents.
Data preparation
\nYou can upload an annotated dataset, or you can upload an unannotated one and label your
data in Language studio.
When defining the testing set, make sure to include example documents that are not present
in the training set. Defining the testing set is an important step to calculate the model
performance. Also, make sure that the testing set include documents that represent all entities
used in your project.
If you haven't already, create a custom NER project. If it's your first time using custom NER,
consider following the quickstart to create an example project. You can also see the how-to
article for more details on what you need to create a project.
Test set
Next steps
\nLabel your data in Language Studio
06/30/2025
Before training your model you need to label your documents with the custom entities you
want to extract. Data labeling is a crucial step in development lifecycle. In this step you can
create the entity types you want to extract from your data and label these entities within your
documents. This data will be used in the next step when training your model so that your
model can learn from the labeled data. If you already have labeled data, you can directly
import it into your project but you need to make sure that your data follows the accepted data
format. See create project to learn more about importing labeled data into your project.
Before creating a custom NER model, you need to have labeled data first. If your data isn't
labeled already, you can label it in the Language Studio
. Labeled data informs the model
how to interpret text, and is used for training and evaluation.
Before you can label your data, you need:
A successfully created project with a configured Azure blob storage account
Text data that has been uploaded to your storage account.
See the project development lifecycle for more information.
After preparing your data, designing your schema and creating your project, you will need to
label your data. Labeling your data is important so your model knows which words will be
associated with the entity types you need to extract. When you label your data in Language
Studio
 (or import labeled data), these labels will be stored in the JSON document in your
storage container that you have connected to this project.
As you label your data, keep in mind:
In general, more labeled data leads to better results, provided the data is labeled
accurately.
The precision, consistency and completeness of your labeled data are key factors to
determining model performance.
Label precisely: Label each entity to its right type always. Only include what you want
extracted, avoid unnecessary data in your labels.
Prerequisites
Data labeling guidelines
\nLabel consistently: The same entity should have the same label across all the
documents.
Label completely: Label all the instances of the entity in all your documents. You can
use the auto labelling feature to ensure complete labeling.
Use the following steps to label your data:
1. Go to your project page in Language Studio
.
2. From the left side menu, select Data labeling. You can find a list of all documents in your
storage container.
3. Change to a single document view from the left side in the top menu or select a specific
document to start labeling. You can find a list of all .txt  documents available in your
project to the left. You can use the Back and Next button from the bottom of the page to
navigate through your documents.
4. In the right side pane, Add entity type to your project so you can start labeling your data
with them.
７ Note
There is no fixed number of labels that can guarantee your model will perform the
best. Model performance is dependent on possible ambiguity in your schema, and
the quality of your labeled data. Nevertheless, we recommend having around 50
labeled instances per entity type.
Label your data
 Tip
You can use the filters in top menu to view the unlabeled documents so that you can
start labeling them. You can also use the filters to view the documents that are
labeled with a specific entity type.
７ Note
If you enabled multiple languages for your project, you will find a Language
dropdown in the top menu, which lets you select the language of each document.