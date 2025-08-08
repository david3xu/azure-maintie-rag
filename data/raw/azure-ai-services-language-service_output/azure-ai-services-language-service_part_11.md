How to prepare data and define a text
classification schema
06/30/2025
In order to create a custom text classification model, you will need quality data to train it. This
article covers how you should select and prepare your data, along with defining a schema.
Defining the schema is the first step in project development lifecycle, and it defines the classes
that you need your model to classify your text into at runtime.
The schema defines the classes that you need your model to classify your text into at runtime.
Review and identify: Review documents in your dataset to be familiar with their structure
and content, then identify how you want to classify your data.
For example, if you are classifying support tickets, you might need the following classes:
login issue, hardware issue, connectivity issue, and new equipment request.
Avoid ambiguity in classes: Ambiguity arises when the classes you specify share similar
meaning to one another. The more ambiguous your schema is, the more labeled data you
might need to differentiate between different classes.
For example, if you are classifying food recipes, they may be similar to an extent. To
differentiate between dessert recipe and main dish recipe, you might need to label more
examples to help your model distinguish between the two classes. Avoiding ambiguity
saves time and yields better results.
Out of scope data: When using your model in production, consider adding an out of
scope class to your schema if you expect documents that don't belong to any of your
classes. Then add a few documents to your dataset to be labeled as out of scope. The
model can learn to recognize irrelevant documents, and predict their labels accordingly.
The quality of data you train your model with affects model performance greatly.
Use real-life data that reflects your domain's problem space to effectively train your
model. You can use synthetic data to accelerate the initial model training process, but it
will likely differ from your real-life data and make your model less effective when used.
Schema design
Data selection
\nBalance your data distribution as much as possible without deviating far from the
distribution in real-life.
Use diverse data whenever possible to avoid overfitting your model. Less diversity in
training data may lead to your model learning spurious correlations that may not exist in
real-life data.
Avoid duplicate documents in your data. Duplicate data has a negative effect on the
training process, model metrics, and model performance.
Consider where your data comes from. If you are collecting data from one person,
department, or part of your scenario, you are likely missing diversity that may be
important for your model to learn about.
As a prerequisite for creating a custom text classification project, your training data needs to
be uploaded to a blob container in your storage account. You can create and upload training
documents from Azure directly, or through using the Azure Storage Explorer tool. Using the
Azure Storage Explorer tool allows you to upload more data quickly.
Create and upload documents from Azure
Create and upload documents using Azure Storage Explorer
You can only use .txt . documents for custom text. If your data is in other format, you can use
CLUtils parse command
 to change your file format.
You can upload an annotated dataset, or you can upload an unannotated one and label your
data in Language studio.
When defining the testing set, make sure to include example documents that are not present
in the training set. Defining the testing set is an important step to calculate the model
７ Note
If your documents are in multiple languages, select the multiple languages option during
project creation and set the language option to the language of the majority of your
documents.
Data preparation
Test set
\nperformance. Also, make sure that the testing set include documents that represent all classes
used in your project.
If you haven't already, create a custom text classification project. If it's your first time using
custom text classification, consider following the quickstart to create an example project. You
can also see the project requirements for more details on what you need to create a project.
Next steps
\nLabel text data for training your model
06/30/2025
Before training your model, you need to label your documents with the classes you want to
categorize them into. Data labeling is a crucial step in development lifecycle; in this step you
can create the classes you want to categorize your data into and label your documents with
these classes. This data will be used in the next step when training your model so that your
model can learn from the labeled data. If you already labeled your data, you can directly import
it into your project but you need to make sure that your data follows the accepted data format.
Before creating a custom text classification model, you need to have labeled data first. If your
data isn't labeled already, you can label it in the Language Studio
. Labeled data informs the
model how to interpret text, and is used for training and evaluation.
Before you can label data, you need:
A successfully created project with a configured Azure blob storage account,
Documents containing the uploaded text data in your storage account.
See the project development lifecycle for more information.
After preparing your data, designing your schema and creating your project, you will need to
label your data. Labeling your data is important so your model knows which documents will be
associated with the classes you need. When you label your data in Language Studio
 (or
import labeled data), these labels are stored in the JSON file in your storage container that
you've connected to this project.
As you label your data, keep in mind:
In general, more labeled data leads to better results, provided the data is labeled
accurately.
There is no fixed number of labels that can guarantee your model performs the best.
Model performance on possible ambiguity in your schema, and the quality of your
labeled data. Nevertheless, we recommend 50 labeled documents per class.
Prerequisites
Data labeling guidelines
Label your data
\nUse the following steps to label your data:
1. Go to your project page in Language Studio
.
2. From the left side menu, select Data labeling. You can find a list of all documents in your
storage container. See the image below.
3. Change to a single file view from the left side in the top menu or select a specific file to
start labeling. You can find a list of all .txt  files available in your projects to the left. You
can use the Back and Next button from the bottom of the page to navigate through your
documents.
4. In the right side pane, Add class to your project so you can start labeling your data with
them.
5. Start labeling your files.
Multi label classification: your file can be labeled with multiple classes. You can do so
by selecting all applicable check boxes next to the classes you want to label this
document with.
 Tip
You can use the filters in top menu to view the unlabeled files so that you can start
labeling them. You can also use the filters to view the documents that are labeled
with a specific class.
７ Note
If you enabled multiple languages for your project, you will find a Language
dropdown in the top menu, which lets you select the language of each document.
Multi label classification
\nYou can also use the auto labeling feature to ensure complete labeling.
6. In the right side pane under the Labels pivot you can find all the classes in your project
and the count of labeled instances per each.
7. In the bottom section of the right side pane you can add the current file you're viewing to
the training set or the testing set. By default all the documents are added to your training
set. Learn more about training and testing sets and how they're used for model training
and evaluation.
8. Under the Distribution pivot you can view the distribution across training and testing
sets. You have two options for viewing:
Total instances where you can view count of all labeled instances of a specific class.
documents with at least one label where each document is counted if it contains at
least one labeled instance of this class.
9. While you're labeling, your changes are synced periodically, if they have not been saved
yet you will find a warning at the top of your page. If you want to save manually, select
Save labels button at the bottom of the page.

 Tip
If you're planning on using Automatic data splitting, use the default option of
assigning all the documents into your training set.
\n![Image](images/page106_image1.png)
\nIf you want to remove a label, uncheck the button next to the class.
To delete a class, select the icon next to the class you want to remove. Deleting a class will
remove all its labeled instances from your dataset.
After you've labeled your data, you can begin training a model that will learn based on your
data.
Remove labels
Delete or classes
Next steps
\nHow to use autolabeling for Custom Text
Classification
06/30/2025
Labeling process is an important part of preparing your dataset. Since this process requires
much time and effort, you can use the autolabeling feature to automatically label your
documents with the classes you want to categorize them into. You can currently start
autolabeling jobs based on a model using GPT models where you may immediately trigger an
autolabeling job without any prior model training. This feature can save you the time and effort
of manually labeling your documents.
Before you can use autolabeling with GPT, you need:
A successfully created project with a configured Azure blob storage account.
Text data that has been uploaded to your storage account.
Class names that are meaningful. The GPT models label documents based on the names
of the classes you've provided.
Labeled data isn't required.
An Azure OpenAI resource and deployment.
When you trigger an autolabeling job with GPT, you're charged to your Azure OpenAI resource
as per your consumption. You're charged an estimate of the number of tokens in each
document being autolabeled. Refer to the Azure OpenAI pricing page
 for a detailed
breakdown of pricing per token of different models.
1. From the left pane, select Data labeling.
2. Select the Autolabel button under the Activity pane to the right of the page.
Prerequisites
Trigger an autolabeling job
\n3. Choose Autolabel with GPT and select Next.
4. Choose your Azure OpenAI resource and deployment. You must create an Azure OpenAI
resource and deploy a model in order to proceed.


\n![Image](images/page109_image1.png)

![Image](images/page109_image2.png)
\n5. Select the classes you want to be included in the autolabeling job. By default, all classes
are selected. Having descriptive names for classes, and including examples for each class
is recommended to achieve good quality labeling with GPT.
6. Choose the documents you want to be automatically labeled. It's recommended to
choose the unlabeled documents from the filter.


７ Note
If a document was automatically labeled, but this label was already user
defined, only the user defined label is used.
\n![Image](images/page110_image1.png)

![Image](images/page110_image2.png)