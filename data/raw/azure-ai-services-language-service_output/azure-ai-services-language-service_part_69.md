5. You have two options to label your document:
Option
Description
Label using a
brush
Select the brush icon next to an entity type in the right pane, then highlight the
text in the document you want to annotate with this entity type.
Label using a
menu
Highlight the word you want to label as an entity, and a menu will appear. Select
the entity type you want to assign for this entity.
The below screenshot shows labeling using a brush.
6. In the right side pane under the Labels pivot you can find all the entity types in your
project and the count of labeled instances per each.
7. In the bottom section of the right side pane you can add the current document you are
viewing to the training set or the testing set. By default all the documents are added to
your training set. Learn more about training and testing sets and how they are used for
model training and evaluation.
8. Under the Distribution pivot you can view the distribution across training and testing
sets. You have two options for viewing:
Total instances where you can view count of all labeled instances of a specific entity
type.
documents with at least one label where each document is counted if it contains at
least one labeled instance of this entity.
ﾉ
Expand table

 Tip
If you are planning on using Automatic data splitting, use the default option of
assigning all the documents into your training set.
\n![Image](images/page681_image1.png)
\n9. When you're labeling, your changes will be synced periodically, if they have not been
saved yet you will find a warning at the top of your page. If you want to save manually,
select Save labels button at the bottom of the page.
To remove a label
1. Select the entity you want to remove a label from.
2. Scroll through the menu that appears, and select Remove label.
To delete an entity, select the delete icon next to the entity you want to remove. Deleting an
entity will remove all its labeled instances from your dataset.
After you've labeled your data, you can begin training a model that will learn based on your
data.
Remove labels
Delete entities
Next steps
\nHow to use autolabeling for Custom
Named Entity Recognition
06/30/2025
Labeling process is an important part of preparing your dataset. Since this process requires
both time and effort, you can use the autolabeling feature to automatically label your entities.
You can start autolabeling jobs based on a model you've previously trained or using GPT
models. With autolabeling based on a model you've previously trained, you can start labeling a
few of your documents, train a model, then create an autolabeling job to produce entity labels
for other documents based on that model. With autolabeling with GPT, you may immediately
trigger an autolabeling job without any prior model training. This feature can save you the time
and effort of manually labeling your entities.
Before you can use autolabeling based on a model you've trained, you need:
A successfully created project with a configured Azure blob storage account.
Text data that has been uploaded to your storage account.
Labeled data
A successfully trained model
When you trigger an autolabeling job based on a model you've trained, there's a monthly
limit of 5,000 text records per month, per resource. This means the same limit applies on
all projects within the same resource.
Prerequisites
Autolabel based on a model you've trained
Trigger an autolabeling job
Autolabel based on a model you've trained
 Tip
A text record is calculated as the ceiling of (Number of characters in a document /
1,000). For example, if a document has 8921 characters, the number of text records is:
\n1. From the left pane, select Data labeling.
2. Select the Autolabel button under the Activity pane to the right of the page.
3. Choose Autolabel based on a model you've trained and select Next.
ceil(8921/1000) = ceil(8.921) , which is 9 text records.


\n![Image](images/page684_image1.png)

![Image](images/page684_image2.png)
\n4. Choose a trained model. It's recommended to check the model performance before
using it for autolabeling.
5. Choose the entities you want to be included in the autolabeling job. By default, all
entities are selected. You can see the total labels, precision and recall of each entity.
It's recommended to include entities that perform well to ensure the quality of the
automatically labeled entities.
6. Choose the documents you want to be automatically labeled. The number of text
records of each document is displayed. When you select one or more documents,
you should see the number of texts records selected. It's recommended to choose
the unlabeled documents from the filter.


\n![Image](images/page685_image1.png)

![Image](images/page685_image2.png)
\n7. Select Autolabel to trigger the autolabeling job. You should see the model used,
number of documents included in the autolabeling job, number of text records and
entities to be automatically labeled. Autolabeling jobs can take anywhere from a few
seconds to a few minutes, depending on the number of documents you included.
７ Note
If an entity was automatically labeled, but has a user defined label, only the
user defined label is used and visible.
You can view the documents by clicking on the document name.


\n![Image](images/page686_image1.png)

![Image](images/page686_image2.png)
\nWhen the autolabeling job is complete, you can see the output documents in the Data
labeling page of Language Studio. Select Review documents with autolabels to view the
documents with the Auto labeled filter applied.
Entities that have been automatically labeled appear with a dotted line. These entities have two
selectors (a checkmark and an "X") that allow you to accept or reject the automatic label.
Once an entity is accepted, the dotted line changes to a solid one, and the label is included in
any further model training becoming a user defined label.
Alternatively, you can accept or reject all automatically labeled entities within the document,
using Accept all or Reject all in the top right corner of the screen.
After you accept or reject the labeled entities, select Save labels to apply the changes.
Review the auto labeled documents

７ Note
We recommend validating automatically labeled entities before accepting them.
All labels that were not accepted are be deleted when you train your model.

\n![Image](images/page687_image1.png)

![Image](images/page687_image2.png)
\nLearn more about labeling your data.
Next steps
\nUse Azure Machine Learning labeling in
Language Studio
06/30/2025
Labeling data is an important part of preparing your dataset. Using the labeling experience in
Azure Machine Learning, you can experience easier collaboration, more flexibility, and the
ability to outsource labeling tasks to external labeling vendors from the Azure Market Place
.
You can use Azure Machine Learning labeling for:
custom text classification
custom named entity recognition
Before you can connect your labeling project to Azure Machine Learning, you need:
A successfully created Language Studio project with a configured Azure blob storage
account.
Text data that has been uploaded to your storage account.
At least:
One entity label for custom named entity recognition, or
Two class labels for custom text classification projects.
An Azure Machine Learning workspace that has been connected to the same Azure blob
storage account that your Language Studio account using.
Connecting your labeling project to Azure Machine Learning is a one-to-one connection.
If you disconnect your project, you will not be able to connect your project back to the
same Azure Machine Learning project
You can't label in the Language Studio and Azure Machine Learning simultaneously. The
labeling experience is enabled in one studio at a time.
The testing and training files in the labeling experience you switch away from will be
ignored when training your model.
Only Azure Machine Learning's JSONL file format can be imported into Language Studio.
Projects with the multi-lingual option enabled can't be connected to Azure Machine
Learning, and not all languages are supported.
Language support is provided by the Azure Machine Learning TextDNNLanguages
Class.
Prerequisites
Limitations
\nThe Azure Machine Learning workspace you're connecting to must be assigned to the
same Azure Storage account that Language Studio is connected to. Be sure that the Azure
Machine Learning workspace has the storage blob data reader permission on the storage
account. The workspace needs to have been linked to the storage account during the
creation process in the Azure portal
.
Switching between the two labeling experiences isn't instantaneous. It may take time to
successfully complete the operation.
Language Studio supports the JSONL file format used by Azure Machine Learning. If you’ve
been labeling data on Azure Machine Learning, you can import your up-to-date labels in a new
custom project to utilize the features of both studios.
1. Start by creating a new project for custom text classification or custom named entity
recognition.
a. In the Create a project screen that appears, follow the prompts to connect your
storage account, and enter the basic information about your project. Be sure that the
Azure resource you're using doesn't have another storage account already connected.
b. In the Choose container section, choose the option indicating that you already have a
correctly formatted file. Then select your most recent Azure Machine Learning labels
file.
Import your Azure Machine Learning labels into
Language Studio

\n![Image](images/page690_image1.png)