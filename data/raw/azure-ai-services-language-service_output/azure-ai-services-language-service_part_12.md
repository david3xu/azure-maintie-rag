7. Select Start job to trigger the autolabeling job. You should be directed to the
autolabeling page displaying the autolabeling jobs initiated. Autolabeling jobs can take
anywhere from a few seconds to a few minutes, depending on the number of documents
you included.
You can view the documents by clicking on the document name.


Review the auto labeled documents
\n![Image](images/page111_image1.png)

![Image](images/page111_image2.png)
\nWhen the autolabeling job is complete, you can see the output documents in the Data
labeling page of Language Studio. Select Review documents with autolabels to view the
documents with the Auto labeled filter applied.
Documents that have been automatically classified have suggested labels in the activity pane
highlighted in purple. Each suggested label has two selectors (a checkmark and a cancel icon)
that allow you to accept or reject the automatic label.
Once a label is accepted, the purple color changes to the default blue one, and the label is
included in any further model training becoming a user defined label.
After you accept or reject the labels for the autolabeled documents, select Save labels to apply
the changes.

７ Note
We recommend validating automatically labeled documents before accepting them.
All labels that were not accepted are deleted when you train your model.
\n![Image](images/page112_image1.png)
\nLearn more about labeling your data.

Next steps
\n![Image](images/page113_image1.png)
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
\n![Image](images/page115_image1.png)
\nBefore you connect to Azure Machine Learning, you need an Azure Machine Learning account
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
\n![Image](images/page116_image1.png)
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
\n![Image](images/page117_image1.png)
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
\n![Image](images/page118_image1.png)
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
\n![Image](images/page119_image1.png)
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
\n![Image](images/page120_image1.png)