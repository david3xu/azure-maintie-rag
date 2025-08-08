You can also review the confusion matrix to identify classes that are often mistakenly predicted
to see if anything can be done to improve model performance. If you notice that a specific
class is often predicted as another class, it's a strong indicator that these two classes are similar
to each other. You might need to rethink your schema. Or you can add more tagged examples
to your dataset to help the model differentiate these classes.
After you've viewed evaluation details for your model, you can improve your model. This
process enables you to view the predicted and tagged classes side by side to determine what
went wrong during model evaluation. If you find that some classes are interchangeably
repeated, consider adding them all to a higher order which represents multiple classes for
better prediction.
Custom text classification gives you the option to use data in multiple languages. You can have
multiple files in your dataset of different languages. Also, you can train your model in one
language and use it to query text in other languages. If you want to use the multilingual
option, you have to enable this option during project creation.
If you notice low scores in a certain language, consider adding more data in this language to
your dataset. To learn more about supported languages, see [this website/azure/ai-
services/language-service/custom-text-classification/language-support).
Introduction to custom text classification
Custom text classification Transparency Note
Data privacy and security
Guidance for integration and responsible use
Microsoft AI principles
Performance varies across features and languages
Next steps
\nData and privacy for Custom text
classification
06/24/2025
This article provides high-level details about how data is processed by custom text
classification. Remember that you're responsible for your use and the implementation of this
technology, which includes complying with all laws and regulations that apply to you. For
example, it's your responsibility to:
Understand where your data is processed and stored by the custom text classification
service to meet regulatory obligations for your application.
Ensure you have all necessary licenses, proprietary rights, or other permissions required
for the content in your dataset that's used as the basis for building your custom text
classification models.
It's your responsibility to comply with all applicable laws and regulations in your jurisdiction.
Custom text classification processes the following data:
User's dataset and tags file: As a prerequisite to creating a custom text classification
project, users need to upload their dataset to their Azure Blob Storage container. A tags
file is a JSON-formatted file that contains references to a user's tagged data and classes.
The user can either bring their own tags or they can tag their data through the UI
experience in the Language Studio
. Either way, a tags file that contains tagged data and
classes is essential for the training.
A user's dataset is split into train and test sets, where the split can either be predefined by
developers in a tags file or chosen at random during training. The train set and the tags
file are processed during training to create the custom text classification model. The test
set is later processed by the trained model to evaluate its performance.
Custom text classification models: Based on the user's request to train the model,
custom text classification processes the provided tagged data to output a trained model.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What data does Custom text classification process?
\nThe user can choose to train a new model or overwrite an existing one. The trained model
is then stored on the service's side and used for processing the model evaluation. After
the developer is content with the model's performance, they request to deploy the model
for consumption use. The deployed model is also stored on the service's side, which is
used to process the user's requests for prediction through the Analyze API.
Data sent for classification: This data is the user's text sent from a customer's client
application through the Analyze API
 to be processed for text classification by the
custom machine learning model. Output of the processed data contains the predicted
classes along with their confidence scores. This output is returned to the client's
application to perform an action to fulfill the user's request.
Custom text classification doesn't collect or store any customer data to improve its machine-
learned models or for product improvement purposes. We use aggregate telemetry, such as
which APIs are used and the number of calls from each subscription and resource, for service
monitoring purposes.
The following diagram illustrates how your data is processed.
How does Custom text classification process data?
\n![Image](images/page83_image1.png)
\nCustom text classification is a data processor for General Data Protection Regulation (GDPR)
purposes. In compliance with GDPR policies, custom text classification users have full control to
view, export, or delete any user content either through the Language Studio
 or
programmatically by using Language APIs.
Your data is only stored in your Azure Storage account. custom text classification only has
access to read from it during training.
Customer controls include:
Tagged data provided by the user as a prerequisite to train the model is saved in the
customer's Azure Storage account that's connected to the project during creation.
Customers can edit or remove tags whenever they want through the Language Studio.
Custom text classification projects metadata is stored in the service's side until the
customer deletes the project. The project's metadata are the fields that you fill in when
you create your project, such as project name, description, language, name of connected
blob container, and tags file location.
Trained custom text classification models are stored in the service's Azure Storage
accounts until the customer deletes them. The model is overwritten each time the user
retrains it.
Deployed custom text classification models persist in the service's Azure Storage accounts
until the customer deletes the deployment or deletes the model itself. The model is
overwritten each time the user deploys to the same deployment name.
Azure services are implemented while maintaining appropriate technical and organizational
measures to protect customer data in the cloud.
To learn more about Microsoft's privacy and security commitments, see the Microsoft Trust
Center
.
Introduction to Custom text classification
Custom text classification Transparency Note
Guidance for integration and responsible use
Microsoft AI principles
How is data retained, and what customer controls
are available?
Optional: Security for customers' data
Next steps
\n\nHow to create custom text classification
project
06/30/2025
Use this article to learn how to set up the requirements for starting with custom text
classification and create a project.
Before you start using custom text classification, you will need:
An Azure subscription - Create one for free
.
Before you start using custom text classification, you will need an Azure AI Language resource.
It is recommended to create your Language resource and connect a storage account to it in
the Azure portal. Creating a resource in the Azure portal lets you create an Azure storage
account at the same time, with all of the required permissions pre-configured. You can also
read further in the article to learn how to use a pre-existing resource, and configure it to work
with custom text classification.
You also will need an Azure storage account where you will upload your .txt  documents that
will be used to train a model to classify text.
Prerequisites
Create a Language resource
７ Note
You need to have an owner role assigned on the resource group to create a
Language resource.
If you will connect a pre-existing storage account, you should have an owner role
assigned to it.
Create Language resource and connect storage
account
７ Note
\n1. Go to the Azure portal
 to create a new Azure AI Language resource.
2. In the window that appears, select Custom text classification & custom named
entity recognition from the custom features. Select Continue to create your
resource at the bottom of the screen.
3. Create a Language resource with following details.
Name
Required value
Subscription
Your Azure subscription.
Resource
group
A resource group that will contain your resource. You can use an existing
one, or create a new one.
Region
One of the supported regions. For example "West US 2".
Name
A name for your resource.
You shouldn't move the storage account to a different resource group or subscription
once it's linked with the Language resource.
Using the Azure portal
Create a new resource from the Azure portal

ﾉ
Expand table
\n![Image](images/page87_image1.png)
\nName
Required value
Pricing tier
One of the supported pricing tiers. You can use the Free (F0) tier to try the
service.
If you get a message saying "your login account is not an owner of the selected
storage account's resource group", your account needs to have an owner role assigned
on the resource group before you can create a Language resource. Contact your
Azure subscription owner for assistance.
You can determine your Azure subscription owner by searching your resource
group
 and following the link to its associated subscription. Then:
a. Select the Access Control (IAM) tab
b. Select Role assignments
c. Filter by Role:Owner.
4. In the Custom text classification & custom named entity recognition section, select
an existing storage account or select New storage account. Note that these values
are to help you get started, and not necessarily the storage account values you’ll
want to use in production environments. To avoid latency during building your
project connect to storage accounts in the same region as your Language resource.
Storage account value
Recommended value
Storage account name
Any name
Storage account type
Standard LRS
5. Make sure the Responsible AI Notice is checked. Select Review + create at the
bottom of the page.
ﾉ
Expand table
７ Note
The process of connecting a storage account to your Language resource is
irreversible, it cannot be disconnected later.
You can only connect your language resource to one storage account.
Using a pre-existing Language resource
\nRequirement
Description
Regions
Make sure your existing resource is provisioned in one of the supported regions. If you
don't have a resource, you will need to create a new one in a supported region.
Pricing tier
The pricing tier for your resource.
Managed
identity
Make sure that the resource's managed identity setting is enabled. Otherwise, read the
next section.
To use custom text classification, you'll need to create an Azure storage account if you don't
have one already.
Your Language resource must have identity management, to enable it using Azure
portal
:
1. Go to your Language resource
2. From left hand menu, under Resource Management section, select Identity
3. From System assigned tab, make sure to set Status to On
Make sure to enable Custom text classification / Custom Named Entity Recognition feature
from Azure portal.
1. Go to your Language resource in Azure portal
2. From the left side menu, under Resource Management section, select Features
3. Enable Custom text classification / Custom Named Entity Recognition feature
4. Connect your storage account
5. Select Apply
ﾉ
Expand table
Enable identity management for your resource
Azure portal
Enable custom text classification feature
） Important
Make sure that your Language resource has storage blob data contributor role
assigned on the storage account you are connecting.
\nUse the following steps to set the required roles for your Language resource and storage
account.
1. Go to your storage account or Language resource in the Azure portal
.
2. Select Access Control (IAM) in the left pane.
3. Select Add to Add Role Assignments, and choose the appropriate role for your account.
You should have the owner or contributor role assigned on your Language resource.
4. Within Assign access to, select User, group, or service principal
5. Select Select members
6. Select your user name. You can search for user names in the Select field. Repeat this for
all roles.
7. Repeat these steps for all the user accounts that need access to this resource.
Set roles for your Azure AI Language resource and storage
account

Roles for your Azure AI Language resource
\n![Image](images/page90_image1.png)