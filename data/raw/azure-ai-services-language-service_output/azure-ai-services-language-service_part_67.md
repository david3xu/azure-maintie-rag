PowerShell
1. Sign in to the Azure portal
 to create a new Azure AI Language resource.
2. In the window that appears, select Custom text classification & custom named entity
recognition from the custom features. Select Continue to create your resource at the
bottom of the screen.
3. Create a Language resource with following details.
Name
Description
Subscription
Your Azure subscription.
Resource
group
A resource group that will contain your resource. You can use an existing one, or
create a new one.
Region
The region for your Language resource. For example, "West US 2".
７ Note
You shouldn't move the storage account to a different resource group or subscription
once it's linked with the Language resource.
Create a new resource from the Azure portal

ﾉ
Expand table
\n![Image](images/page661_image1.png)
\nName
Description
Name
A name for your resource.
Pricing tier
The pricing tier for your Language resource. You can use the Free (F0) tier to try
the service.
4. In the Custom text classification & custom named entity recognition section, select an
existing storage account or select New storage account. These values are to help you get
started, and not necessarily the storage account values you’ll want to use in production
environments. To avoid latency during building your project connect to storage accounts
in the same region as your Language resource.
Storage account value
Recommended value
Storage account name
Any name
Storage account type
Standard LRS
5. Make sure the Responsible AI Notice is checked. Select Review + create at the bottom of
the page, then select Create.
If it's your first time logging in, you'll see a window in Language Studio
 that will let you
choose an existing Language resource or create a new one. You can also create a resource by
clicking the settings icon in the top-right corner, selecting Resources, then clicking Create a
new resource.
Create a Language resource with following details.
７ Note
If you get a message saying "your login account is not an owner of the selected
storage account's resource group", your account needs to have an owner role
assigned on the resource group before you can create a Language resource. Contact
your Azure subscription owner for assistance.
ﾉ
Expand table
Create a new Language resource from Language Studio
ﾉ
Expand table
\nInstance detail
Required value
Azure subscription
Your Azure subscription
Azure resource group
Your Azure resource group
Azure resource name
Your Azure resource name
Location
The region of your Language resource.
Pricing tier
The pricing tier of your Language resource.
To use custom named entity recognition, you'll need to create an Azure storage account if you
don't have one already.
You can create a new resource and a storage account using the following CLI template
 and
parameters
 files, which are hosted on GitHub.
Edit the following values in the parameters file:
Parameter name
Value description
name
Name of your Language resource
location
Region in which your resource is hosted. for more information, see Service
limits.
sku
Pricing tier of your resource.
storageResourceName
Name of your storage account
storageLocation
Region in which your storage account is hosted.
storageSkuType
SKU of your storage account.
storageResourceGroupName
Resource group of your storage account
） Important
Make sure to enable Managed Identity when you create a Language resource.
Read and confirm Responsible AI notice
Create a new Language resource using PowerShell
ﾉ
Expand table
\nUse the following PowerShell command to deploy the Azure Resource Manager (ARM)
template with the files you edited.
PowerShell
See the ARM template documentation for information on deploying templates and parameter
files.
You can use an existing Language resource to get started with custom NER as long as this
resource meets the below requirements:
Requirement
Description
Regions
Make sure your existing resource is provisioned in one of the supported regions. If not,
you will need to create a new resource in one of these regions.
Pricing tier
Learn more about supported pricing tiers.
Managed
identity
Make sure that the resource's managed identity setting is enabled. Otherwise, read the
next section.
To use custom named entity recognition, you'll need to create an Azure storage account if you
don't have one already.
New-AzResourceGroupDeployment -Name ExampleDeployment -ResourceGroupName 
ExampleResourceGroup `
  -TemplateFile <path-to-arm-template> `
  -TemplateParameterFile <path-to-parameters-file>
７ Note
The process of connecting a storage account to your Language resource is
irreversible, it cannot be disconnected later.
You can only connect your language resource to one storage account.
Using a pre-existing Language resource
ﾉ
Expand table
Enable identity management for your resource
Azure portal
\nYour Language resource must have identity management, to enable it using the Azure
portal
:
1. Go to your Language resource
2. From left hand menu, under Resource Management section, select Identity
3. From System assigned tab, make sure to set Status to On
Make sure to enable Custom text classification / Custom Named Entity Recognition feature
from Azure portal.
1. Go to your Language resource in the Azure portal
.
2. From the left side menu, under Resource Management section, select Features.
3. Enable Custom text classification / Custom Named Entity Recognition feature.
4. Connect your storage account.
5. Select Apply.
Use the following steps to set the required roles for your Language resource and storage
account.
Enable custom named entity recognition feature
） Important
Make sure that the user making changes has storage blob data contributor role assigned
for them.
Add required roles
\n1. Go to your storage account or Language resource in the Azure portal
.
2. Select Access Control (IAM) in the left pane.
3. Select Add to Add Role Assignments, and choose the appropriate role for your account.
You should have the owner or contributor role assigned on your Language resource.
4. Within Assign access to, select User, group, or service principal
5. Select Select members
6. Select your user name. You can search for user names in the Select field. Repeat this for
all roles.
7. Repeat these steps for all the user accounts that need access to this resource.
1. Go to your storage account page in the Azure portal
.
2. Select Access Control (IAM) in the left pane.
3. Select Add to Add Role Assignments, and choose the Storage blob data contributor role
on the storage account.

Roles for your Azure AI Language resource
Roles for your storage account
\n![Image](images/page666_image1.png)
\n4. Within Assign access to, select Managed identity.
5. Select Select members
6. Select your subscription, and Language as the managed identity. You can search for user
names in the Select field.
1. Go to your storage account page in the Azure portal
.
2. Select Access Control (IAM) in the left pane.
3. Select Add to Add Role Assignments, and choose the Storage blob data contributor role
on the storage account.
4. Within Assign access to, select User, group, or service principal.
5. Select Select members
6. Select your User. You can search for user names in the Select field.
Make sure to allow (GET, PUT, DELETE) methods when enabling Cross-Origin Resource Sharing
(CORS). Set allowed origins field to https://language.cognitive.azure.com . Allow all header by
adding *  to the allowed header values, and set the maximum age to 500 .
Roles for your user
） Important
If you skip this step, you'll have a 403 error when trying to connect to your custom project.
It's important that your current user has this role to access storage account blob data,
even if you're the owner of the storage account.
） Important
If you have a virtual network or private endpoint, be sure to select Allow Azure services
on the trusted services list to access this storage account in the Azure portal.
Enable CORS for your storage account
\nOnce your resource and storage container are configured, create a new custom NER project. A
project is a work area for building your custom AI models based on your data. Your project can
only be accessed by you and others who have access to the Azure resource being used. If you
have labeled data, you can use it to get started by importing a project.
1. Sign into the Language Studio
. A window will appear to let you select your
subscription and Language resource. Select the Language resource you created in
the above step.
2. Under the Extract information section of Language Studio, select Custom named
entity recognition.

Create a custom named entity recognition project
Language Studio

\n![Image](images/page668_image1.png)

![Image](images/page668_image2.png)
\n3. Select Create new project from the top menu in your projects page. Creating a
project will let you tag data, train, evaluate, improve, and deploy your models.
4. After you click, Create new project, a window will appear to let you connect your
storage account. If you've already connected a storage account, you will see the
storage accounted connected. If not, choose your storage account from the
dropdown that appears and select Connect storage account; this will set the
required roles for your storage account. This step will possibly return an error if you
are not assigned as owner on the storage account.

７ Note
You only need to do this step once for each new resource you use.
This process is irreversible, if you connect a storage account to your
Language resource you cannot disconnect it later.
You can only connect your Language resource to one storage account.
\n![Image](images/page669_image1.png)
\n5. Enter the project information, including a name, description, and the language of the
files in your project. If you're using the example dataset
, select English. You won’t
be able to change the name of your project later. Select Next
6. Select the container where you have uploaded your dataset. If you have already
labeled data make sure it follows the supported format and select Yes, my files are
already labeled and I have formatted JSON labels file and select the labels file from
the drop-down menu. Select Next.
7. Review the data you entered and select Create Project.
If you have already labeled data, you can use it to get started with the service. Make sure that
your labeled data follows the accepted data formats.

 Tip
Your dataset doesn't have to be entirely in the same language. You can have
multiple documents, each with different supported languages. If your dataset
contains documents of different languages or if you expect text from different
languages during runtime, select enable multi-lingual dataset option when you
enter the basic information for your project. This option can be enabled later
from the Project settings page.
Import project
\n![Image](images/page670_image1.png)