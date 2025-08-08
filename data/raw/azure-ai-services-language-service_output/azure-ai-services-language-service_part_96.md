Deploy and run containers on Azure
Container Instance
05/19/2025
With the following steps, scale Azure AI services applications in the cloud easily with Azure
Container Instances. Containerization helps you focus on building your applications instead of
managing the infrastructure. For more information on using containers, see features and
benefits.
The recipe works with any Azure AI services container. The Azure AI Foundry resource must be
created before using the recipe. Each Azure AI service that supports containers has a "How to
install" article for installing and configuring the service for a container. Some services require a
file or set of files as input for the container, it is important that you understand and have used
the container successfully before using this solution.
An Azure resource for the Azure AI service that you're using.
Azure resource endpoint URL - review your specific service's "How to install" for the
container, to find where the endpoint URL is from within the Azure portal, and what a
correct example of the URL looks like. The exact format can change from service to
service.
Azure resource key - the keys are on the Keys page for the Azure resource. You only need
one of the two keys. The key is a string of 84 alpha-numeric characters.
A single Azure AI services container on your local host (your computer). Make sure you
can:
Pull down the image with a docker pull  command.
Run the local container successfully with all required configuration settings with a
docker run  command.
Call the container's endpoint, getting a response of HTTP 2xx and a JSON response
back.
All variables in angle brackets, <> , need to be replaced with your own values. This replacement
includes the angle brackets.
Prerequisites
） Important
\n1. Go to the Create
 page for Container Instances.
2. On the Basics tab, enter the following details:
Setting
Value
Subscription
Select your subscription.
Resource
group
Select the available resource group or create a new one such as cognitive-
services .
Container
name
Enter a name such as cognitive-container-instance . The name must be in
lower caps.
Location
Select a region for deployment.
Image type
If your container image is stored in a container registry that doesn’t require
credentials, choose Public . If accessing your container image requires
credentials, choose Private . Refer to container repositories and images for
details on whether or not the container image is Public  or Private  ("Public
Preview").
The LUIS container requires a .gz  model file that is pulled in at runtime. The container
must be able to access this model file via a volume mount from the container instance. To
upload a model file, follow these steps:
1. Create an Azure file share. Take note of the Azure Storage account name, key, and
file share name as you'll need them later.
2. export your LUIS model (packaged app) from the LUIS portal.
3. In the Azure portal, navigate to the Overview page of your storage account resource,
and select File shares.
4. Select the file share name that you recently created, then select Upload. Then upload
your packaged app.
Azure portal
Create an Azure Container Instance resource
using the Azure portal
ﾉ
Expand table
\nSetting
Value
Image name
Enter the Azure AI services container location. The location is what's used as an
argument to the docker pull  command. Refer to the container repositories
and images for the available image names and their corresponding repository.
The image name must be fully qualified specifying three parts. First, the
container registry, then the repository, finally the image name: <container-
registry>/<repository>/<image-name> .
Here is an example, mcr.microsoft.com/azure-cognitive-services/keyphrase
would represent the Key Phrase Extraction image in the Microsoft Container
Registry under the Azure AI services repository. Another example is,
containerpreview.azurecr.io/microsoft/cognitive-services-speech-to-text
which would represent the Speech to text image in the Microsoft repository of
the Container Preview container registry.
OS type
Linux
Size
Change size to the suggested recommendations for your specific Azure AI
container:
2 CPU cores
4 GB
3. On the Networking tab, enter the following details:
Setting
Value
Ports
Set the TCP port to 5000 . Exposes the container on port 5000.
4. On the Advanced tab, enter the required Environment Variables for the container
billing settings of the Azure Container Instance resource:
Key
Value
ApiKey
Copied from the Keys and endpoint page of the resource. It is a 84 alphanumeric-
character string with no spaces or dashes, xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx .
Billing
Your endpoint URL copied from the Keys and endpoint page of the resource.
Eula
accept
5. Select Review and Create
ﾉ
Expand table
ﾉ
Expand table
\n6. After validation passes, click Create to finish the creation process
7. When the resource is successfully deployed, it's ready
1. Select the Overview and copy the IP address. It will be a numeric IP address such as
55.55.55.55 .
2. Open a new browser tab and use the IP address, for example, http://<IP-
address>:5000 (http://55.55.55.55:5000 ). You will see the container's home page,
letting you know the container is running.
3. Select Service API Description to view the swagger page for the container.
4. Select any of the POST APIs and select Try it out. The parameters are displayed
including the input. Fill in the parameters.
5. Select Execute to send the request to your Container Instance.
You have successfully created and used Azure AI containers in Azure Container
Instance.
Use the Container Instance
Azure portal
\n![Image](images/page954_image1.png)
\nUse Docker containers in disconnected
environments
Article • 01/17/2025
Containers enable you to run Azure AI services APIs in your own environment, and are
great for your specific security and data governance requirements. Disconnected
containers enable you to use several of these APIs disconnected from the internet.
Currently, the following containers can be run in this manner:
Speech to text
Custom Speech to text
Neural Text to speech
Text Translation (Standard)
Azure AI Language
Sentiment Analysis
Key Phrase Extraction
Language Detection
Summarization
Named Entity Recognition
Personally Identifiable Information (PII) detection
Conversational Language Understanding (CLU)
Azure AI Vision - Read
Document Intelligence
Before attempting to run a Docker container in an offline environment, make sure you
know the steps to successfully download and use the container. For example:
Host computer requirements and recommendations.
The Docker pull  command you use to download the container.
How to validate that a container is running.
How to send queries to the container's endpoint, once it's running.
Fill out and submit the request form
 to request access to the containers disconnected
from the internet.
Request access to use containers in
disconnected environments
\nThe form requests information about you, your company, and the user scenario for
which you'll use the container. After you submit the form, the Azure AI services team
reviews it and emails you with a decision within 10 business days.
After you're approved, you'll be able to run the container after you download it from the
Microsoft Container Registry (MCR), described later in the article.
You won't be able to run the container if your Azure subscription hasn't been approved.
Access is limited to customers that meet the following requirements:
Your organization should be identified as strategic customer or partner with
Microsoft.
Disconnected containers are expected to run fully offline, hence your use cases
must meet one of these or similar requirements:
Environments or devices with zero connectivity to internet.
Remote location that occasionally has internet access.
Organization under strict regulation of not sending any kind of data back to
cloud.
Application completed as instructed - Pay close attention to guidance provided
throughout the application to ensure you provide all the necessary information
required for approval.
1. Sign in to the Azure portal
 and select Create a new resource for one of the
applicable Azure AI services listed.
） Important
On the form, you must use an email address associated with an Azure
subscription ID.
The Azure resource you use to run the container must have been created with
the approved Azure subscription ID.
Check your email (both inbox and junk folders) for updates on the status of
your application from Microsoft.
Purchase a commitment tier pricing plan for
disconnected containers
Create a new resource
\n2. Enter the applicable information to create your resource. Be sure to select
Commitment tier disconnected containers as your pricing tier.
3. Select Review + Create at the bottom of the page. Review the information, and
select Create.
See the following documentation for steps on downloading and configuring the
container for disconnected usage:
Vision - Read
Language Understanding (LUIS)
Text Translation (Standard)
Document Intelligence
Speech service
Speech to text
Custom Speech to text
Neural Text to speech
Language service
Sentiment Analysis
Key Phrase Extraction
Language Detection
Named Entity Recognition
Personally Identifiable Information (PII) detection
Conversational Language Understanding (CLU)
７ Note
You only see the option to purchase a commitment tier if you have been
approved by Microsoft.
Pricing details are only examples.
Configure container for disconnected usage
Environment variable names in Kubernetes
deployments
\nSome Azure AI Containers, for example Translator, require users to pass environmental
variable names that include colons ( : ) when running the container. This works fine
when using Docker, but Kubernetes doesn't accept colons in environmental variable
names. To resolve this, you can replace colons with double underscore characters ( __ )
when deploying to Kubernetes. See the following example of an acceptable format for
environment variable names:
Kubernetes
This example replaces the default format for the Mounts:License  and Mounts:Output
environment variable names in the docker run command.
Container license files are used as keys to decrypt certain files within each container
image. If these encrypted files happen to be updated within a new container image, the
license file you have may fail to start the container even if it worked with the previous
version of the container image. To avoid this issue, we recommend that you download a
new license file from the resource endpoint for your container provided in Azure portal
after you pull new image versions from mcr.microsoft.com.
To download a new license file, you can add DownloadLicense=True  to your docker run
command along with a license mount, your API Key, and your billing endpoint. Refer to
your container's documentation for detailed instructions.
When operating Docker containers in a disconnected environment, the container writes
usage records to a volume where they're collected over time. You can also call a REST
endpoint to generate a report about service usage.
When run in a disconnected environment, an output mount must be available to the
container to store usage logs. For example, you would include -v /host/output:
        env:  
        - name: Mounts__License
          value: "/license"
        - name: Mounts__Output
          value: "/output"
Container image and license updates
Usage records
Arguments for storing logs
\n{OUTPUT_PATH}  and Mounts:Output={OUTPUT_PATH}  in the example below, replacing
{OUTPUT_PATH}  with the path where the logs are stored:
Docker
The container provides two endpoints for returning records about its usage.
The following endpoint provides a report summarizing all of the usage collected in the
mounted billing record directory.
HTTP
It returns JSON similar to the example below.
JSON
The following endpoint provides a report summarizing usage over a specific month and
year.
HTTP
docker run -v /host/output:{OUTPUT_PATH} ... <image> ... Mounts:Output=
{OUTPUT_PATH}
Get records using the container endpoints
Get all records
https://<service>/records/usage-logs/
{
  "apiType": "noop",
  "serviceName": "noop",
  "meters": [
    {
      "name": "Sample.Meter",
      "quantity": 253
    }
  ]
}
Get records for a specific month
\nIt returns a JSON response similar to the example below:
JSON
Commitment plans for disconnected containers have a calendar year commitment
period. When you purchase a plan, you are charged the full price immediately. During
the commitment period, you can't change your commitment plan, however you can
purchase more units at a pro-rated price for the remaining days in the year. You have
until midnight (UTC) on the last day of your commitment, to end a commitment plan.
You can choose a different commitment plan in the Commitment Tier pricing settings
of your resource.
If you decide that you don't want to continue purchasing a commitment plan, you can
set your resource's auto-renewal to Do not auto-renew. Your commitment plan expires
on the displayed commitment end date. After this date, you won't be charged for the
commitment plan. You are able to continue using the Azure resource to make API calls,
charged at pay-as-you-go pricing. You have until midnight (UTC) on the last day of the
year to end a commitment plan for disconnected containers, and not be charged for the
following year.
https://<service>/records/usage-logs/{MONTH}/{YEAR}
{
  "apiType": "string",
  "serviceName": "string",
  "meters": [
    {
      "name": "string",
      "quantity": 253
    }
  ]
}
Purchase a commitment plan to use containers
in disconnected environments
End a commitment plan
Troubleshooting