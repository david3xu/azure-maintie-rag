Placeholder
Value
Example
{ENDPOINT}
The endpoint for authenticating your API
request.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT-
NAME}
The name for your project. This value is
case-sensitive.
myProject
{API-
VERSION}
The version of the API you are calling. The
value referenced here is for the latest
version released. See Model lifecycle to
learn more about other available API
versions.
2022-05-01
Use the following header to authenticate your request.
Key
Value
Ocp-Apim-Subscription-Key
The key to your resource. Used for authenticating your API requests.
JSON
{API-VERSION}
ﾉ
Expand table
Headers
ﾉ
Expand table
Response body
    {
        "createdDateTime": "2021-10-19T23:24:41.572Z",
        "lastModifiedDateTime": "2021-10-19T23:24:41.572Z",
        "lastTrainedDateTime": "2021-10-19T23:24:41.572Z",
        "lastDeployedDateTime": "2021-10-19T23:24:41.572Z",
        "projectKind": "CustomEntityRecognition",
        "storageInputContainerName": "{CONTAINER-NAME}",
        "projectName": "{PROJECT-NAME}",
        "multilingual": false,
        "description": "Project description",
        "language": "{LANGUAGE-CODE}"
    }
\nRepeat the same steps for your replicated project using {SECONDARY-ENDPOINT}  and {SECONDARY-
RESOURCE-KEY} . Compare the returned lastModifiedDateTime  from both projects. If your
primary project was modified sooner than your secondary one, you need to repeat the steps of
exporting, importing, training and deploying.
In this article, you have learned how to use the export and import APIs to replicate your project
to a secondary Language resource in other region. Next, explore the API reference docs to see
what else you can do with authoring APIs.
Authoring REST API reference
Runtime prediction REST API reference
Next steps
\nInstall and run Custom Named Entity
Recognition containers
06/30/2025
Containers enable you to host the Custom Named Entity Recognition API on your own
infrastructure using your own trained model. If you have security or data governance
requirements that can't be fulfilled by calling Custom Named Entity Recognition remotely, then
containers might be a good option.
If you don't have an Azure subscription, create a free account
.
Docker
 installed on a host computer. Docker must be configured to allow the
containers to connect with and send billing data to Azure.
On Windows, Docker must also be configured to support Linux containers.
You should have a basic understanding of Docker concepts
.
A Language resource 
with the free (F0) or standard (S) pricing tier
.
A trained and deployed Custom Named Entity Recognition model
Three primary parameters for all Azure AI containers are required. The Microsoft Software
License Terms must be present with a value of accept. An Endpoint URI and API key are also
needed.
The {ENDPOINT_URI}  value is available on the Azure portal Overview page of the corresponding
Azure AI services resource. Go to the Overview page, hover over the endpoint, and a Copy to
clipboard ＝ icon appears. Copy and use the endpoint where needed.
７ Note
The free account is limited to 5,000 text records per month and only the Free and
Standard pricing tiers
 are valid for containers. For more information on
transaction request rates, see Data and service limits.
Prerequisites
Gather required parameters
Endpoint URI
\nThe {API_KEY}  value is used to start the container and is available on the Azure portal's Keys
page of the corresponding Azure AI services resource. Go to the Keys page, and select the
Copy to clipboard ＝ icon.
The host is an x64-based computer that runs the Docker container. It can be a computer on
your premises or a Docker hosting service in Azure, such as:
Keys
） Important
These subscription keys are used to access your Azure AI services API. Don't share your
keys. Store them securely. For example, use Azure Key Vault. We also recommend that you
regenerate these keys regularly. Only one key is necessary to make an API call. When you
regenerate the first key, you can use the second key for continued access to the service.
Host computer requirements and
recommendations
\n![Image](images/page734_image1.png)

![Image](images/page734_image2.png)
\nAzure Kubernetes Service.
Azure Container Instances.
A Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
The following table describes the minimum and recommended specifications for Custom
Named Entity Recognition containers. Each CPU core must be at least 2.6 gigahertz (GHz) or
faster. The allowable Transactions Per Second (TPS) are also listed.
Minimum host
specs
Recommended host
specs
Minimum
TPS
Maximum
TPS
Custom Named Entity
Recognition
1 core, 2 GB
memory
1 core, 4 GB memory
15
30
CPU core and memory correspond to the --cpus  and --memory  settings, which are used as part
of the docker run  command.
Before you proceed with running the docker image, you will need to export your own trained
model to expose it to your container. Use the following command to extract your model and
replace the placeholders below with your own values:
Placeholder
Value
Format or example
{API_KEY}
The key for your Custom
Named Entity Recognition
resource. You can find it on
your resource's Key and
endpoint page, on the Azure
portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
{ENDPOINT_URI}
The endpoint for accessing
the Custom Named Entity
Recognition API. You can find
it on your resource's Key and
endpoint page, on the Azure
portal.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
ﾉ
Expand table
Export your Custom Named Entity Recognition
model
ﾉ
Expand table
\nPlaceholder
Value
Format or example
{PROJECT_NAME}
The name of the project
containing the model that you
want to export. You can find it
on your projects tab in the
Language Studio portal.
myProject
{TRAINED_MODEL_NAME}
The name of the trained
model you want to export.
You can find your trained
models on your model
evaluation tab under your
project in the Language
Studio portal.
myTrainedModel
Bash
The Custom Named Entity Recognition container image can be found on the
mcr.microsoft.com  container registry syndicate. It resides within the azure-cognitive-
services/textanalytics/  repository and is named customner . The fully qualified container
image name is, mcr.microsoft.com/azure-cognitive-services/textanalytics/customner .
To use the latest version of the container, you can use the latest  tag. You can also find a full
list of tags on the MCR
.
Use the docker pull
 command to download a container image from Microsoft Container
Registry.
curl --location --request PUT '{ENDPOINT_URI}/language/authoring/analyze-
text/projects/{PROJECT_NAME}/exported-models/{TRAINED_MODEL_NAME}?api-
version=2023-04-15-preview' \
--header 'Ocp-Apim-Subscription-Key: {API_KEY}' \
--header 'Content-Type: application/json' \
--data-raw '{
    "TrainedmodelLabel": "{TRAINED_MODEL_NAME}"
}'
Get the container image with docker pull
docker pull mcr.microsoft.com/azure-cognitive-
services/textanalytics/customner:latest
\nOnce the container is on the host computer, use the docker run
 command to run the
containers. The container will continue to run until you stop it.
To run the Custom Named Entity Recognition container, execute the following docker run
command. Replace the placeholders below with your own values:
Placeholder
Value
Format or example
{API_KEY}
The key for your Custom
Named Entity Recognition
resource. You can find it on
your resource's Key and
endpoint page, on the Azure
portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
 Tip
You can use the docker images
 command to list your downloaded container images.
For example, the following command lists the ID, repository, and tag of each downloaded
container image, formatted as a table:
docker images --format "table {{.ID}}\t{{.Repository}}\t{{.Tag}}"
IMAGE ID         REPOSITORY                TAG
<image-id>       <repository-path/name>    <tag-name>
Run the container with docker run
） Important
The docker commands in the following sections use the back slash, \ , as a line
continuation character. Replace or remove this based on your host operating
system's requirements.
The Eula , Billing , and ApiKey  options must be specified to run the container;
otherwise, the container won't start. For more information, see Billing.
ﾉ
Expand table
\nPlaceholder
Value
Format or example
{ENDPOINT_URI}
The endpoint for accessing
the Custom Named Entity
Recognition API. You can find
it on your resource's Key and
endpoint page, on the Azure
portal.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{PROJECT_NAME}
The name of the project
containing the model that you
want to export. You can find it
on your projects tab in the
Language Studio portal.
myProject
{LOCAL_PATH}
The path where the exported
model in the previous step
will be downloaded in. You
can choose any path of your
liking.
C:/custom-ner-model
{TRAINED_MODEL_NAME}
The name of the trained
model you want to export.
You can find your trained
models on your model
evaluation tab under your
project in the Language
Studio portal.
myTrainedModel
Bash
This command:
Runs a Custom Named Entity Recognition container and downloads your exported model
to the local path specified.
Allocates one CPU core and 4 gigabytes (GB) of memory
Exposes TCP port 5000 and allocates a pseudo-TTY for the container
Automatically removes the container after it exits. The container image is still available on
the host computer.
docker run --rm -it -p5000:5000  --memory 4g --cpus 1 \
-v {LOCAL_PATH}:/modelPath \
mcr.microsoft.com/azure-cognitive-services/textanalytics/customner:latest \
EULA=accept \
BILLING={ENDPOINT_URI} \
APIKEY={API_KEY} \
projectName={PROJECT_NAME}
exportedModelName={TRAINED_MODEL_NAME}
\nIf you intend to run multiple containers with exposed ports, make sure to run each container
with a different exposed port. For example, run the first container on port 5000 and the second
container on port 5001.
You can have this container and a different Azure AI services container running on the HOST
together. You also can have multiple containers of the same Azure AI services container
running.
The container provides REST-based query prediction endpoint APIs.
Use the host, http://localhost:5000 , for container APIs.
There are several ways to validate that the container is running. Locate the External IP address
and exposed port of the container in question, and open your favorite web browser. Use the
various request URLs that follow to validate the container is running. The example request URLs
listed here are http://localhost:5000 , but your specific container might vary. Make sure to rely
on your container's External IP address and exposed port.
Request URL
Purpose
http://localhost:5000/
The container provides a home page.
http://localhost:5000/ready
Requested with GET, this URL provides a verification that the container
is ready to accept a query against the model. This request can be used
for Kubernetes liveness and readiness probes
.
http://localhost:5000/status
Also requested with GET, this URL verifies if the api-key used to start
the container is valid without causing an endpoint query. This request
can be used for Kubernetes liveness and readiness probes
.
http://localhost:5000/swagger
The container provides a full set of documentation for the endpoints
and a Try it out feature. With this feature, you can enter your settings
into a web-based HTML form and make the query without having to
write any code. After the query returns, an example CURL command is
provided to demonstrate the HTTP headers and body format that's
required.
Run multiple containers on the same host
Query the container's prediction endpoint
Validate that a container is running
ﾉ
Expand table
\nTo shut down the container, in the command-line environment where the container is running,
select Ctrl+C .
If you run the container with an output mount and logging enabled, the container generates
log files that are helpful to troubleshoot issues that happen while starting or running the
container.
The Custom Named Entity Recognition containers send billing information to Azure, using a
Custom Named Entity Recognition resource on your Azure account.
Queries to the container are billed at the pricing tier of the Azure resource that's used for the
ApiKey  parameter.
Stop the container
Troubleshooting
 Tip
For more troubleshooting information and guidance, see Azure AI containers frequently
asked questions (FAQ).
Billing
\n![Image](images/page740_image1.png)