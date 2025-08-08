Placeholder
Value
Format or example
{PROJECT_NAME}
The name of the project
containing the model that
you want to export. You can
find it on your projects tab in
the Language Studio portal.
myProject
{TRAINED_MODEL_NAME}
The name of the trained
model you want to export.
You can find your trained
models on your model
evaluation tab under your
project in the Language
Studio portal
myTrainedModel
{EXPORTED_MODEL_NAME}
The name to assign for the
new exported model created.
myExportedModel
Bash
The CLU container image can be found on the mcr.microsoft.com  container registry syndicate.
It resides within the azure-cognitive-services/language/  repository and is named clu . The
fully qualified container image name is, mcr.microsoft.com/azure-cognitive-
services/language/clu
To use the latest version of the container, you can use the latest  tag, which is for English. You
can also find a full list of containers for supported languages using the tags on the MCR
.
The latest CLU container is available in several languages. To download the container for the
English container, use the command below.
curl --location --request PUT '{ENDPOINT_URI}/language/authoring/analyze-
conversations/projects/{PROJECT_NAME}/exported-models/{EXPORTED_MODEL_NAME}?api-
version=2024-11-15-preview' \ 
--header 'Ocp-Apim-Subscription-Key: {API_KEY}' \ 
--header 'Content-Type: application/json' \ 
--data-raw '{ 
    "TrainedModelLabel": "{TRAINED_MODEL_NAME}" 
}' 
Get the container image with docker pull
\nAfter creating the exported model in the section above, users have to run the container in
order to download the deployment package that was created specifically for their exported
models.
Placeholder
Value
Format or example
{API_KEY} 
The key for your
Language resource. You
can find it on your
resource's Key and
endpoint page, on the
Azure portal.  
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
 
{ENDPOINT_URI} 
The endpoint for
accessing the API. You
can find it on your
resource's Key and
endpoint page, on the
Azure portal. 
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{IMAGE_TAG} 
The image tag
representing the
language of the
container you want to
run. Make sure this
latest 
docker pull mcr.microsoft.com/azure-cognitive-services/language/clu:latest
 Tip
You can use the docker images
 command to list your downloaded container images.
For example, the following command lists the ID, repository, and tag of each downloaded
container image, formatted as a table:
docker images --format "table {{.ID}}\t{{.Repository}}\t{{.Tag}}"
IMAGE ID         REPOSITORY                TAG
<image-id>       <repository-path/name>    <tag-name>
Run the container in download model mode
ﾉ
Expand table
\nPlaceholder
Value
Format or example
matches the  docker
pull  command you used. 
{LOCAL_CLU_PORT}
Port number assigned for
the container in local
machine.
5000
{LOCAL_MODEL_DIRECTORY}
Absolute directory in
host machine where
exported models are
saved in.
C:\usr\local\myDeploymentPackage
{PROJECT_NAME}
Name of the project that
the exported model
belongs to
myProject
{EXPORTED_MODEL_NAME}
Exported model to be
downloaded
myExportedModel
Bash
DO NOT alter the downloaded files. Even altering the name or folder structure can affect the
integrity of the container and might break it.
Repeat those steps to download as many models as you'd like to test. They can belong to
different projects and have different exported model names.
Once the container is on the host computer, use the docker run
 command to run the
containers. The container continues to run until you stop it. Replace the placeholders below
with your own values:
docker run --rm -it -p {LOCAL_CLU_PORT}:80 \ 
mcr.microsoft.com/azure-cognitive-services/language/clu:{IMAGE_TAG} \   
-v {LOCAL_MODEL_DIRECTORY}:/DeploymentPackage \ 
Billing={ENDPOINT_URI} \   
ApiKey={API_KEY} \ 
downloadmodel \ 
projectName={PROJECT_NAME} \ 
exportedModelName={EXPORTED_MODEL_NAME} 
Run the container with docker run
） Important
\nTo run the CLU container, execute the following docker run  command. Replace the
placeholders below with your own values:
Placeholder
Value
Format or example
{API_KEY}
The key for your
Language resource. You
can find it on your
resource's Key and
endpoint page, on the
Azure portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
{ENDPOINT_URI}
The endpoint for
accessing the API. You
can find it on your
resource's Key and
endpoint page, on the
Azure portal.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{IMAGE_TAG}
The image tag
representing the
language of the
container you want to
run. Make sure this
matches the docker pull
command you used.
latest
{LOCAL_CLU_PORT}
Port number assigned for
the container in local
machine.
5000
{LOCAL_NER_PORT}
Port number of the NER
container. See Run NER
Container section below.
5001 (Has to be different that the above
port number)
{LOCAL_LOGGING_DIRECTORY}
Absolute directory in
host machine where that
logs are saved in.
C:\usr\local\mylogs
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
{LOCAL_MODEL_DIRECTORY}
Absolute directory in
host machine where
exported models are
saved in.
C:\usr\local\myDeploymentPackage
Bash
This command:
Runs a CLU container from the container image
Allocates one CPU core and 8 gigabytes (GB) of memory
Exposes TCP port 5000 and allocates a pseudo-TTY for the container
Automatically removes the container after it exits. The container image is still available on
the host computer.
If you intend to run multiple containers with exposed ports, make sure to run each container
with a different exposed port. For example, run the first container on port 5000 and the second
container on port 5001.
You can have this container and a different Azure AI services container running on the HOST
together. You also can have multiple containers of the same Azure AI services container
running.
CLU relies on NER to handle prebuilt entities. The CLU container works properly without NER if
users decide not to integrate it. NER billing is disabled when it’s used through CLU, no extra
charges are generated unless a call is made directly to NER’s container.
To set up NER in CLU container
Follow the NER container documentation.
docker run --rm -it -p 5000:5000 --memory 8g --cpus 1 \
mcr.microsoft.com/azure-cognitive-services/language/clu:{IMAGE_TAG} \
Eula=accept \
Billing={ENDPOINT_URI} \
ApiKey={API_KEY}
Run multiple containers on the same host
Running NER Container
\nWhen running CLU container, make sure to set the parameter Ner_Url so that
Ner_Url=http://host.docker.internal:{LOCAL_NER_PORT}
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
Query the container's prediction endpoint
Validate that a container is running
ﾉ
Expand table
\nFor information on how to call CLU see our guide.
To use this container disconnected from the internet, you must first request access by filling
out an application, and purchasing a commitment plan. See Use Docker containers in
disconnected environments for more information.
If you have been approved to run the container disconnected from the internet, use the
following example shows the formatting of the docker run  command you'll use, with
placeholder values. Replace these placeholder values with your own values.
The DownloadLicense=True  parameter in your docker run  command will download a license file
that will enable your Docker container to run when it isn't connected to the internet. It also
contains an expiration date, after which the license file will be invalid to run the container. You
can only use a license file with the appropriate container that you've been approved for. For
example, you can't use a license file for a speech to text container with a Document
Intelligence container.
Placeholder
Value
Format or example
{IMAGE}
The container image
you want to use.
mcr.microsoft.com/azure-cognitive-
services/form-recognizer/invoice
{LICENSE_MOUNT}
The path where the
license will be
/host/license:/path/to/license/directory
Run the container disconnected from the internet
ﾉ
Expand table
\n![Image](images/page217_image1.png)
\nPlaceholder
Value
Format or example
downloaded, and
mounted.
{ENDPOINT_URI}
The endpoint for
authenticating your
service request. You can
find it on your
resource's Key and
endpoint page, on the
Azure portal.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{API_KEY}
The key for your Text
Analytics resource. You
can find it on your
resource's Key and
endpoint page, on the
Azure portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
{CONTAINER_LICENSE_DIRECTORY}
Location of the license
folder on the
container's local
filesystem.
/path/to/license/directory
Bash
Once the license file has been downloaded, you can run the container in a disconnected
environment. The following example shows the formatting of the docker run  command you'll
use, with placeholder values. Replace these placeholder values with your own values.
Wherever the container is run, the license file must be mounted to the container and the
location of the license folder on the container's local filesystem must be specified with
Mounts:License= . An output mount must also be specified so that billing usage records can be
written.
docker run --rm -it -p 5000:5000 \ 
-v {LICENSE_MOUNT} \
{IMAGE} \
eula=accept \
billing={ENDPOINT_URI} \
apikey={API_KEY} \
DownloadLicense=True \
Mounts:License={CONTAINER_LICENSE_DIRECTORY} 
ﾉ
Expand table
\nPlaceholder
Value
Format or example
{IMAGE}
The container image
you want to use.
mcr.microsoft.com/azure-cognitive-
services/form-recognizer/invoice
{MEMORY_SIZE}
The appropriate size
of memory to
allocate for your
container.
4g
{NUMBER_CPUS}
The appropriate
number of CPUs to
allocate for your
container.
4
{LICENSE_MOUNT}
The path where the
license will be
located and
mounted.
/host/license:/path/to/license/directory
{OUTPUT_PATH}
The output path for
logging usage
records.
/host/output:/path/to/output/directory
{CONTAINER_LICENSE_DIRECTORY}
Location of the
license folder on the
container's local
filesystem.
/path/to/license/directory
{CONTAINER_OUTPUT_DIRECTORY}
Location of the
output folder on the
container's local
filesystem.
/path/to/output/directory
Bash
To shut down the container, in the command-line environment where the container is running,
select Ctrl+C .
docker run --rm -it -p 5000:5000 --memory {MEMORY_SIZE} --cpus {NUMBER_CPUS} \ 
-v {LICENSE_MOUNT} \ 
-v {OUTPUT_PATH} \
{IMAGE} \
eula=accept \
Mounts:License={CONTAINER_LICENSE_DIRECTORY}
Mounts:Output={CONTAINER_OUTPUT_DIRECTORY}
Stop the container
\nIf you run the container with an output mount and logging enabled, the container generates
log files that are helpful to troubleshoot issues that happen while starting or running the
container.
The CLU containers send billing information to Azure, using a Language resource on your Azure
account.
Queries to the container are billed at the pricing tier of the Azure resource that's used for the
ApiKey  parameter.
Azure AI services containers aren't licensed to run without being connected to the metering or
billing endpoint. You must enable the containers to communicate billing information with the
billing endpoint at all times. Azure AI services containers don't send customer data, such as the
image or text that's being analyzed, to Microsoft.
The container needs the billing argument values to run. These values allow the container to
connect to the billing endpoint. The container reports usage about every 10 to 15 minutes. If
the container doesn't connect to Azure within the allowed time window, the container
continues to run but doesn't serve queries until the billing endpoint is restored. The connection
is attempted 10 times at the same time interval of 10 to 15 minutes. If it can't connect to the
billing endpoint within the 10 tries, the container stops serving requests. See the Azure AI
services container FAQ for an example of the information sent to Microsoft for billing.
The docker run Ｍ
 command will start the container when all three of the following options
are provided with valid values:
Troubleshooting
 Tip
For more troubleshooting information and guidance, see Azure AI containers frequently
asked questions (FAQ).
Billing
Connect to Azure
Billing arguments
ﾉ
Expand table