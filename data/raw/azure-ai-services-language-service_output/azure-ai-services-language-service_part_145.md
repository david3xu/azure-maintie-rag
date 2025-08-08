Azure Kubernetes Service.
Azure Container Instances.
A Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
The following table describes the minimum and recommended specifications for the
summarization container skills. Listed CPU/memory combinations are for a 4000 token input
(conversation consumption is for all the aspects in the same request).
Container Type
Recommended number
of CPU cores
Recommended
memory
Notes
Summarization CPU
container
16
48 GB
Summarization GPU
container
2
24 GB
Requires an NVIDIA GPU that
supports Cuda 11.8 with 16GB
VRAM.
CPU core and memory correspond to the --cpus  and --memory  settings, which are used as part
of the docker run  command.
The Summarization container image can be found on the mcr.microsoft.com  container registry
syndicate. It resides within the azure-cognitive-services/textanalytics/  repository and is
named summarization . The fully qualified container image name is, mcr.microsoft.com/azure-
cognitive-services/textanalytics/summarization
To use the latest version of the container, you can use the latest  tag. You can also find a full
list of tags on the MCR
.
Use the docker pull
 command to download a container image from the Microsoft Container
Registry.
for CPU containers,
ﾉ
Expand table
Get the container image with docker pull
docker pull mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu
\nfor GPU containers.
A pre-requisite for running the summarization container is to download the models first. This
can be done by running one of the following commands using a CPU container image as an
example:
Bash
It's not recommended to download models for all skills inside the same HOST_MODELS_PATH , as
the container loads all models inside the HOST_MODELS_PATH . Doing so would use a large
amount of memory. It's recommended to only download the model for the skill you need in a
particular HOST_MODELS_PATH .
docker pull mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:gpu
 Tip
You can use the docker images
 command to list your downloaded container images.
For example, the following command lists the ID, repository, and tag of each downloaded
container image, formatted as a table:
docker images --format "table {{.ID}}\t{{.Repository}}\t{{.Tag}}"
IMAGE ID         REPOSITORY                TAG
<image-id>       <repository-path/name>    <tag-name>
Download the summarization container models
docker run -v {HOST_MODELS_PATH}:/models mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu downloadModels=ExtractiveSummarization 
billing={ENDPOINT_URI} apikey={API_KEY}
docker run -v {HOST_MODELS_PATH}:/models mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu downloadModels=AbstractiveSummarization 
billing={ENDPOINT_URI} apikey={API_KEY}
docker run -v {HOST_MODELS_PATH}:/models mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu downloadModels=ConversationSummarization 
billing={ENDPOINT_URI} apikey={API_KEY}
\nIn order to ensure compatibility between models and the container, re-download the utilized
models whenever you create a container using a new image version. When using a
disconnected container, the license should be downloaded again after downloading the
models.
Once the Summarization container is on the host computer, use the following docker run
command to run the containers. The container will continue to run until you stop it. Replace
the placeholders below with your own values:
Placeholder
Value
Format or example
{HOST_MODELS_PATH}
The host computer volume
mount
, which Docker uses
to persist the model.
An example is c:\SummarizationModel where
the c:\ drive is located on the host machine.
{ENDPOINT_URI}
The endpoint for accessing
the summarization API. You
can find it on your resource's
Key and endpoint page, on
the Azure portal.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{API_KEY}
The key for your Language
resource. You can find it on
your resource's Key and
endpoint page, on the Azure
portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Bash
Or if you are running a GPU container, use this command instead.
Bash
Run the container with docker run
ﾉ
Expand table
docker run -p 5000:5000 -v {HOST_MODELS_PATH}:/models mcr.microsoft.com/azure-
cognitive-services/textanalytics/summarization:cpu eula=accept rai_terms=accept 
billing={ENDPOINT_URI} apikey={API_KEY}
docker run -p 5000:5000 --gpus all -v {HOST_MODELS_PATH}:/models 
mcr.microsoft.com/azure-cognitive-services/textanalytics/summarization:gpu 
eula=accept rai_terms=accept billing={ENDPOINT_URI} apikey={API_KEY}
\nIf there is more than one GPU on the machine, replace --gpus all  with --gpus device=
{DEVICE_ID} .
This command:
Runs a Summarization container from the container image
Allocates one CPU core and 4 gigabytes (GB) of memory
Exposes TCP port 5000 and allocates a pseudo-TTY for the container
Automatically removes the container after it exits. The container image is still available on
the host computer.
If you intend to run multiple containers with exposed ports, make sure to run each container
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
） Important
The docker commands in the following sections use the back slash, \ , as a line
continuation character. Replace or remove this based on your host operating
system's requirements.
The Eula , Billing , rai_terms  and ApiKey  options must be specified to run the
container; otherwise, the container won't start. For more information, see Billing.
Run multiple containers on the same host
Query the container's prediction endpoint
Validate that a container is running
\nlisted here are http://localhost:5000 , but your specific container might vary. Make sure to rely
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
To use this container disconnected from the internet, you must first request access by filling
out an application, and purchasing a commitment plan. See Use Docker containers in
disconnected environments for more information.
ﾉ
Expand table
Run the container disconnected from the internet
\n![Image](images/page1445_image1.png)
\nIf you have been approved to run the container disconnected from the internet, use the
following example shows the formatting of the docker run  command you'll use, with
placeholder values. Replace these placeholder values with your own values.
The DownloadLicense=True  parameter in your docker run  command will download a license file
that will enable your Docker container to run when it isn't connected to the internet. It also
contains an expiration date, after which the license file will be invalid to run the container. You
can only use a license file with the appropriate container that you've been approved for. For
example, you can't use a license file for a speech to text container with a Language services
container.
A pre-requisite for running the summarization container is to download the models first. This
can be done by running one of the following commands using a CPU container image as an
example:
Bash
It's not recommended to download models for all skills inside the same HOST_MODELS_PATH , as
the container loads all models inside the HOST_MODELS_PATH . Doing so would use a large
amount of memory. It's recommended to only download the model for the skill you need in a
particular HOST_MODELS_PATH .
In order to ensure compatibility between models and the container, re-download the utilized
models whenever you create a container using a new image version. When using a
disconnected container, the license should be downloaded again after downloading the
models.
Download the summarization disconnected
container models
docker run -v {HOST_MODELS_PATH}:/models mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu downloadModels=ExtractiveSummarization 
billing={ENDPOINT_URI} apikey={API_KEY}
docker run -v {HOST_MODELS_PATH}:/models mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu downloadModels=AbstractiveSummarization 
billing={ENDPOINT_URI} apikey={API_KEY}
docker run -v {HOST_MODELS_PATH}:/models mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu downloadModels=ConversationSummarization 
billing={ENDPOINT_URI} apikey={API_KEY}
Run the disconnected container with docker run
\nPlaceholder
Value
Format or example
{IMAGE}
The container image
you want to use.
mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu
{LICENSE_MOUNT}
The path where the
license will be
downloaded, and
mounted.
/host/license:/path/to/license/directory
{HOST_MODELS_PATH}
The path where the
models were
downloaded, and
mounted.
/host/models:/models
{ENDPOINT_URI}
The endpoint for
authenticating your
service request. You
can find it on your
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
ﾉ
Expand table
docker run --rm -it -p 5000:5000 \ 
-v {LICENSE_MOUNT} \
-v {HOST_MODELS_PATH} \
{IMAGE} \
eula=accept \
rai_terms=accept \
billing={ENDPOINT_URI} \
apikey={API_KEY} \
DownloadLicense=True \
Mounts:License={CONTAINER_LICENSE_DIRECTORY} 
\nOnce the license file has been downloaded, you can run the container in a disconnected
environment. The following example shows the formatting of the docker run  command you'll
use, with placeholder values. Replace these placeholder values with your own values.
Wherever the container is run, the license file must be mounted to the container and the
location of the license folder on the container's local filesystem must be specified with
Mounts:License= . An output mount must also be specified so that billing usage records can be
written.
Placeholder
Value
Format or example
{IMAGE}
The container image
you want to use.
mcr.microsoft.com/azure-cognitive-
services/textanalytics/summarization:cpu
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
{HOST_MODELS_PATH}
The path where the
models were
downloaded, and
mounted.
/host/models:/models
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
ﾉ
Expand table
\nBash
To shut down the container, in the command-line environment where the container is running,
select Ctrl+C .
If you run the container with an output mount and logging enabled, the container generates
log files that are helpful to troubleshoot issues that happen while starting or running the
container.
The summarization containers send billing information to Azure, using a Language resource on
your Azure account.
Queries to the container are billed at the pricing tier of the Azure resource that's used for the
ApiKey  parameter.
Azure AI services containers aren't licensed to run without being connected to the metering or
billing endpoint. You must enable the containers to communicate billing information with the
billing endpoint at all times. Azure AI services containers don't send customer data, such as the
image or text that's being analyzed, to Microsoft.
docker run --rm -it -p 5000:5000 --memory {MEMORY_SIZE} --cpus {NUMBER_CPUS} \ 
-v {LICENSE_MOUNT} \ 
-v {HOST_MODELS_PATH} \
-v {OUTPUT_PATH} \
{IMAGE} \
eula=accept \
rai_terms=accept \
Mounts:License={CONTAINER_LICENSE_DIRECTORY}
Mounts:Output={CONTAINER_OUTPUT_DIRECTORY}
Stop the container
Troubleshooting
 Tip
For more troubleshooting information and guidance, see Azure AI containers frequently
asked questions (FAQ).
Billing
\nThe container needs the billing argument values to run. These values allow the container to
connect to the billing endpoint. The container reports usage about every 10 to 15 minutes. If
the container doesn't connect to Azure within the allowed time window, the container
continues to run but doesn't serve queries until the billing endpoint is restored. The connection
is attempted 10 times at the same time interval of 10 to 15 minutes. If it can't connect to the
billing endpoint within the 10 tries, the container stops serving requests. See the Azure AI
services container FAQ for an example of the information sent to Microsoft for billing.
The docker run Ｍ
 command will start the container when all three of the following options
are provided with valid values:
Option
Description
ApiKey
The API key of the Azure AI services resource that's used to track billing information.
The value of this option must be set to an API key for the provisioned resource that's specified
in Billing .
Billing
The endpoint of the Azure AI services resource that's used to track billing information.
The value of this option must be set to the endpoint URI of a provisioned Azure resource.
Eula
Indicates that you accepted the license for the container.
The value of this option must be set to accept.
For more information about these options, see Configure containers.
In this article, you learned concepts and workflow for downloading, installing, and running
summarization containers. In summary:
Summarization provides Linux containers for Docker
Container images are downloaded from the Microsoft Container Registry (MCR).
Container images run in Docker.
You must specify billing information when instantiating a container.
Connect to Azure
Billing arguments
ﾉ
Expand table
Summary
） Important