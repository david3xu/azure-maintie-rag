The {API_KEY}  value is used to start the container and is available on the Azure portal's Keys
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
\n![Image](images/page1251_image1.png)

![Image](images/page1251_image2.png)
\nAzure Kubernetes Service.
Azure Container Instances.
A Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
The following table describes the minimum and recommended specifications for the available
container. Each CPU core must be at least 2.6 gigahertz (GHz) or faster. The allowable
Transactions Per Second (TPS) are also listed.
Minimum host
specs
Recommended host
specs
Minimum
TPS
Maximum
TPS
Sentiment
Analysis
1 core, 2GB memory
4 cores, 8GB memory
15
30
CPU core and memory correspond to the --cpus  and --memory  settings, which are used as part
of the docker run  command.
The Sentiment Analysis container image can be found on the mcr.microsoft.com  container
registry syndicate. It resides within the azure-cognitive-services/textanalytics/  repository
and is named sentiment . The fully qualified container image name is,
mcr.microsoft.com/azure-cognitive-services/textanalytics/sentiment
To use the latest version of the container, you can use the latest  tag, which is for English. You
can also find a full list of containers for supported languages using the tags on the MCR
.
The sentiment analysis container v3 container is available in several languages. To download
the container for the English container, use the command below.
ﾉ
Expand table
Get the container image with docker pull
docker pull mcr.microsoft.com/azure-cognitive-
services/textanalytics/sentiment:3.0-en
 Tip
You can use the docker images
 command to list your downloaded container images.
For example, the following command lists the ID, repository, and tag of each downloaded
\nOnce the container is on the host computer, use the docker run
 command to run the
containers. The container will continue to run until you stop it.
To run the Sentiment Analysis container, execute the following docker run  command. Replace
the placeholders below with your own values:
Placeholder
Value
Format or example
{API_KEY}
The key for your Language resource.
You can find it on your resource's Key
and endpoint page, on the Azure
portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
{ENDPOINT_URI}
The endpoint for accessing the API.
You can find it on your resource's Key
and endpoint page, on the Azure
portal.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
{IMAGE_TAG}
The image tag representing the
language of the container you want
to run. Make sure this matches the
docker pull  command you used.
3.0-en
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
\nBash
This command:
Runs a Sentiment Analysis container from the container image
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
The container provides REST-based query prediction endpoint APIs.
Use the host, http://localhost:5000 , for container APIs.
There are several ways to validate that the container is running. Locate the External IP address
and exposed port of the container in question, and open your favorite web browser. Use the
various request URLs that follow to validate the container is running. The example request URLs
listed here are http://localhost:5000 , but your specific container might vary. Make sure to rely
on your container's External IP address and exposed port.
docker run --rm -it -p 5000:5000 --memory 8g --cpus 1 \
mcr.microsoft.com/azure-cognitive-services/textanalytics/sentiment:{IMAGE_TAG} \
Eula=accept \
Billing={ENDPOINT_URI} \
ApiKey={API_KEY}
Run multiple containers on the same host
Query the container's prediction endpoint
Validate that a container is running
ﾉ
Expand table
\nRequest URL
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
If you have been approved to run the container disconnected from the internet, use the
following example shows the formatting of the docker run  command you'll use, with
placeholder values. Replace these placeholder values with your own values.
Run the container disconnected from the internet
\n![Image](images/page1255_image1.png)
\nThe DownloadLicense=True  parameter in your docker run  command will download a license file
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
downloaded, and
mounted.
/host/license:/path/to/license/directory
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
ﾉ
Expand table
docker run --rm -it -p 5000:5000 \ 
-v {LICENSE_MOUNT} \
{IMAGE} \
eula=accept \
billing={ENDPOINT_URI} \
apikey={API_KEY} \
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
DownloadLicense=True \
Mounts:License={CONTAINER_LICENSE_DIRECTORY} 
ﾉ
Expand table
\nBash
To shut down the container, in the command-line environment where the container is running,
select Ctrl+C .
If you run the container with an output mount and logging enabled, the container generates
log files that are helpful to troubleshoot issues that happen while starting or running the
container.
The Sentiment Analysis containers send billing information to Azure, using a Language
resource on your Azure account.
Queries to the container are billed at the pricing tier of the Azure resource that's used for the
ApiKey  parameter.
Azure AI services containers aren't licensed to run without being connected to the metering or
billing endpoint. You must enable the containers to communicate billing information with the
billing endpoint at all times. Azure AI services containers don't send customer data, such as the
image or text that's being analyzed, to Microsoft.
docker run --rm -it -p 5000:5000 --memory {MEMORY_SIZE} --cpus {NUMBER_CPUS} \ 
-v {LICENSE_MOUNT} \ 
-v {OUTPUT_PATH} \
{IMAGE} \
eula=accept \
Mounts:License={CONTAINER_LICENSE_DIRECTORY}
Mounts:Output={CONTAINER_OUTPUT_DIRECTORY}
Stop the container
Troubleshooting
 Tip
For more troubleshooting information and guidance, see Azure AI containers frequently
asked questions (FAQ).
Billing
Connect to Azure
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
Sentiment Analysis containers. In summary:
Sentiment Analysis provides Linux containers for Docker
Container images are downloaded from the Microsoft Container Registry (MCR).
Container images run in Docker.
You must specify billing information when instantiating a container.
Billing arguments
ﾉ
Expand table
Summary
） Important
Azure AI containers are not licensed to run without being connected to Azure for
metering. Customers need to enable the containers to communicate billing information
\nSee Configure containers for configuration settings.
with the metering service at all times. Azure AI containers do not send customer data (for
example, text that is being analyzed) to Microsoft.
Next steps