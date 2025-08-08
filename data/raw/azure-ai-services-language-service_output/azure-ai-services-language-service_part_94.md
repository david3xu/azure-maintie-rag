Parameter
Subparameters and Descriptions
name
Category, type, and tag to return if there's a regex match.
decription
(optional) User-readable rule description.
regexPatterns
List of regex patterns used to find entities.
- id : Identifier of the regex pattern.
- matchScore : Confidence score for regex matches.
- locales : Languages valid for the regex pattern.
matchcontext
Regex patterns providing context to matched entities. Context matching is a
bidirectional search from the matched entity that increases confidence score in case it's
found. If multiple hints are supporting a match the hint with the highest score is used.
- hints : List of regex patterns giving context to matched entities.
- hintText : Regex pattern providing context to matched entities.
- boostingScore : (optional) Score added to confidence score from a matched entity.
          "locales": [ 
            "en" 
          ] 
        } 
      ], 
      "matchContext": { // patterns to give matches context 
        "hints": [ 
          { 
            "hintText": "ssa(\\s*)number", // regex pattern to find to give a 
match context. 
            "boostingScore": 0.2, // score to boost match confidence if hint is 
found 
            "locales": [ // list of languages valid for this context 
              "en" 
            ] 
          }, 
          { 
            "hintText": "social(\\s*)security(\\s*)(#*)", 
            "boostingScore": 0.2, 
            "locales": [ 
              "en" 
            ] 
          } 
        ], 
      } 
    } 
] 
Overview of each regex recognition file parameter
ﾉ
Expand table
\nParameter
Subparameters and Descriptions
- locales : Language valid for hintText.
- contextLimit : (optional) Distance from the matched entity to search for context.
To display information about the running regexRules , add the following property to enable
debug logging: Logging:Console:LogLevel:Default=Debug
Bash
Rule names must begin with "CE_"
Rule names must be unique.
Rule names may only use alphanumeric characters and underscores ("_")
Regex patterns follow the .NET regular Expressions format. See our documentation on
.NET regular expressions for more information.
Logging
docker run --rm -it -p 5000:5000 --memory 8g --cpus 1 \ 
mcr.microsoft.com/azure-cognitive-services/textanalytics/pii:{IMAGE_TAG} \ 
Eula=accept \ 
Billing={ENDPOINT_URI} \ 
ApiKey={API_KEY} \ 
UserRegexRuleFilePath={REGEX_RULE_FILE_PATH} \ 
Logging:Console:LogLevel:Default=Debug 
Regex rule constraints
\nInstall and run Personally Identifiable
Information (PII) Detection containers
Article • 04/29/2025
Containers enable you to host the PII detection API on your own infrastructure. If you have
security or data governance requirements that can't be fulfilled by calling PII detection
remotely, then containers might be a good option.
If you don't have an Azure subscription, create a free account
 before you begin.
You must meet the following prerequisites before using PII detection containers.
If you don't have an Azure subscription, create a free account
.
Docker
 installed on a host computer. Docker must be configured to allow the
containers to connect with and send billing data to Azure.
On Windows, Docker must also be configured to support Linux containers.
You should have a basic understanding of Docker concepts
.
A Language resource
Three primary parameters for all Azure AI containers are required. The Microsoft Software
License Terms must be present with a value of accept. An Endpoint URI and API key are also
needed.
The {ENDPOINT_URI}  value is available on the Azure portal Overview page of the corresponding
Azure AI services resource. Go to the Overview page, hover over the endpoint, and a Copy to
clipboard ＝ icon appears. Copy and use the endpoint where needed.
７ Note
The data limits in a single synchronous API call for the PII container are 5,120 characters
per document and up to 10 documents per call.
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
\n![Image](images/page934_image1.png)

![Image](images/page934_image2.png)
\nAzure Kubernetes Service.
Azure Container Instances.
A Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
The following table describes the minimum and recommended specifications for the available
container. Each CPU core must be at least 2.6 gigahertz (GHz) or faster.
It's recommended to have a CPU with AVX-512 instruction set, for the best experience
(performance and accuracy).
Minimum host specs
Recommended host specs
PII detection
1 core, 2 GB memory
4 cores, 8 GB memory
CPU core and memory correspond to the --cpus  and --memory  settings, which are used as part
of the docker run  command.
The PII detection container image can be found on the mcr.microsoft.com  container registry
syndicate. It resides within the azure-cognitive-services/textanalytics/  repository and is
named pii . The fully qualified container image name is, mcr.microsoft.com/azure-cognitive-
services/textanalytics/pii
To use the latest version of the container, you can use the latest  tag, which is for English. You
can also find a full list of containers for supported languages using the tags on the MCR
.
The latest PII detection container is available in several languages. To download the container
for the English container, use the command below.
ﾉ
Expand table
Get the container image with docker pull
docker pull mcr.microsoft.com/azure-cognitive-services/textanalytics/pii:latest
 Tip
You can use the docker images
 command to list your downloaded container images.
For example, the following command lists the ID, repository, and tag of each downloaded
container image, formatted as a table:
\nOnce the container is on the host computer, use the docker run
 command to run the
containers. The container continues to run until you stop it. Replace the placeholders below
with your own values:
To run the PII detection container, execute the following docker run  command. Replace the
placeholders below with your own values:
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
latest
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
Runs a PII detection container from the container image
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
mcr.microsoft.com/azure-cognitive-services/textanalytics/pii:{IMAGE_TAG} \
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
For information on how to call PII see our guide.
To use this container disconnected from the internet, you must first request access by filling
out an application, and purchasing a commitment plan. See Use Docker containers in
disconnected environments for more information.
If you have been approved to run the container disconnected from the internet, use the
following example shows the formatting of the docker run  command you'll use, with
Run the container disconnected from the internet
\n![Image](images/page938_image1.png)
\nplaceholder values. Replace these placeholder values with your own values.
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