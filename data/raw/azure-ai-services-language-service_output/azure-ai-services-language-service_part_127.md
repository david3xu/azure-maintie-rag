Configure Language service docker
containers
Article • 04/29/2025
Language service provides each container with a common configuration framework, so that
you can easily configure and manage storage, logging and telemetry, and security settings for
your containers. This article applies to the following containers:
Sentiment Analysis
Language Detection
Key Phrase Extraction
Text Analytics for Health
Summarization
Named Entity Recognition (NER)
Personally Identifiable (PII) detection
Conversational Language Understanding (CLU)
The container has the following configuration settings:
Required
Setting
Purpose
Yes
ApiKey
Tracks billing information.
No
ApplicationInsights
Enables adding Azure Application Insights telemetry support to your
container.
Yes
Billing
Specifies the endpoint URI of the service resource on Azure.
Yes
Eula
Indicates that you've accepted the license for the container.
No
Fluentd
Writes log and, optionally, metric data to a Fluentd server.
No
HTTP Proxy
Configures an HTTP proxy for making outbound requests.
No
Logging
Provides ASP.NET Core logging support for your container.
No
Mounts
Reads and writes data from the host computer to the container and
from the container back to the host computer.
Configuration settings
ﾉ
Expand table
） Important
\nThe ApiKey  setting specifies the Azure resource key used to track billing information for the
container. You must specify a value for the key and it must be a valid key for the Language
resource specified for the Billing configuration setting.
The ApplicationInsights  setting allows you to add Azure Application Insights telemetry
support to your container. Application Insights provides in-depth monitoring of your container.
You can easily monitor your container for availability, performance, and usage. You can also
quickly identify and diagnose errors in your container.
The following table describes the configuration settings supported under the
ApplicationInsights  section.
Required
Name
Data
type
Description
No
InstrumentationKey
String
The instrumentation key of the Application Insights instance
to which telemetry data for the container is sent. For more
information, see Application Insights for ASP.NET Core.
Example:
InstrumentationKey=123456789
The Billing  setting specifies the endpoint URI of the Language resource on Azure used to
meter billing information for the container. You must specify a value for this configuration
setting, and the value must be a valid endpoint URI for a Language resource on Azure. The
container reports usage about every 10 to 15 minutes.
The ApiKey, Billing, and Eula settings are used together, and you must provide valid
values for all three of them; otherwise your container won't start.
ApiKey configuration setting
ApplicationInsights setting
ﾉ
Expand table
Billing configuration setting
ﾉ
Expand table
\nRequired
Name
Data type
Description
Yes
Billing
String
Billing endpoint URI.
The Eula  setting indicates that you've accepted the license for the container. You must specify
a value for this configuration setting, and the value must be set to accept .
Required
Name
Data type
Description
Yes
Eula
String
License acceptance
Example:
Eula=accept
Azure AI services containers are licensed under your agreement
 governing your use of Azure.
If you do not have an existing agreement governing your use of Azure, you agree that your
agreement governing use of Azure is the Microsoft Online Subscription Agreement
, which
incorporates the Online Services Terms
. For previews, you also agree to the Supplemental
Terms of Use for Microsoft Azure Previews
. By using the container you agree to these terms.
Fluentd is an open-source data collector for unified logging. The Fluentd  settings manage the
container's connection to a Fluentd
 server. The container includes a Fluentd logging
provider, which allows your container to write logs and, optionally, metric data to a Fluentd
server.
The following table describes the configuration settings supported under the Fluentd  section.
Name
Data
type
Description
Host
String
The IP address or DNS host name of the Fluentd
server.
Port
Integer
The port of the Fluentd server.
The default value is 24224.
EULA setting
ﾉ
Expand table
Fluentd settings
ﾉ
Expand table
\nName
Data
type
Description
HeartbeatMs
Integer
The heartbeat interval, in milliseconds. If no event
traffic has been sent before this interval expires, a
heartbeat is sent to the Fluentd server. The default
value is 60000 milliseconds (1 minute).
SendBufferSize
Integer
The network buffer space, in bytes, allocated for send
operations. The default value is 32768 bytes (32
kilobytes).
TlsConnectionEstablishmentTimeoutMs
Integer
The timeout, in milliseconds, to establish a SSL/TLS
connection with the Fluentd server. The default value
is 10000 milliseconds (10 seconds).
If UseTLS  is set to false, this value is ignored.
UseTLS
Boolean
Indicates whether the container should use SSL/TLS
for communicating with the Fluentd server. The
default value is false.
If you need to configure an HTTP proxy for making outbound requests, use these two
arguments:
Name
Data
type
Description
HTTP_PROXY
string
The proxy to use, for example, http://proxy:8888
<proxy-url>
HTTP_PROXY_CREDS
string
Any credentials needed to authenticate against the proxy, for example,
username:password . This value must be in lower-case.
<proxy-user>
string
The user for the proxy.
<proxy-password>
string
The password associated with <proxy-user>  for the proxy.
Bash
Http proxy credentials settings
ﾉ
Expand table
docker run --rm -it -p 5000:5000 \
--memory 2g --cpus 1 \
--mount type=bind,src=/home/azureuser/output,target=/output \
<registry-location>/<image-name> \
\nThe Logging  settings manage ASP.NET Core logging support for your container. You can use
the same configuration settings and values for your container that you use for an ASP.NET
Core application.
The following logging providers are supported by the container:
Provider
Purpose
Console
The ASP.NET Core Console  logging provider. All of the ASP.NET Core configuration settings
and default values for this logging provider are supported.
Debug
The ASP.NET Core Debug  logging provider. All of the ASP.NET Core configuration settings and
default values for this logging provider are supported.
Disk
The JSON logging provider. This logging provider writes log data to the output mount.
This container command stores logging information in the JSON format to the output mount:
Bash
This container command shows debugging information, prefixed with dbug , while the
container is running:
Bash
Eula=accept \
Billing=<endpoint> \
ApiKey=<api-key> \
HTTP_PROXY=<proxy-url> \
HTTP_PROXY_CREDS=<proxy-user>:<proxy-password> \
Logging settings
ﾉ
Expand table
docker run --rm -it -p 5000:5000 \
--memory 2g --cpus 1 \
--mount type=bind,src=/home/azureuser/output,target=/output \
<registry-location>/<image-name> \
Eula=accept \
Billing=<endpoint> \
ApiKey=<api-key> \
Logging:Disk:Format=json \
Mounts:Output=/output
docker run --rm -it -p 5000:5000 \
--memory 2g --cpus 1 \
\nThe Disk  logging provider supports the following configuration settings:
Name
Data
type
Description
Format
String
The output format for log files.
Note: This value must be set to json  to enable the logging provider. If this
value is specified without also specifying an output mount while instantiating a
container, an error occurs.
MaxFileSize
Integer
The maximum size, in megabytes (MB), of a log file. When the size of the current
log file meets or exceeds this value, a new log file is started by the logging
provider. If -1 is specified, the size of the log file is limited only by the maximum
file size, if any, for the output mount. The default value is 1.
For more information about configuring ASP.NET Core logging support, see Settings file
configuration.
Use bind mounts to read and write data to and from the container. You can specify an input
mount or output mount by specifying the --mount  option in the docker run
 command.
The Language service containers don't use input or output mounts to store training or service
data.
The exact syntax of the host mount location varies depending on the host operating system.
The host computer's mount location may not be accessible due to a conflict between the
docker service account permissions and the host mount location permissions.
<registry-location>/<image-name> \
Eula=accept \
Billing=<endpoint> \
ApiKey=<api-key> \
Logging:Console:LogLevel:Default=Debug
Disk logging
ﾉ
Expand table
Mount settings
ﾉ
Expand table
\nOptional
Name
Data
type
Description
Not
allowed
Input
String
Language service containers don't use this.
Optional
Output
String
The target of the output mount. The default value is /output . This is the
location of the logs. This includes container logs.
Example:
--mount type=bind,src=c:\output,target=/output
Use more Azure AI containers
Next steps
\nDeploy and run containers on Azure
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