Bash
To shut down the container, in the command-line environment where the container is running,
select Ctrl+C .
If you run the container with an output mount and logging enabled, the container generates
log files that are helpful to troubleshoot issues that happen while starting or running the
container.
The PII detection containers send billing information to Azure, using a Language resource on
your Azure account.
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
In this article, you learned concepts and workflow for downloading, installing, and running PII
detection containers. In summary:
PII detection provides Linux containers for Docker
Container images are downloaded from the Microsoft Container Registry (MCR).
Container images run in Docker.
You must specify billing information when instantiating a container.
Billing arguments
ﾉ
Expand table
Summary
） Important
Azure AI containers aren't licensed to run without being connected to Azure for metering.
Customers need to enable the containers to communicate billing information with the
\nSee Configure containers for configuration settings.
metering service at all times. Azure AI containers don't send customer data (for example,
text that is being analyzed) to Microsoft.
Next steps
\nConfigure Language service docker
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