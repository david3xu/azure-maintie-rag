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
\n![Image](images/page1321_image1.png)

![Image](images/page1321_image2.png)
\nAzure Kubernetes Service.
Azure Container Instances.
A Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
The following table describes the minimum and recommended specifications for the Text
Analytics for health containers. Each CPU core must be at least 2.6 gigahertz (GHz) or faster.
The allowable Transactions Per Second (TPS) are also listed.
Minimum host
specs
Recommended host
specs
Minimum
TPS
Maximum
TPS
1 document/request
4 core, 12GB
memory
6 core, 12GB memory
15
30
10
documents/request
6 core, 16GB
memory
8 core, 20GB memory
15
30
CPU core and memory correspond to the --cpus  and --memory  settings, which are used as part
of the docker run  command.
The Text Analytics for health container image can be found on the mcr.microsoft.com  container
registry syndicate. It resides within the azure-cognitive-services/textanalytics/  repository
and is named healthcare . The fully qualified container image name is
mcr.microsoft.com/azure-cognitive-services/textanalytics/healthcare
To use the latest version of the container, you can use the latest  tag. You can also find a full
list of tags on the MCR
.
Use the docker pull
 command to download this container image from the Microsoft public
container registry. You can find the featured tags on the Microsoft Container Registry
ﾉ
Expand table
Get the container image with docker pull
docker pull mcr.microsoft.com/azure-cognitive-services/textanalytics/healthcare:
<tag-name>
 Tip
\nOnce the container is on the host computer, use the docker run
 command to run the
containers. The container will continue to run until you stop it.
There are multiple ways you can install and run the Text Analytics for health container.
Use the Azure portal to create a Language resource, and use Docker to get your
container.
Use an Azure VM with Docker to run the container.
Use the following PowerShell and Azure CLI scripts to automate resource deployment and
container configuration.
When you use the Text Analytics for health container, the data contained in your API requests
and responses is not visible to Microsoft, and is not used for training the model applied to your
data.
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
The responsible AI (RAI) acknowledgment must also be present with a value of
accept .
The sentiment analysis and language detection containers use v3 of the API, and are
generally available. The key phrase extraction container uses v2 of the API, and is in
preview.
\nTo run the container in your own environment after downloading the container image, execute
the following docker run  command. Replace the placeholders below with your own values:
Placeholder
Value
Format or example
{API_KEY}
The key for your Language resource.
You can find it on your resource's
Key and endpoint page, on the
Azure portal.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
{ENDPOINT_URI}
The endpoint for accessing the API.
You can find it on your resource's
Key and endpoint page, on the
Azure portal.
https://<your-custom-
subdomain>.cognitiveservices.azure.com
Bash
This command:
Runs the Text Analytics for health container from the container image
Allocates 6 CPU core and 12 gigabytes (GB) of memory
Exposes TCP port 5000 and allocates a pseudo-TTY for the container
Accepts the end user license agreement (EULA) and responsible AI (RAI) terms
Automatically removes the container after it exits. The container image is still available on
the host computer.
The container provides REST-based query prediction endpoint APIs. We have also provided a
visualization tool in the container that is accessible by appending /demo  to the endpoint of the
container. For example:
Run the container locally
ﾉ
Expand table
docker run --rm -it -p 5000:5000 --cpus 6 --memory 12g \
mcr.microsoft.com/azure-cognitive-services/textanalytics/healthcare:<tag-name> \
Eula=accept \
rai_terms=accept \
Billing={ENDPOINT_URI} \
ApiKey={API_KEY} 
Demo UI to visualize output
\nUse the example cURL request below to submit a query to the container you have deployed
replacing the serverURL  variable with the appropriate value.
Bash
Azure Web App for Containers
 is an Azure resource dedicated to running containers in the
cloud. It brings out-of-the-box capabilities such as autoscaling, support for docker containers
and docker compose, HTTPS support and much more.
Run this PowerShell script using the Azure CLI to create a Web App for Containers, using your
subscription and the container image over HTTPS. Wait for the script to complete
(approximately 25-30 minutes) before submitting the first request.
Azure CLI
http://<serverURL>:5000/demo
curl -X POST 'http://<serverURL>:5000/text/analytics/v3.1/entities/health' --
header 'Content-Type: application/json' --header 'accept: application/json' --
data-binary @example.json
Install the container using Azure Web App for Containers
７ Note
Using Azure Web App you will automatically get a domain in the form of
<appservice_name>.azurewebsites.net
$subscription_name = ""                    # THe name of the subscription you want 
you resource to be created on.
$resource_group_name = ""                  # The name of the resource group you 
want the AppServicePlan
                                           #    and AppSerivce to be attached to.
$resources_location = ""                   # This is the location you wish the 
AppServicePlan to be deployed to.
                                           #    You can use the "az account list-
locations -o table" command to
                                           #    get the list of available 
locations and location code names.
$appservice_plan_name = ""                 # This is the AppServicePlan name you 
wish to have.
$appservice_name = ""                      # This is the AppService resource name 
you wish to have.
$TEXT_ANALYTICS_RESOURCE_API_KEY = ""      # This should be taken from the 
\nYou can also use an Azure Container Instance (ACI) to make deployment easier. ACI is a
resource that allows you to run Docker containers on-demand in a managed, serverless Azure
environment.
See How to use Azure Container Instances for steps on deploying an ACI resource using the
Azure portal. You can also use the below PowerShell script using Azure CLI, which will create an
ACI on your subscription using the container image. Wait for the script to complete
(approximately 25-30 minutes) before submitting the first request. Due to the limit on the
maximum number of CPUs per ACI resource, do not select this option if you expect to submit
more than 5 large documents (approximately 5000 characters each) per request. See the ACI
regional support article for availability information.
Azure CLI
Language resource.
$TEXT_ANALYTICS_RESOURCE_API_ENDPOINT = "" # This should be taken from the 
Language resource.
$DOCKER_IMAGE_NAME = "mcr.microsoft.com/azure-cognitive-
services/textanalytics/healthcare:latest"
az login
az account set -s $subscription_name
az appservice plan create -n $appservice_plan_name -g $resource_group_name --is-
linux -l $resources_location --sku P3V2
az webapp create -g $resource_group_name -p $appservice_plan_name -n 
$appservice_name -i $DOCKER_IMAGE_NAME 
az webapp config appsettings set -g $resource_group_name -n $appservice_name --
settings Eula=accept rai_terms=accept 
Billing=$TEXT_ANALYTICS_RESOURCE_API_ENDPOINT 
ApiKey=$TEXT_ANALYTICS_RESOURCE_API_KEY
# Once deployment complete, the resource should be available at: 
https://<appservice_name>.azurewebsites.net
Install the container using Azure Container Instance
７ Note
Azure Container Instances don't include HTTPS support for the builtin domains. If you
need HTTPS, you will need to manually configure it, including creating a certificate and
registering a domain. You can find instructions to do this with NGINX below.
$subscription_name = ""                    # The name of the subscription you want 
you resource to be created on.
$resource_group_name = ""                  # The name of the resource group you 
want the AppServicePlan
\nBy default there is no security provided when using ACI with container API. This is because
typically containers will run as part of a pod which is protected from the outside by a network
bridge. You can however modify a container with a front-facing component, keeping the
container endpoint private. The following examples use NGINX
 as an ingress gateway to
support HTTPS/SSL and client-certificate authentication.
                                           # and AppService to be attached to.
$resources_location = ""                   # This is the location you wish the web 
app to be deployed to.
                                           # You can use the "az account list-
locations -o table" command to
                                           # Get the list of available locations 
and location code names.
$azure_container_instance_name = ""        # This is the AzureContainerInstance 
name you wish to have.
$TEXT_ANALYTICS_RESOURCE_API_KEY = ""      # This should be taken from the 
Language resource.
$TEXT_ANALYTICS_RESOURCE_API_ENDPOINT = "" # This should be taken from the 
Language resource.
$DNS_LABEL = ""                            # This is the DNS label name you wish 
your ACI will have
$DOCKER_IMAGE_NAME = "mcr.microsoft.com/azure-cognitive-
services/textanalytics/healthcare:latest"
az login
az account set -s $subscription_name
az container create --resource-group $resource_group_name --name 
$azure_container_instance_name --image $DOCKER_IMAGE_NAME --cpu 4 --memory 12 --
port 5000 --dns-name-label $DNS_LABEL --environment-variables Eula=accept 
rai_terms=accept Billing=$TEXT_ANALYTICS_RESOURCE_API_ENDPOINT 
ApiKey=$TEXT_ANALYTICS_RESOURCE_API_KEY
# Once deployment complete, the resource should be available at: 
http://<unique_dns_label>.<resource_group_region>.azurecontainer.io:5000
Secure ACI connectivity
７ Note
NGINX is an open-source, high-performance HTTP server and proxy. An NGINX container
can be used to terminate a TLS connection for a single container. More complex NGINX
ingress-based TLS termination solutions are also possible.
Set up NGINX as an ingress gateway
\nNGINX uses configuration files
 to enable features at runtime. In order to enable TLS
termination for another service, you must specify an SSL certificate to terminate the TLS
connection and proxy_pass  to specify an address for the service. A sample is provided below.
The NGINX container will load all of the files in the _.conf_  that are mounted under
/etc/nginx/conf.d/  into the HTTP configuration path.
nginx
The below example shows how a docker compose
 file can be created to deploy NGINX and
health containers:
YAML
７ Note
ssl_certificate  expects a path to be specified within the NGINX container's local
filesystem. The address specified for proxy_pass  must be available from within the NGINX
container's network.
server {
  listen              80;
  return 301 https://$host$request_uri;
}
server {
  listen              443 ssl;
  # replace with .crt and .key paths
  ssl_certificate     /cert/Local.crt;
  ssl_certificate_key /cert/Local.key;
  location / {
    proxy_pass http://cognitive-service:5000;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Real-IP  $remote_addr;
  }
}
Example Docker compose file
version: "3.7"
services:
  cognitive-service:
    image: {IMAGE_ID}
    ports:
      - 5000:5000
    environment:
\nTo initiate this Docker compose file, execute the following command from a console at the root
level of the file:
Bash
For more information, see NGINX's documentation on NGINX SSL Termination
.
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
      - eula=accept
      - billing={ENDPOINT_URI}
      - apikey={API_KEY}
    volumes:
        # replace with path to logs folder
      - <path-to-logs-folder>:/output
  nginx:
    image: nginx
    ports:
      - 443:443
    volumes:
        # replace with paths for certs and conf folders
      - <path-to-certs-folder>:/cert
      - <path-to-conf-folder>:/etc/nginx/conf.d/
docker-compose up
Run multiple containers on the same host
Query the container's prediction endpoint
Validate that a container is running
\nvarious request URLs that follow to validate the container is running. The example request URLs
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
You can use the Visual Studio Code REST Client extension
 or the example cURL request
below to submit a query to the container you deployed, replacing the serverURL  variable with
ﾉ
Expand table
Structure the API request for the container
\n![Image](images/page1330_image1.png)