Bash
Using these parameters we can successfully filter on only Location  entity types, since
the GPE  entity tag included in the includeList  parameter, falls under the Location  type.
We then filter on only Geopolitical entities and exclude any entities tagged with
Continent  or CountryRegion  tags.
In order to provide users with more insight into an entity's types and provide increased
usability, NER supports these attributes in the output:
Name of the
attribute
Type
Definition
type
String
The most specific type of detected entity.
For example, “Seattle” is a City , a GPE  (Geo Political Entity) and a
Location . The most granular classification for “Seattle” is that it is a
City . The type would be City  for the text “Seattle".
tags
List
(tags)
A list of tag objects which expresses the affinity of the detected entity
to a hierarchy or any other grouping.
A tag contains two fields:
1. name : A unique name for the tag.
2. confidenceScore : The associated confidence score for a tag ranging
from 0 to 1.
    "parameters": 
    {
        "includeList" :
        [
            "GPE"
        ],
        "excludeList": :
        [
            "Continent",
            "CountryRegion"
        ]
    }
    
Additional output attributes
ﾉ
Expand table
\nName of the
attribute
Type
Definition
This unique tagName is used to filter in the inclusionList  and
exclusionList  parameters.
metadata
Object
Metadata is an object containing more data about the entity type
detected. It changes based on the field metadataKind .
This sample output includes an example of the additional output attributes.
Bash
Sample output
{ 
    "kind": "EntityRecognitionResults", 
    "results": { 
        "documents": [ 
            { 
                "id": "1", 
                "entities": [ 
                    { 
                        "text": "Microsoft", 
                        "category": "Organization", 
                        "type": "Organization", 
                        "offset": 0, 
                        "length": 9, 
                        "confidenceScore": 0.97, 
                        "tags": [ 
                            { 
                                "name": "Organization", 
                                "confidenceScore": 0.97 
                            } 
                        ] 
                    }, 
                    { 
                        "text": "One", 
                        "category": "Quantity", 
                        "type": "Number", 
                        "subcategory": "Number", 
                        "offset": 21, 
                        "length": 3, 
                        "confidenceScore": 0.9, 
                        "tags": [ 
                            { 
                                "name": "Number", 
                                "confidenceScore": 0.8 
                            }, 
                            { 
                                "name": "Quantity", 
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
By default, this feature uses the latest available AI model on your text. You can also
configure your API requests to use a specific model version.
For information on the size and number of requests you can send per minute and
second, see the service limits article.
Named Entity Recognition overview
                                "confidenceScore": 0.8 
                            }, 
                            { 
                                "name": "Numeric", 
                                "confidenceScore": 0.8 
                            } 
                        ], 
                        "metadata": { 
                            "metadataKind": "NumberMetadata", 
                            "numberKind": "Integer", 
                            "value": 1.0 
                        } 
                    } 
                ], 
                "warnings": [] 
            } 
        ], 
        "errors": [], 
        "modelVersion": "2023-09-01" 
    } 
} 
Specify the NER model
Service and data limits
Next steps
Yes
No
\nUsing named entity recognition skill
parameters
Article • 04/29/2025
Use this article to get an overview of the different API parameters used to adjust the input to a
Named Entity Recognition (NER) API call. The Generally Available NER service now supports the
ability to specify a list of entity tags to be included into the response or excluded from the
response. If a piece of text is classified as more than one entity type, the overlapPolicy
parameter allows customers to specify how the service will handle the overlap. The
inferenceOptions  parameter allows for users to adjust the inference, such as excluding the
detected entity values from being normalized and included in the metadata.
The inclusionList  parameter allows for you to specify which of the NER entity tags, you would
like included in the entity list output in your inference JSON listing out all words and
categorizations recognized by the NER service. By default, all recognized entities are listed.
The exclusionList  parameter allows for you to specify which of the NER entity tags, you would
like excluded in the entity list output in your inference JSON listing out all words and
categorizations recognized by the NER service. By default, all recognized entities are listed.
The overlapPolicy  parameter allows for you to specify how you like the NER service to
respond to recognized words/phrases that fall into more than one category.
By default, the overlapPolicy  parameter is set to matchLongest . This option categorizes the
extracted word/phrase under the entity category that can encompass the longest span of the
extracted word/phrase (longest defined by the most number of characters included).
The alternative option for this parameter is allowOverlap , where all possible entity categories
are listed. Parameters by supported API version
InclusionList parameter
ExclusionList parameter
overlapPolicy parameter
inferenceOptions parameter
\nDefines a selection of options available for adjusting the inference. Currently we have only one
property called excludeNormalizedValues  which excludes the detected entity values to be
normalized and included in the metadata. The numeric and temporal entity types support value
normalization.
This bit of sample code explains how to use skill parameters.
Bash
See Configure containers for configuration settings.
Sample
{ 
    "analysisInput": { 
        "documents": [ 
            { 
                "id": "1", 
                "text": "My name is John Doe", 
                "language": "en" 
            } 
        ] 
    }, 
    "kind": "EntityRecognition", 
    "parameters": { 
        "overlapPolicy": { 
            "policyKind": "AllowOverlap" //AllowOverlap|MatchLongest(default) 
        }, 
        "inferenceOptions": { 
            "excludeNormalizedValues": true //(Default: false) 
        }, 
        "inclusionList": [ 
            "DateAndTime" // A list of entity tags to be used to allow into the 
response. 
        ], 
        "exclusionList": ["Date"] // A list of entity tags to be used to filter 
out from the response. 
    } 
} 
Next steps
\nInstall and run Named Entity Recognition
containers
06/21/2025
Containers enable you to host the Named Entity Recognition API on your own infrastructure. If
you have security or data governance requirements that can't be fulfilled by calling Named
Entity Recognition remotely, then containers might be a good option.
If you don't have an Azure subscription, create a free account
 before you begin.
You must meet the following prerequisites before using Named Entity Recognition containers.
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
\n![Image](images/page557_image1.png)

![Image](images/page557_image2.png)
\nAzure Kubernetes Service.
Azure Container Instances.
A Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
The following table describes the minimum and recommended specifications for the available
container. Each CPU core must be at least 2.6 gigahertz (GHz) or faster.
It is recommended to have a CPU with AVX-512 instruction set, for the best experience
(performance and accuracy).
Minimum host specs
Recommended host specs
Named Entity Recognition
1 core, 2GB memory
4 cores, 8GB memory
CPU core and memory correspond to the --cpus  and --memory  settings, which are used as part
of the docker run  command.
The Named Entity Recognition container image can be found on the mcr.microsoft.com
container registry syndicate. It resides within the azure-cognitive-services/textanalytics/
repository and is named ner . The fully qualified container image name is,
mcr.microsoft.com/azure-cognitive-services/textanalytics/ner
To use the latest version of the container, you can use the latest  tag, which is for English. You
can also find a full list of containers for supported languages using the tags on the MCR
.
The latest Named Entity Recognition container is available in several languages. To download
the container for the English container, use the command below.
ﾉ
Expand table
Get the container image with docker pull
docker pull mcr.microsoft.com/azure-cognitive-services/textanalytics/ner:latest
 Tip
You can use the docker images
 command to list your downloaded container images.
For example, the following command lists the ID, repository, and tag of each downloaded
container image, formatted as a table:
\nOnce the container is on the host computer, use the docker run
 command to run the
containers. The container will continue to run until you stop it. Replace the placeholders below
with your own values:
To run the Named Entity Recognition container, execute the following docker run  command.
Replace the placeholders below with your own values:
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
Runs a Named Entity Recognition container from the container image
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
mcr.microsoft.com/azure-cognitive-services/textanalytics/ner:{IMAGE_TAG} \
Eula=accept \
Billing={ENDPOINT_URI} \
ApiKey={API_KEY}
Run multiple containers on the same host
Query the container's prediction endpoint
Validate that a container is running
ﾉ
Expand table