Use Docker containers in disconnected
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
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
If you run the container with an output mount and logging enabled, the container
generates log files that are helpful to troubleshoot issues that happen while starting or
running the container.
Azure AI containers overview
 Tip
For more troubleshooting information and guidance, see Disconnected containers
Frequently asked questions (FAQ).
Next steps
Yes
No
\nWhat are Azure AI containers?
Article • 03/31/2025
Azure AI services provide several Docker containers
 that let you use the same APIs
that are available in Azure, on-premises. Using these containers gives you the flexibility
to bring Azure AI services closer to your data for compliance, security or other
operational reasons. Container support is currently available for a subset of Azure AI
services.
Containerization is an approach to software distribution in which an application or
service, including its dependencies & configuration, is packaged together as a container
image. With little or no modification, a container image can be deployed on a container
host. Containers are isolated from each other and the underlying operating system, with
a smaller footprint than a virtual machine. Containers can be instantiated from container
images for short-term tasks, and removed when no longer needed.
Immutable infrastructure: Enable DevOps teams to leverage a consistent and
reliable set of known system parameters, while being able to adapt to change.
Containers provide the flexibility to pivot within a predictable ecosystem and avoid
configuration drift.
Control over data: Choose where your data gets processed by Azure AI services.
This can be essential if you can't send data to the cloud but need access to Azure
AI services APIs. Support consistency in hybrid environments – across data,
management, identity, and security.
Control over model updates: Flexibility in versioning and updating of models
deployed in their solutions.
Portable architecture: Enables the creation of a portable application architecture
that can be deployed on Azure, on-premises and the edge. Containers can be
deployed directly to Azure Kubernetes Service, Azure Container Instances, or to a
Kubernetes
 cluster deployed to Azure Stack. For more information, see Deploy
Kubernetes to Azure Stack.
High throughput / low latency: Provide customers the ability to scale for high
throughput and low latency requirements by enabling Azure AI services to run
physically close to their application logic and data. Containers don't cap
transactions per second (TPS) and can be made to scale both up and out to handle
demand if you provide the necessary hardware resources.
https://www.youtube-nocookie.com/embed/hdfbn4Q8jbo
Features and benefits
\nScalability: With the ever growing popularity of containerization and container
orchestration software, such as Kubernetes; scalability is at the forefront of
technological advancements. Building on a scalable cluster foundation, application
development caters to high availability.
Azure AI containers provide the following set of Docker containers, each of which
contains a subset of functionality from services in Azure AI services. You can find
instructions and image locations in the tables below.
Service
Container
Description
Availability
Anomaly
detector
Anomaly
Detector
(image
)
The Anomaly Detector API enables you to
monitor and detect abnormalities in your time
series data with machine learning.
Generally
available
Service
Container
Description
Availability
LUIS
LUIS (image
)
Loads a trained or published Language
Understanding model, also known as a
LUIS app, into a docker container and
provides access to the query
predictions from the container's API
endpoints. You can collect query logs
from the container and upload these
back to the LUIS portal
 to improve
the app's prediction accuracy.
Generally
available.
This container
can also run in
disconnected
environments.
Containers in Azure AI services
７ Note
See Install and run Document Intelligence containers for Azure AI Document
Intelligence container instructions and image locations.
Decision containers
ﾉ
Expand table
Language containers
ﾉ
Expand table
\nService
Container
Description
Availability
Language
service
Key Phrase
Extraction
(image
)
Extracts key phrases to identify the
main points. For example, for the input
text "The food was delicious and there
were wonderful staff," the API returns
the main talking points: "food" and
"wonderful staff".
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Text Language
Detection
(image
)
For up to 120 languages, detects which
language the input text is written in and
report a single language code for every
document submitted on the request.
The language code is paired with a
score indicating the strength of the
score.
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Sentiment
Analysis (image
)
Analyzes raw text for clues about
positive or negative sentiment. This
version of sentiment analysis returns
sentiment labels (for example positive
or negative) for each document and
sentence within it.
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Text Analytics for
health (image
)
Extract and label medical information
from unstructured clinical text.
Generally
available
Language
service
Named Entity
Recognition
(image
)
Extract named entities from text.
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Personally
Identifiable
Information (PII)
detection
(image
)
Detect and redact personally
identifiable information entities from
text.
Generally
available.
This container
can also run in
disconnected
environments.
Language
service
Custom Named
Entity Recognition
(image
)
Extract named entities from text, using
a custom model you create using your
data.
Generally
available
Language
service
Summarization
(image
)
Summarize text from various sources.
Public preview.
This container
can also run in
disconnected
environments.