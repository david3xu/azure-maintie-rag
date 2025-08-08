v1.0 client libraries for conversational language understanding and orchestration
workflow are Generally Available for the following languages:
C#
Python
v1.1.0b1 client library for conversation summarization is available as a preview for:
Python
There's a new endpoint URL and request format for making REST API calls to prebuilt
Language service features. See the following quickstart guides and reference
documentation for information on structuring your API calls. All text analytics 3.2-
preview.2  API users can begin migrating their workloads to this new endpoint.
Entity linking
Language detection
Key phrase extraction
Named entity recognition
PII detection
Sentiment analysis and opinion mining
Text analytics for health
PII detection for conversations.
Rebranded Text Summarization to Document summarization.
Conversation summarization is now available in public preview.
The following features are now Generally Available (GA):
Custom text classification
Custom Named Entity Recognition (NER)
Conversational language understanding
Orchestration workflow
The following updates for custom text classification, custom Named Entity Recognition
(NER), conversational language understanding, and orchestration workflow:
Data splitting controls.
Ability to cancel training jobs.
Custom deployments can be named. You can have up to 10 deployments.
Ability to swap deployments.
Auto labeling (preview) for custom named entity recognition
June 2022
May 2022
\nEnterprise readiness support
Training modes for conversational language understanding
Updated service limits
Ability to use free (F0) tier for Language resources
Expanded regional availability
Updated model life cycle to add training configuration versions
Fast Healthcare Interoperability Resources (FHIR) support is available in the Language
REST API preview for Text Analytics for health.
Expanded language support for:
Custom text classification
Custom Named Entity Recognition (NER)
Conversational language understanding
Model improvements for latest model-version for text summarization
Model 2021-10-01  is Generally Available (GA) for Sentiment Analysis and Opinion Mining,
featuring enhanced modeling for emojis and better accuracy across all supported
languages.
Question Answering: Active learning v2 incorporates a better clustering logic providing
improved accuracy of suggestions. It considers user actions when suggestions are
accepted or rejected to avoid duplicate suggestions, and improve query suggestions.
The version 3.1-preview.x REST endpoints and 5.1.0-beta.x client library are retired.
Upgrade to the General Available version of the API(v3.1). If you're using the client
libraries, use package version 5.1.0 or higher. See the migration guide for details.
April 2022
March 2022
February 2022
December 2021
November 2021
\nBased on ongoing customer feedback, we increased the character limit per document for
Text Analytics for health from 5,120 to 30,720.
Azure AI Language release, with support for:
Question Answering (now Generally Available)
Sentiment Analysis and opinion mining
Key Phrase Extraction
Named Entity Recognition (NER), Personally Identifying Information (PII)
Language Detection
Text Analytics for health
Text summarization preview
Custom Named Entity Recognition (Custom NER) preview
Custom Text Classification preview
Conversational Language Understanding preview
Preview model version 2021-10-01-preview  for Sentiment Analysis and Opinion mining,
which provides:
Improved prediction quality.
Added language support for the opinion mining feature.
For more information, see the project z-code site
.
To use this model version, you must specify it in your API calls, using the model version
parameter.
SDK support for sending requests to custom models:
Custom Named Entity Recognition
Custom text classification
Custom language understanding
See the previous updates article for service updates not listed here.
Next steps
\nWhat is custom text classification?
Article • 03/24/2025
Custom text classification is one of the custom features offered by Azure AI Language. It
is a cloud-based API service that applies machine-learning intelligence to enable you to
build custom models for text classification tasks.
Custom text classification enables users to build custom AI models to classify text into
custom classes pre-defined by the user. By creating a custom text classification project,
developers can iteratively label data, train, evaluate, and improve model performance
before making it available for consumption. The quality of the labeled data greatly
impacts model performance. To simplify building and customizing your model, the
service offers a custom web portal that can be accessed through the Language studio
.
You can easily get started with the service by following the steps in this quickstart.
Custom text classification supports two types of projects:
Single label classification - you can assign a single class for each document in
your dataset. For example, a movie script could only be classified as "Romance" or
"Comedy".
Multi label classification - you can assign multiple classes for each document in
your dataset. For example, a movie script could be classified as "Comedy" or
"Romance" and "Comedy".
This documentation contains the following article types:
Quickstarts are getting-started instructions to guide you through making requests
to the service.
Concepts provide explanations of the service functionality and features.
How-to guides contain instructions for using the service in more specific or
customized ways.
Custom text classification can be used in multiple scenarios across a variety of industries:
Support centers of all types receive a high volume of emails or tickets containing
unstructured, freeform text and attachments. Timely review, acknowledgment, and
routing to subject matter experts within internal teams is critical. Email triage at this
Example usage scenarios
Automatic emails or ticket triage
\nscale requires people to review and route to the right departments, which takes time
and resources. Custom text classification can be used to analyze incoming text, and
triage and categorize the content to be automatically routed to the relevant
departments for further action.
Search is foundational to any app that surfaces text content to users. Common scenarios
include catalog or document searches, retail product searches, or knowledge mining for
data science. Many enterprises across various industries are seeking to build a rich
search experience over private, heterogeneous content, which includes both structured
and unstructured documents. As a part of their pipeline, developers can use custom text
classification to categorize their text into classes that are relevant to their industry. The
predicted classes can be used to enrich the indexing of the file for a more customized
search experience.
Creating a custom text classification project typically involves several different steps.
Follow these steps to get the most out of your model:
1. Define your schema: Know your data and identify the classes you want
differentiate between, to avoid ambiguity.
2. Label your data: The quality of data labeling is a key factor in determining model
performance. Documents that belong to the same class should always have the
same class, if you have a document that can fall into two classes use Multi label
classification projects. Avoid class ambiguity, make sure that your classes are
clearly separable from each other, especially with single label classification
projects.
Knowledge mining to enhance/enrich semantic search
Project development lifecycle

\n![Image](images/page45_image1.png)
\n3. Train the model: Your model starts learning from your labeled data.
4. View the model's performance: View the evaluation details for your model to
determine how well it performs when introduced to new data.
5. Deploy the model: Deploying a model makes it available for use via the Analyze
API
.
6. Classify text: Use your custom model for custom text classification tasks.
As you use custom text classification, see the following reference documentation and
samples for Azure AI Language:
Development option
/ language
Reference
documentation
Samples
REST APIs (Authoring)
REST API
documentation
REST APIs (Runtime)
REST API
documentation
C# (Runtime)
C# documentation
C# samples - Single label classification
 C#
samples - Multi label classification
Java (Runtime)
Java documentation
Java Samples - Single label classification
 Java
Samples - Multi label classification
JavaScript (Runtime)
JavaScript
documentation
JavaScript samples - Single label classification
JavaScript samples - Multi label classification
Python (Runtime)
Python
documentation
Python samples - Single label classification
Python samples - Multi label classification
An AI system includes not only the technology, but also the people who will use it, the
people who will be affected by it, and the environment in which it is deployed. Read the
transparency note for custom text classification to learn about responsible AI use and
deployment in your systems. You can also see the following articles for more
information:
Reference documentation and code samples
ﾉ
Expand table
Responsible AI
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
Use the quickstart article to start using custom text classification.
As you go through the project development lifecycle, review the glossary to learn
more about the terms used throughout the documentation for this feature.
Remember to view the service limits for information such as regional availability.
Next steps
Yes
No
\nQuickstart: Custom text classification
Article • 04/29/2025
Use this article to get started with creating a custom text classification project where you can
train custom models for text classification. A model is artificial intelligence software that's
trained to do a certain task. For this system, the models classify text, and are trained by
learning from tagged data.
Custom text classification supports two types of projects:
Single label classification - you can assign a single class for each document in your
dataset. For example, a movie script could only be classified as "Romance" or "Comedy".
Multi label classification - you can assign multiple classes for each document in your
dataset. For example, a movie script could be classified as "Comedy" or "Romance" and
"Comedy".
In this quickstart you can use the sample datasets provided to build a multi label classification
where you can classify movie scripts into one or more categories or you can use single label
classification dataset where you can classify abstracts of scientific papers into one of the
defined domains.
Azure subscription - Create one for free
.
Before you can use custom text classification, you'll need to create an Azure AI Language
resource, which will give you the credentials that you need to create a project and start training
a model. You'll also need an Azure storage account, where you can upload your dataset that
will be used to build your model.
Prerequisites
Create a new Azure AI Language resource and
Azure storage account
） Important
To quickly get started, we recommend creating a new Azure AI Language resource using
the steps provided in this article. Using the steps in this article will let you create the
Language resource and storage account at the same time, which is easier than doing it
later.
\n1. Go to the Azure portal
 to create a new Azure AI Language resource.
2. In the window that appears, select Custom text classification & custom named entity
recognition from the custom features. Select Continue to create your resource at the
bottom of the screen.
3. Create a Language resource with following details.
Name
Required value
Subscription
Your Azure subscription.
Resource
group
A resource group that will contain your resource. You can use an existing one, or
create a new one.
Region
One of the supported regions. For example "West US 2".
Name
A name for your resource.
Pricing tier
One of the supported pricing tiers. You can use the Free (F0) tier to try the service.
If you have a pre-existing resource that you'd like to use, you will need to connect it to
storage account.
Create a new resource from the Azure portal

ﾉ
Expand table
\n![Image](images/page49_image1.png)
\nIf you get a message saying "your login account is not an owner of the selected storage
account's resource group", your account needs to have an owner role assigned on the
resource group before you can create a Language resource. Contact your Azure
subscription owner for assistance.
You can determine your Azure subscription owner by searching your resource group
and following the link to its associated subscription. Then:
a. Select the Access Control (IAM) tab
b. Select Role assignments
c. Filter by Role:Owner.
4. In the Custom text classification & custom named entity recognition section, select an
existing storage account or select New storage account. Note that these values are to
help you get started, and not necessarily the storage account values you’ll want to use in
production environments. To avoid latency during building your project connect to
storage accounts in the same region as your Language resource.
Storage account value
Recommended value
Storage account name
Any name
Storage account type
Standard LRS
5. Make sure the Responsible AI Notice is checked. Select Review + create at the bottom of
the page.
After you have created an Azure storage account and connected it to your Language resource,
you will need to upload the documents from the sample dataset to the root directory of your
container. These documents will later be used to train your model.
1. Download the sample dataset for multi label classification projects
.
2. Open the .zip file, and extract the folder containing the documents.
The provided sample dataset contains about 200 documents, each of which is a summary
for a movie. Each document belongs to one or more of the following classes:
ﾉ
Expand table
Upload sample data to blob container
Multi label classification