Orchestration workflow overview
Next steps
\nTerms and definitions used in orchestration
workflow
06/21/2025
Use this article to learn about some of the definitions and terms you may encounter when
using orchestration workflow.
The F1 score is a function of Precision and Recall. It's needed when you seek a balance between
precision and recall.
An intent represents a task or action the user wants to perform. It is a purpose or goal
expressed in a user's input, such as booking a flight, or paying a bill.
A model is an object that's trained to do a certain task, in this case conversation understanding
tasks. Models are trained by providing labeled data to learn from so they can later be used to
understand utterances.
Model evaluation is the process that happens right after training to know how well does
your model perform.
Deployment is the process of assigning your model to a deployment to make it available
for use via the prediction API
.
Overfitting happens when the model is fixated on the specific examples and is not able to
generalize well.
Measures how precise/accurate your model is. It's the ratio between the correctly identified
positives (true positives) and all identified positives. The precision metric reveals how many of
the predicted classes are correctly labeled.
F1 score
Intent
Model
Overfitting
Precision
\nA project is a work area for building your custom ML models based on your data. Your project
can only be accessed by you and others who have access to the Azure resource being used.
Measures the model's ability to predict actual positive classes. It's the ratio between the
predicted true positives and what was actually tagged. The recall metric reveals how many of
the predicted classes are correct.
Schema is defined as the combination of intents within your project. Schema design is a crucial
part of your project's success. When creating a schema, you want think about which intents
should be included in your project
Training data is the set of information that is needed to train a model.
An utterance is user input that is short text representative of a sentence in a conversation. It is
a natural language phrase such as "book 2 tickets to Seattle next Tuesday". Example utterances
are added to train the model and the model predicts on new utterance at runtime
Data and service limits.
Orchestration workflow overview.
Project
Recall
Schema
Training data
Utterance
Next steps
\nWhat is Azure AI Language Personally
Identifiable Information (PII) detection?
Article • 03/24/2025
Azure AI Language Personally Identifiable Information (PII) detection is a feature offered
by Azure AI Language. The PII detection service is a cloud-based API that utilizes
machine learning and AI algorithms to help you develop intelligent applications with
advanced natural language understanding. Azure AI Language PII detection uses Named
Entity Recognition (NER) to identify and redact sensitive information from input data.
The service classifies sensitive personal data into predefined categories. These
categories include phone numbers, email addresses, and identification documents. This
classification helps to efficiently detect and eliminate such information.
The Text PII and Conversational PII detection preview API (version 2024-11-15-preview )
now supports the option to mask detected sensitive entities with a label beyond just
redaction characters. Customers can specify if personal data content such as names and
phone numbers, that is, "John Doe received a call from 424-878-9192" , are masked
with a redaction character, that is, "******** received a call from ************" , or
masked with an entity label, that is, "[PERSON_1] received a call from
[PHONENUMBER_1]" . More on how to specify the redaction policy style for your outputs
can be found in our how-to guides.
The Conversational PII detection models (both version 2024-11-01-preview  and GA ) are
updated to provide enhanced AI quality and accuracy. The numeric identifier entity type
now also includes Drivers License and Medicare Beneficiary Identifier.
As of June 2024, we now provide General Availability support for the Conversational PII
service (English-language only). Customers can now redact transcripts, chats, and other
text written in a conversational style (that is, text with um s, ah s, multiple speakers, and
the spelling out of words for more clarity) with better confidence in AI quality, Azure
 Tip
Try PII detection in Azure AI Foundry portal
. There you can utilize a currently
existing Language Studio resource or create a new Azure AI Foundry resource.
What's new
\nSLA  support and production environment support, and enterprise-grade security in
mind.
Currently, PII support is available for the following capabilities:
General text PII detection for processing sensitive information (PII) and health
information (PHI) in unstructured text across several predefined categories.
Conversation PII detection, a specialized model designed to handle speech
transcriptions and the informal, conversational tone found in meeting and call
transcripts.
Native Document PII detection for processing structured document files.
Azure AI Language is a cloud-based service that applies Natural Language
Processing (NLP) features to detect categories of personal information (PII) in text-
based data. This documentation contains the following types:
Quickstarts are getting-started instructions to guide you through making
requests to the service.
How-to guides contain instructions for using the service in more specific or
customized ways.
To use this feature, you submit data for analysis and handle the API output in your
application. Analysis is performed as-is, with no added customization to the model
used on your data.
1. Create an Azure AI Language resource, which grants you access to the
features offered by Azure AI Language. It generates a password (called a key)
and an endpoint URL that you use to authenticate API requests.
2. Create a request using either the REST API or the client library for C#, Java,
JavaScript, and Python. You can also send asynchronous calls with a batch
request to combine API requests for multiple features into a single call.
3. Send the request containing your text data. Your key and endpoint are used
for authentication.
Capabilities
Text PII
Typical workflow
\n4. Stream or store the response locally.
Azure AI Language offers named entity recognition to identify and categorize
information within your text. The feature detects PII categories including names,
organizations, addresses, phone numbers, financial account numbers or codes, and
government identification numbers. A subset of this PII is protected health
information (PHI). By specifying domain=phi in your request, only PHI entities are
returned.
To use PII detection, you submit text for analysis and handle the API output in your
application. Analysis is performed as-is, with no customization to the model used on
your data. There are two ways to use PII detection:
Development
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use personally
identifying information detection with text examples with your own data
when you sign up. For more information, see the Azure AI Foundry
website
 or Azure AI Foundry documentation.
REST API or Client
library (Azure SDK)
Integrate PII detection into your applications using the REST API, or the
client library available in various languages. For more information, see the
PII detection quickstart.
As you use this feature in your applications, see the following reference documentation
and samples for Azure AI Language:
Development option / language
Reference documentation
Samples
REST API
REST API documentation
Key features for text PII
Get started with PII detection
ﾉ
Expand table
Reference documentation and code samples
ﾉ
Expand table
\nDevelopment option / language
Reference documentation
Samples
C#
C# documentation
C# samples
Java
Java documentation
Java Samples
JavaScript
JavaScript documentation
JavaScript samples
Python
Python documentation
Python samples
Text PII takes text for analysis. For more information, see Data and service
limits in the how-to guide.
PII works with various written languages. For more information, see language
support. You can specify in which supported languages your source text is
written. If you don't specify a language, the extraction defaults to English. The
API may return offsets in the response to support different multilingual and
emoji encodings.
An AI system includes not only the technology, but also the people who use it, the
people affected by it, and the deployment environment. Read the transparency note for
PII to learn about responsible AI use and deployment in your systems. For more
information, see the following articles:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
Apply sensitivity labels - For example, based on the results from the PII service, a
public sensitivity label might be applied to documents where no PII entities are
detected. For documents where US addresses and phone numbers are recognized,
a confidential label might be applied. A highly confidential label might be used for
documents where bank routing numbers are recognized.
Input requirements and service limits
Text PII
Responsible AI
Example scenarios
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Redact some categories of personal information from documents that get wider
circulation - For example, if customer contact records are accessible to frontline
support representatives, the company can redact the customer's personal
information besides their name from the version of the customer history to
preserve the customer's privacy.
Redact personal information in order to reduce unconscious bias - For example,
during a company's resume review process, they can block name, address, and
phone number to help reduce unconscious gender or other biases.
Replace personal information in source data for machine learning to reduce
unfairness – For example, if you want to remove names that might reveal gender
when training a machine learning model, you could use the service to identify
them and you could replace them with generic placeholders for model training.
Remove personal information from call center transcription – For example, if you
want to remove names or other PII data that happen between the agent and the
customer in a call center scenario. You could use the service to identify and remove
them.
Data cleaning for data science - PII can be used to make the data ready for data
scientists and engineers to be able to use these data to train their machine
learning models. Redacting the data to make sure that customer data isn't
exposed.
There are two ways to get started using the entity linking feature:
Azure AI Foundry is a web-based platform that lets you use several Language
service features without needing to write code.
The quickstart article for instructions on making requests to the service using the
REST API and client library SDK.
Next steps
Yes
No
\nQuickstart: Detect Personally Identifiable
Information (PII)
05/23/2025
Create a Project in Foundry in the Azure AI Foundry Portal
Using the left side pane, select Playgrounds. Then select the Try the Language Playground
button.
７ Note
This quickstart only covers PII detection in documents. To learn more about detecting PII
in conversations, see How to detect and redact PII in conversations.
Prerequisites
Navigate to the Azure AI Foundry Playground
\nThe Language Playground consists of four sections:
Top banner: You can select any of the currently available Language services here.
Right pane: This pane is where you can find the Configuration options for the service,
such as the API and model version, along with features specific to the service.
Center pane: This pane is where you enter your text for processing. After the operation is
run, some results will be shown here.
Right pane: This pane is where Details of the run operation are shown.
Here you can select from two Personally Identifying Information (PII) detection capabilities by
choosing the top banner tiles, Extract PII from conversation or Extract PII from text. Each is
for a different scenario.

Use PII in the Azure AI Foundry Playground
Extract PII from conversation
\n![Image](images/page890_image1.png)