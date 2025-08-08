In custom named entity recognition, the data being saved to the snapshot is the labels file.
Learn how to load or export model data for:
Conversational language understanding
Custom text classification
Custom NER
Orchestration workflow
Custom NER
Next steps
\nTutorial: Triage incoming emails with Power
Automate
06/30/2025
In this tutorial you will categorize and triage incoming email using custom text classification.
Using this Power Automate flow, when a new email is received, its contents will have a
classification applied, and depending on the result, a message will be sent to a designated
channel on Microsoft Teams
.
Azure subscription - Create one for free
A Language resource
A trained custom text classification model.
You will need the key and endpoint from your Language resource to authenticate your
Power Automate flow.
A successfully created and deployed single text classification custom model
1. Sign in to Power Automate
2. From the left side menu, select My flows and create a Automated cloud flow
3. Name your flow EmailTriage . Below Choose your flow's triggers, search for email and
select When a new email arrives. Then select create
Prerequisites
Create a Power Automate flow

\n![Image](images/page172_image1.png)
\n4. Add the right connection to your email account. This connection will be used to access
the email content.
5. To add a Language service connector, search for Azure AI Language.
6. Search for CustomSingleLabelClassification.


\n![Image](images/page173_image1.png)

![Image](images/page173_image2.png)
\n7. Start by adding the right connection to your connector. This connection will be used to
access the classification project.
8. In the documents ID field, add 1.
9. In the documents text field, add body from dynamic content.
10. Fill in the project name and deployment name of your deployed custom text classification
model.
11. Add a condition to send a Microsoft Teams message to the right team by:
a. Select results from dynamic content, and add the condition. For this tutorial, we are
looking for Computer_science  related emails. In the Yes condition, choose your desired
option to notify a team channel. In the No condition, you can add additional
conditions to perform alternative actions.


\n![Image](images/page174_image1.png)

![Image](images/page174_image2.png)
\nUse the Language service with Power Automate
Available Language service connectors

Next steps
\n![Image](images/page175_image1.png)
\nTerms and definitions used in custom text
classification
06/30/2025
Use this article to learn about some of the definitions and terms you may encounter when
using custom text classification.
A class is a user-defined category that indicates the overall classification of the text. Developers
label their data with their classes before they pass it to the model for training.
The F1 score is a function of Precision and Recall. It's needed when you seek a balance between
precision and recall.
A model is an object that's trained to do a certain task, in this case text classification tasks.
Models are trained by providing labeled data to learn from so they can later be used for
classification tasks.
Model training is the process of teaching your model how to classify documents based
on your labeled data.
Model evaluation is the process that happens right after training to know how well does
your model perform.
Deployment is the process of assigning your model to a deployment to make it available
for use via the prediction API
.
Measures how precise/accurate your model is. It's the ratio between the correctly identified
positives (true positives) and all identified positives. The precision metric reveals how many of
the predicted classes are correctly labeled.
Class
F1 score
Model
Precision
Project
\nA project is a work area for building your custom ML models based on your data. Your project
can only be accessed by you and others who have access to the Azure resource being used. As
a prerequisite to creating a custom text classification project, you have to connect your
resource to a storage account with your dataset when you create a new project. Your project
automatically includes all the .txt  files available in your container.
Within your project you can do the following:
Label your data: The process of labeling your data so that when you train your model it
learns what you want to extract.
Build and train your model: The core step of your project, where your model starts
learning from your labeled data.
View model evaluation details: Review your model performance to decide if there is
room for improvement, or you are satisfied with the results.
Deployment: After you have reviewed model performance and decide it's fit to be used in
your environment; you need to assign it to a deployment to be able to query it. Assigning
the model to a deployment makes it available for use through the prediction API
.
Test model: After deploying your model, you can use this operation in Language Studio
to try it out your deployment and see how it would perform in production.
Custom text classification supports two types of projects
Single label classification - you can assign a single class for each document in your
dataset. For example, a movie script could only be classified as "Romance" or "Comedy".
Multi label classification - you can assign multiple classes for each document in your
dataset. For example, a movie script could be classified as "Comedy" or "Romance" and
"Comedy".
Measures the model's ability to predict actual positive classes. It's the ratio between the
predicted true positives and what was actually tagged. The recall metric reveals how many of
the predicted classes are correct.
Data and service limits.
Custom text classification overview.
Project types
Recall
Next steps
\nCustom text classification limits
06/30/2025
Use this article to learn about the data and service limits when using custom text classification.
Your Language resource has to be created in one of the supported regions and pricing
tiers listed below.
You can only connect 1 storage account per resource. This process is irreversible. If you
connect a storage account to your resource, you cannot unlink it later. Learn more about
connecting a storage account
You can have up to 500 projects per resource.
Project names have to be unique within the same resource across all custom features.
Custom text classification is available with the following pricing tiers:
Tier
Description
Limit
F0
Free tier
You are only allowed one F0 tier Language resource per subscription.
S
Paid tier
You can have unlimited Language S tier resources per subscription.
See pricing
 for more information.
See Language service regional availability.
Language resource limits
Pricing tiers
ﾉ
Expand table
Regional availability
API limits
ﾉ
Expand table
\nItem
Request
type
Maximum limit
Authoring
API
POST
10 per minute
Authoring
API
GET
100 per minute
Prediction
API
GET/POST
1,000 per minute
Document
size
--
125,000 characters. You can send up to 25 documents as long as they
collectively do not exceed 125,000 characters
Pricing tier
Item
Limit
F
Training time
1 hour per month
S
Training time
Unlimited, Standard
F
Prediction Calls
5,000 text records per month
S
Prediction Calls
Unlimited, Standard
You can only use .txt . files. If your data is in another format, you can use the CLUtils
parse command
 to open your document and extract the text.
All files uploaded in your container must contain data. Empty files are not allowed for
training.
All files should be available at the root of your container.
 Tip
If you need to send larger files than the limit allows, you can break the text into smaller
chunks of text before sending them to the API. You use can the chunk command from
CLUtils
 for this process.
Quota limits
ﾉ
Expand table
Document limits
\nThe following limits are observed for the custom text classification.
Item
Lower
Limit
Upper Limit
Documents count
10
100,000
Document length in characters
1
128,000 characters; approximately 28,000 words or
56 pages.
Count of classes
1
200
Count of trained models per project
0
10
Count of deployments per project
(paid tier)
0
10
Count of deployments per project
(free tier)
0
1
Item
Limits
Project name
You can only use letters (a-z, A-Z) , and numbers (0-9)  ,symbols _ . - ,with no
spaces. Maximum allowed length is 50 characters.
Model name
You can only use letters (a-z, A-Z) , numbers (0-9)  and symbols _ . - . Maximum
allowed length is 50 characters.
Deployment
name
You can only use letters (a-z, A-Z) , numbers (0-9)  and symbols _ . - . Maximum
allowed length is 50 characters.
Class name
You can only use letters (a-z, A-Z) , numbers (0-9)  and all symbols except ":", $ & % *
( ) + ~ # / ? . Maximum allowed length is 50 characters.
Document
name
You can only use letters (a-z, A-Z) , and numbers (0-9)  with no spaces.
Data limits
ﾉ
Expand table
Naming limits
ﾉ
Expand table
Next steps