Minimize the Apply to each 3 action by clicking on the name. Then add another Apply
to each action to Apply to each 2, like before. its named Apply to each 4. Select the text
box, and add entities as the output for this action.

Get the phone number
\n![Image](images/page631_image1.png)
\nWithin Apply to each 4, add a Condition control. Its be named Condition 2. In the first
text box, search for, and add categories from the Dynamic content window. Be sure the
center box is set to is equal to. Then, in the right text box, enter var_phone .
In the If yes condition, add an Update a row action. Then enter the information like we
did above, for the phone numbers column of the Excel sheet. This appends the phone
number detected by the API to the Excel sheet.


\n![Image](images/page632_image1.png)

![Image](images/page632_image2.png)
\nMinimize Apply to each 4 by clicking on the name. Then create another Apply to each
in the parent action. Select the text box, and add Entities as the output for this action
from the Dynamic content window.

Get the plumbing issues
\n![Image](images/page633_image1.png)
\nNext, the flow checks if the issue description from the Excel table row contains the word
"plumbing". If yes, it adds "plumbing" in the IssueType column. If not, we enter "other."
Inside the Apply to each 4 action, add a Condition Control. Its named Condition 3. In
the first text box, search for, and add Description from the Excel file, using the Dynamic
content window. Be sure the center box says contains. Then, in the right text box, find
and select var_plumbing .

\n![Image](images/page634_image1.png)
\nIn the If yes condition, select Add an action, and select Update a row. Then enter the
information like before. In the IssueType column, select var_plumbing . This applies a
"plumbing" label to the row.
In the If no condition, select Add an action, and select Update a row. Then enter the
information like before. In the IssueType column, select var_other . This applies an
"other" label to the row.
In the top-right corner of the screen, select Save, then Test. Under Test Flow, select
manually. Then select Test, and Run flow.
The Excel file gets updated in your OneDrive account. It looks like the below.


Test the workflow
\n![Image](images/page635_image1.png)

![Image](images/page635_image2.png)
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A

Next steps
Yes
No
\n![Image](images/page636_image1.png)
\nWhat is custom named entity recognition?
Article • 04/29/2025
Custom NER is one of the custom features offered by Azure AI Language. It is a cloud-based
API service that applies machine-learning intelligence to enable you to build custom models
for custom named entity recognition tasks.
Custom NER enables users to build custom AI models to extract domain-specific entities from
unstructured text, such as contracts or financial documents. By creating a Custom NER project,
developers can iteratively label data, train, evaluate, and improve model performance before
making it available for consumption. The quality of the labeled data greatly impacts model
performance. To simplify building and customizing your model, the service offers a custom web
portal that can be accessed through the Language studio
. You can easily get started with the
service by following the steps in this quickstart.
This documentation contains the following article types:
Quickstarts are getting-started instructions to guide you through making requests to the
service.
Concepts provide explanations of the service functionality and features.
How-to guides contain instructions for using the service in more specific or customized
ways.
Custom named entity recognition can be used in multiple scenarios across a variety of
industries:
Many financial and legal organizations extract and normalize data from thousands of complex,
unstructured text sources on a daily basis. Such sources include bank statements, legal
agreements, or bank forms. For example, mortgage application data extraction done manually
by human reviewers may take several days to extract. Automating these steps by building a
custom NER model simplifies the process and saves cost, time, and effort.
Search is foundational to any app that surfaces text content to users. Common scenarios
include catalog or document search, retail product search, or knowledge mining for data
science. Many enterprises across various industries want to build a rich search experience over
Example usage scenarios
Information extraction
Knowledge mining to enhance/enrich semantic search
\nprivate, heterogeneous content, which includes both structured and unstructured documents.
As a part of their pipeline, developers can use custom NER for extracting entities from the text
that are relevant to their industry. These entities can be used to enrich the indexing of the file
for a more customized search experience.
Instead of manually reviewing significantly long text files to audit and apply policies, IT
departments in financial or legal enterprises can use custom NER to build automated solutions.
These solutions can be helpful to enforce compliance policies, and set up necessary business
rules based on knowledge mining pipelines that process structured and unstructured content.
Using custom NER typically involves several different steps.
1. Define your schema: Know your data and identify the entities you want extracted. Avoid
ambiguity.
2. Label your data: Labeling data is a key factor in determining model performance. Label
precisely, consistently and completely.
a. Label precisely: Label each entity to its right type always. Only include what you want
extracted, avoid unnecessary data in your labels.
b. Label consistently: The same entity should have the same label across all the files.
c. Label completely: Label all the instances of the entity in all your files.
3. Train the model: Your model starts learning from your labeled data.
4. View the model's performance: After training is completed, view the model's evaluation
details, its performance and guidance on how to improve it.
5. Deploy the model: Deploying a model makes it available for use via the Analyze API
.
Audit and compliance
Project development lifecycle

\n![Image](images/page638_image1.png)
\n6. Extract entities: Use your custom models for entity extraction tasks.
As you use custom NER, see the following reference documentation and samples for Azure AI
Language:
Development option / language
Reference documentation
Samples
REST APIs (Authoring)
REST API documentation
REST APIs (Runtime)
REST API documentation
C# (Runtime)
C# documentation
C# samples
Java (Runtime)
Java documentation
Java Samples
JavaScript (Runtime)
JavaScript documentation
JavaScript samples
Python (Runtime)
Python documentation
Python samples
An AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Read the transparency
note for custom NER to learn about responsible AI use and deployment in your systems. You
can also see the following articles for more information:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
Use the quickstart article to start using custom named entity recognition.
As you go through the project development lifecycle, review the glossary to learn more
about the terms used throughout the documentation for this feature.
Remember to view the service limits for information such as regional availability.
Reference documentation and code samples
ﾉ
Expand table
Responsible AI
Next steps
\nQuickstart: Custom named entity
recognition
Article • 01/31/2025
Use this article to get started with creating a custom NER project where you can train
custom models for custom entity recognition. A model is artificial intelligence software
that's trained to do a certain task. For this system, the models extract named entities
and are trained by learning from tagged data.
In this article, we use Language Studio to demonstrate key concepts of custom Named
Entity Recognition (NER). As an example we’ll build a custom NER model to extract
relevant entities from loan agreements, such as the:
Date of the agreement
Borrower's name, address, city and state
Lender's name, address, city and state
Loan and interest amounts
Azure subscription - Create one for free
Before you can use custom NER, you'll need to create an Azure AI Language resource,
which will give you the credentials that you need to create a project and start training a
model. You'll also need an Azure storage account, where you can upload your dataset
that will be used to build your model.
Prerequisites
Create a new Azure AI Language resource and
Azure storage account
） Important
To quickly get started, we recommend creating a new Azure AI Language resource
using the steps provided in this article. Using the steps in this article will let you
create the Language resource and storage account at the same time, which is easier
than doing it later.