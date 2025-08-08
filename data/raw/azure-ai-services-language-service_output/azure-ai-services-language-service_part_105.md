If the answer of a new QnA pair matches the answer of an old QnA pair, the two pairs are
merged. The new pair's question is added as an alternate question to the old QnA pair. For
example, consider Q3A3 exists in the old source. When you refresh the source, a new QnA pair
Q3'A3 is introduced. In that case, the two QnA pairs are merged: Q3' is added to Q3 as an
alternate question.
If the old QnA pair has a metadata value, that data is retained and persisted in the newly
merged pair.
If the old QnA pair has follow-up prompts associated with it, then the following scenarios may
arise:
If the prompt attached to the old pair is from the source being refreshed, then it's
deleted, and the prompt of the new pair (if any exists) is appended to the newly merged
QnA pair.
If the prompt attached to the old pair is from a different source, then it's maintained as-is
and the prompt from the new question (if any exists) is appended to the newly merged
QnA pair.
See the following example of a merge operation with differing questions and prompts:
Source
iteration
Question
Answer
Prompts
old
"What is the new HR
policy?"
"You might have to choose among the
following options:"
P1, P2
new
"What is the new payroll
policy?"
"You might have to choose among the
following options:"
P3, P4
The prompts P1 and P2 come from the original source and are different from prompts P3 and
P4 of the new QnA pair. They both have the same answer, You might have to choose among the
following options: , but it leads to different prompts. In this case, the resulting QnA pair would
look like this:
Merge pairs
Merge example
ﾉ
Expand table
ﾉ
Expand table
\nQuestion
Answer
Prompts
"What is the new HR policy?"
(alternate question: "What is the new
payroll policy?")
"You might have to choose among the
following options:"
P3, P4
When the original source has two or more QnA pairs with the same answer (as in, Q1A1 and
Q2A1), the merge behavior may be more complex.
If these two QnA pairs have individual prompts attached to them (for example, Q1A1+P1 and
Q2A1+P2), and the refreshed source content has a new QnA pair generated with the same
answer A1 and a new prompt P3 (Q1'A1+P3), then the new question will be added as an
alternate question to the original pairs (as described above). But all of the original attached
prompts will be overwritten by the new prompt. So the final pair set will look like this:
Question
Answer
Prompts
Q1
(alternate question: Q1')
A1
P3
Q2
(alternate question: Q1')
A1
P3
Custom question answering quickstart
Update Sources API reference
Duplicate answers scenario
ﾉ
Expand table
Next steps
\nCustom question answering encryption of
data at rest
06/21/2025
Custom question answering automatically encrypts your data when it is persisted to the cloud,
helping to meet your organizational security and compliance goals.
By default, your subscription uses Microsoft-managed encryption keys. There is also the option
to manage your resource with your own keys called customer-managed keys (CMK). CMK
offers greater flexibility to create, rotate, disable, and revoke access controls. You can also audit
the encryption keys used to protect your data. If CMK is configured for your subscription,
double encryption is provided, which offers a second layer of protection, while allowing you to
control the encryption key through your Azure Key Vault.
Custom question answering uses CMK support from Azure search, and associates the provided
CMK to encrypt the data stored in Azure search index. Please follow the steps listed in this
article to configure Key Vault access for the Azure search service.
Follow these steps to enable CMKs:
1. Go to the Encryption tab of your language resource with custom question answering
enabled.
About encryption key management
７ Note
Whenever the CMK is being rotated, make sure there is a period of overlap between the
old and new versions of the key where both are enabled and not expired.
） Important
Your Azure Search service resource must have been created after January 2019 and cannot
be in the free (shared) tier. There is no support to configure customer-managed keys in
the Azure portal.
Enable customer-managed keys
\n2. Select the Customer Managed Keys option. Provide the details of your customer-
managed keys and select Save.
3. On a successful save, the CMK will be used to encrypt the data stored in the Azure Search
Index.
Customer-managed keys are available in all Azure Search regions.
Language Studio runs in the user's browser. Every action triggers a direct call to the respective
Azure AI services API. Hence, custom question answering is compliant for data in transit.
Encryption in Azure Search using CMKs in Azure Key Vault
） Important
It is recommended to set your CMK in a fresh Azure AI Search service before any projects
are created. If you set CMK in a language resource with existing projects, you might lose
access to them. Read more about working with encrypted content in Azure AI Search.
Regional availability
Encryption of data in transit
Next steps
\n![Image](images/page1044_image1.png)
\nData encryption at rest
Learn more about Azure Key Vault
\nMove projects and question answer pairs
06/21/2025
This article deals with the process to export and move projects and sources from one
Language resource to another.
You might want to create copies of your projects or sources for several reasons:
To implement a backup and restore process
Integrate with your CI/CD pipeline
When you wish to move your data to different regions
If you don't have an Azure subscription, create a free account
 before you begin.
A language resource
 with the custom question answering feature enabled in the Azure
portal. Remember your Microsoft Entra ID, Subscription, and the Language resource name
you selected when you created the resource.
Exporting a project allows you to back up all the question answer sources that are contained
within a single project.
1. Sign in to the Language Studio
.
2. Select the Language resource you want to move a project from.
3. Go to Custom Question Answering service. On the Projects page, you have the options to
export in two formats, Excel or TSV. This will determine the contents of the file. The file
itself will be exported as a .zip containing the contents of your project.
4. You can export only one project at a time.
1. Select the Language resource, which will be the destination for your previously exported
project.
７ Note
Prerequisites
Export a project
Import a project
\n2. Go to Custom Question Answering service. On the Projects page, select Import and
choose the format used when you selected export. Then browse to the local .zip file
containing your exported project. Enter a name for your newly imported project and
select Done.
1. Select the language resource you want to move an individual question answer source
from.
2. Select the project that contains the question and answer source you wish to export.
3. On the Edit project page, select the ellipsis ( ... ) icon to the right of Enable rich text in
the toolbar. You have the option to export in either Excel or TSV.
1. Select the language resource, which will be the destination for your previously exported
question and answer source.
2. Select the project where you want to import a question and answer source.
3. On the Edit project page, select the ellipsis ( ... ) icon to the right of Enable rich text in
the toolbar. You have the option to import either an Excel or TSV file.
4. Browse to the local location of the file with the Choose File option and select Done.
Test the question answer source by selecting the Test option from the toolbar in the Edit
project page which will launch the test panel. Learn how to test your project.
Deploy the project and create a chat bot. Learn how to deploy your project.
There is no way to move chat logs with projects. If diagnostic logs are enabled, chat logs are
stored in the associated Azure Monitor resource.
Export sources
Import question and answers
Test
Deploy
Chat logs
Next steps
\nUse chitchat with a project
06/30/2025
Adding chitchat to your bot makes it more conversational and engaging. The chitchat feature
in custom question answering allows you to easily add a prepopulated set of the top chitchat,
into your project. This can be a starting point for your bot's personality, and it will save you the
time and cost of writing them from scratch.
This dataset has about 100 scenarios of chitchat in the voice of multiple personas, like
Professional, Friendly and Witty. Choose the persona that most closely resembles your bot's
voice. Given a user query, custom question answering tries to match it with the closest known
chitchat question and answer.
Some examples of the different personalities are below. You can see all the personality
datasets
 along with details of the personalities.
For the user query of When is your birthday? , each personality has a styled response:
Personality
Example
Professional
Age doesn't really apply to me.
Friendly
I don't really have an age.
Witty
I'm age-free.
Caring
I don't have an age.
Enthusiastic
I'm a bot, so I don't have an age.
Chitchat data sets are supported in the following languages:
Language
Chinese
English
ﾉ
Expand table
Language support
ﾉ
Expand table
\nLanguage
French
Germany
Italian
Japanese
Korean
Portuguese
Spanish
After you create your project you can add sources from URLs, files, as well as chitchat from the
Manage sources pane.
Choose the personality that you want as your chitchat base.
Add chitchat source
\n![Image](images/page1049_image1.png)
\nWhen you edit your project, you will see a new source for chitchat, based on the personality
you selected. You can now add altered questions or edit the responses, just like with any other
source.
To turn the views for context and metadata on and off, select Show columns in the toolbar.
You can add a new chitchat question pair that is not in the predefined data set. Ensure that you
are not duplicating a question pair that is already covered in the chitchat set. When you add
any new chitchat question pair, it gets added to your Editorial source. To ensure the ranker
understands that this is chitchat, add the metadata key/value pair "Editorial: chitchat," as seen
in the following image:
Edit your chitchat questions and answers
Add more chitchat questions and answers
\n![Image](images/page1050_image1.png)

![Image](images/page1050_image2.png)