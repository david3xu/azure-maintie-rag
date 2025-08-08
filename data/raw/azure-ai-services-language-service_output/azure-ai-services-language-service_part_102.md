Data and privacy for question answering
06/24/2025
This article provides some high level details regarding how data is processed by question
answering. Its important to remember that you are responsible for your use and the
implementation of this technology, including complying with all applicable laws and
regulations that apply to you. For example, it's your responsibility to:
Understand where your data is processed and stored by the question answering service in
order to meet regulatory obligations for your application.
Inform the users of your applications that information like chat logs will be logged and
can be used for further processing.
Ensure that you have all necessary licenses, proprietary rights or other permissions
required to the content in your knowledge base that is used as the basis for developing
the QnAs.
question answering uses several Azure services, each with a different purpose. For a detailed
explanation of how these services are used read the documentation here.
Question answering handles two kinds of customer data:
Data sources: Any sources (documents or URLs) added to question answering via the
portal or APIs are parsed to extract the QnA pairs. These QnAs are stored in a Azure
Cognitive Search service
 in the customer's subscription. After extracting QnA pairs the
management service discards the data sources, so no customer data is stored with the
question answering service.
Chat logs: If diagnostic logs are turned on, all chat logs are stored in the Azure Monitor
service in the customer's subscription.
In both of these cases, Microsoft acts as a data processor. Data is stored and served directly
from the customer's subscription.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What data does question answering process?
\nThere are two main parts in the question answering stack that process data:
Extraction of question and answer pairs: Any data sources added by the user to the
knowledge base are parsed to extract these pairs. The algorithm looks for a repeating
pattern in the source documents, or for a particular layout of the content, to determine
which sections constitute a question and answer. question answering optimizes the
extraction for display in a chat bot, which typically has a small surface area. The extracted
QnAs are stored in Azure Cognitive Search.
Search for the best answer match: When the Azure Cognitive Search index is built, the
ranking looks for the best match for any incoming user question. It does so by applying
natural language processing techniques.
The question answering knowledge base and the user chat logs are stored in Azure Cognitive
Search and Azure Monitor in the user's subscription itself.
Only users who have access to the customer's Azure subscription can view the chat logs
stored in Azure Monitor. The owner of the subscription can control who has access by
using role-based access control.
To control access to a question answering knowledge base, you can assign the
appropriate roles to users by using question answering specific roles.
To learn more about privacy and security commitments, see the Microsoft Trust Center
.
Microsoft AI principles
Microsoft responsible AI resources
Microsoft principles for developing and deploying facial recognition technology
Identify principles and practices for responsible AI
Building responsible bots
How does question answering process data?
How is data retained and what customer controls are
available?
Next steps
\nMigrate from QnA Maker to Custom
question answering
06/04/2025
Purpose of this document: This article aims to provide information that can be used to
successfully migrate applications that use QnA Maker to custom question answering. Using this
article, we hope customers gain clarity on the following:
Comparison of features across QnA Maker and custom question answering
Pricing
Simplified Provisioning and Development Experience
Migration phases
Common migration scenarios
Migration steps
Intended Audience: Existing QnA Maker customers
You'll also need to re-enable analytics for the language resource.
In addition to a new set of features, custom question answering provides many technical
improvements to common features.
） Important
Custom question Answering, a feature of Azure AI Language was introduced in November
2021 with several new capabilities including enhanced relevance using a deep learning
ranker, precise answers, and end-to-end region support. Each Custom question answering
project is equivalent to a knowledge base in QnA Maker. Resource level settings such as
Role-based access control (RBAC) aren't migrated to the new resource. These resource
level settings would have to be reconfigured for the language resource post migration:
Automatic RBAC to Language project (not resource)
Automatic enabling of analytics.
Comparison of features
ﾉ
Expand table
\nFeature
QnA
Maker
Custom
question
answering
Details
State of the art
transformer-based
models
➖
✔️
Turing based models that enable search of QnA
at web scale.
Prebuilt capability
➖
✔️
Using this capability one can use the power of
custom question answering without having to
ingest content and manage resources.
Precise answering
➖
✔️
Custom question answering supports precise
answering with the help of SOTA models.
Smart URL Refresh
➖
✔️
Custom question answering provides a means to
refresh ingested content from public sources
with a single click.
Q&A over knowledge
base (hierarchical
extraction)
✔️
✔️
Active learning
✔️
✔️
Custom question answering has an improved
active learning model.
Alternate Questions
✔️
✔️
The improved models in custom question
answering reduce the need to add alternate
questions.
Synonyms
✔️
✔️
Metadata
✔️
✔️
Question Generation
(private preview)
➖
✔️
This new feature allows generation of questions
over text.
Support for
unstructured
documents
➖
✔️
Users can now ingest unstructured documents
as input sources and query the content for
responses
.NET SDK
✔️
✔️
API
✔️
✔️
Unified Authoring
experience
➖
✔️
A single authoring experience across all Azure AI
Language
Multi region support
➖
✔️
\nWhen you're looking at migrating to custom question answering, consider the following:
Component
QnA
Maker
Custom
question
answering
Details
QnA Maker Service
cost
✔️
➖
The fixed cost per resource per month. Only
applicable for QnAMaker.
Custom question
answering service
cost
➖
✔️
The custom question answering cost according
to the Standard model. Only applicable for
custom question answering.
Azure Search cost
✔️
✔️
Applicable for both QnA Maker and custom
question answering.
App Service cost
✔️
➖
Only applicable for QnA Maker. This is the
biggest cost savings for users moving to custom
question answering.
Users may select a higher tier with higher capacity, which will impact overall price they
pay. It doesn’t impact the price on language component of custom question answering.
"Text Records" in custom question answering features refers to the query submitted by
the user to the runtime, and it's a concept common to all features within Language
service. Sometimes a query may have more text records when the query length is higher.
Example price estimations
Usage
Number of
resources in
QnA Maker
Number of app
services in QnA
Maker (Tier)
Monthly
inference calls
in QnA Maker
Search
Partitions x
search replica
(Tier)
Relative cost in
custom question
answering
High
5
5(P1)
8M
9x3(S2)
More expensive
High
100
100(P1)
6M
9x3(S2)
Less expensive
Medium
10
10(S1)
800K
4x3(S1)
Less expensive
Low
4
4(B1)
100K
3x3(S1)
Less expensive
Pricing
ﾉ
Expand table
ﾉ
Expand table
\nSummary: Customers should save cost across the most common configurations as seen in the
relative cost column.
Here you can find the pricing details for custom question answering
 and QnA Maker
.
The Azure pricing calculator
 can provide even more detail.
With the Language service, QnA Maker customers now benefit from a single service that
provides Text Analytics, LUIS, and custom question answering as features of the language
resource. The Language service provides:
One Language resource to access all above capabilities
A single pane of authoring experience across capabilities
A unified set of APIs across all the capabilities
A cohesive, simpler, and powerful product
If you or your organization have applications in development or production that use QnA
Maker, you should update them to use custom question answering as soon as possible. See the
following links for available APIs, SDKs, Bot SDKs, and code samples.
Following are the broad migration phases to consider:
Simplified Provisioning and Development
Experience
Migration Phases
\n![Image](images/page1016_image1.png)
\nMore links that can help:
Authoring portal
API
SDK
Bot SDK: For bots to use custom question answering, use the Bot.Builder.AI.QnA
 SDK –
We recommend customers to continue to use this for their Bot integrations. Here are
some sample usages of the same in the bot’s code: Sample 1
 Sample 2
This topic compares two hypothetical scenarios when migrating from QnA Maker to custom
question answering. These scenarios can help you to determine the right set of migration steps
to execute for the given scenario.
In the first migration scenario, the customer uses qnamaker.ai as the authoring portal and they
want to migrate their QnA Maker knowledge bases to custom question answering.
Migrate your project from QnA Maker to custom question answering
Once migrated to custom question answering:
The resource level settings need to be reconfigured for the language resource
Customer validations should start on the migrated knowledge bases on:
Common migration scenarios
７ Note
An attempt has been made to ensure these scenarios are representative of real customer
migrations, however, individual customer scenarios differ. Also, this article doesn't include
pricing details. Visit the pricing
 page for more information.
） Important
Each Custom question answering project is equivalent to a knowledge base in QnA Maker.
Resource level settings such as Role-based access control (RBAC) aren't migrated to the
new resource. These resource level settings would have to be reconfigured for the
language resource post migration. You'll also need to re-enable analytics for the language
resource.
Migration scenario 1: No custom authoring portal
\nSize validation
Number of QnA pairs in all KBs to match pre and post migration
Customers need to establish new thresholds for their knowledge bases in custom
question answering as the Confidence score mapping is different when compared to QnA
Maker.
Answers for sample questions in pre and post migration
Response time for Questions answered in v1 vs v2
Retaining of prompts
Customers can use the batch testing tool post migration to test the newly created
project in custom question answering.
Old QnA Maker resources need to be manually deleted.
Here are some detailed steps on migration scenario 1.
In this migration scenario, the customer may have created their own authoring frontend using
the QnA Maker authoring APIs or QnA Maker SDKs.
They should perform these steps required for migration of SDKs:
This SDK Migration Guide
 is intended to help the migration to the new custom question
answering client library, Azure.AI.Language.QuestionAnswering
, from the old one,
Microsoft.Azure.CognitiveServices.Knowledge.QnAMaker
. It will focus on side-by-side
comparisons for similar operations between the two packages.
They should perform the steps required for migration of Knowledge bases to the new Project
within Language resource.
Once migrated to custom question answering:
The resource level settings need to be reconfigured for the language resource
Customer validations should start on the migrated knowledge bases on
Size validation
Number of QnA pairs in all KBs to match pre and post migration
Confidence score mapping
Answers for sample questions in pre and post migration
Response time for Questions answered in v1 vs v2
Retaining of prompts
Batch testing pre and post migration
Old QnA Maker resources need to be manually deleted.
Migration scenario 2
\nAdditionally, for customers who have to migrate and upgrade Bot, upgrade bot code is
published as NuGet package.
Here you can find some code samples: Sample 1
 Sample 2
Here are detailed steps on migration scenario 2
Learn more about the prebuilt API
Learn more about the custom question answering Get Answers REST API
Note that some of these steps are needed depending on the customers existing architecture.
Kindly look at migration phases given above for getting more clarity on which steps are
needed by you for migration.
Migration steps
\n![Image](images/page1019_image1.png)
\nMigrate from QnA Maker knowledge bases
to custom question answering
06/04/2025
Custom question answering, a feature of Azure AI Language was introduced in May 2021 with
several new capabilities including enhanced relevance using a deep learning ranker, precise
answers, and end-to-end region support. Each Custom question answering project is
equivalent to a knowledge base in QnA Maker. You can easily migrate knowledge bases from a
QnA Maker resource to custom question answering projects within a language resource
. You
can also choose to migrate knowledge bases from multiple QnA Maker resources to a specific
language resource.
To successfully migrate knowledge bases, the account performing the migration needs
contributor access to the selected QnA Maker and language resource. When a knowledge
base is migrated, the following details are copied to the new custom question answering
project:
QnA pairs including active learning suggestions.
Synonyms and default answer from the QnA Maker resource.
Knowledge base name is copied to project description field.
Resource level settings such as Role-based access control (RBAC) aren't migrated to the new
resource. These resource level settings would have to be reconfigured for the language
resource post migration. You'll also need to re-enable analytics for the language resource.
This SDK Migration Guide
 is intended to help the migration to the new custom question
answering client library, Azure.AI.Language.QuestionAnswering
, from the old one,
Microsoft.Azure.CognitiveServices.Knowledge.QnAMaker
. It focuses on side-by-side
comparisons for similar operations between the two packages.
You can follow the steps below to migrate knowledge bases:
1. Create a language resource
 with custom question answering enabled in advance. When
you create the language resource in the Azure portal, you'll see the option to enable
custom question answering. When you select that option and proceed, you'll be asked for
Azure Search details to save the knowledge bases.
Steps to migrate SDKs
Steps to migrate knowledge bases