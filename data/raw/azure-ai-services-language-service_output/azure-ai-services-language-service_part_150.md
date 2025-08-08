SDK and REST developer guide for the
Language service
06/04/2025
Use this article to find information on integrating the Language service SDK and REST API into
your applications.
The Language service provides support through a REST API, and client libraries in several
languages.
The Language service provides three namespaces for using the available features.
Depending on which features and programming language you're using, you'll need to
download one or more of the following packages, and have the following
framework/language version support:
Framework/Language
Minimum supported version
.NET
.NET Framework 4.6.1  or newer, or .NET (formerly .NET Core) 2.0  or
newer.
Java
v8  or later
JavaScript
v14 LTS  or later
Python
v3.7  or later
Development options
Client library (Azure SDK)
Client libraries (Azure SDK)
ﾉ
Expand table
Azure.AI.TextAnalytics
７ Note
\nThe Azure.AI.TextAnalytics  namespace enables you to use the following Language
features. Use the following links for articles to help you send API requests using the SDK.
Custom named entity recognition
Custom text classification
Document summarization
Entity linking
Key phrase extraction
Named entity recognition (NER)
Personally Identifying Information (PII) detection
Sentiment analysis and opinion mining
Text analytics for health
As you use these features in your application, use the following documentation and code
samples for additional information.
Language → Latest GA version
Reference documentation
Samples
C#/.NET → v5.2.0
C# documentation
C# samples
Java → v5.2.0
Java documentation
Java Samples
JavaScript → v1.0.0
JavaScript documentation
JavaScript samples
Python → v5.2.0
Python documentation
Python samples
If you're using custom named entity recognition or custom text classification, you will
need to create a project and train a model before using the SDK. The SDK only
provides the ability to analyze text using models you create. See the following
quickstarts for information on creating a model.
Custom named entity recognition
Custom text classification
ﾉ
Expand table
Azure.AI.Language.Conversations
７ Note
If you're using conversational language understanding or orchestration workflow,
you'll need to create a project and train a model before using the SDK. The SDK only
\nThe Azure.AI.Language.Conversations  namespace enables you to use the following
Language features. Use the following links for articles to help you send API requests using
the SDK.
Conversational language understanding
Orchestration workflow
Conversation summarization (Python only)
Personally Identifying Information (PII) detection for conversations
As you use these features in your application, use the following documentation and code
samples for additional information.
Language → Latest GA version
Reference documentation
Samples
C#/.NET → v1.0.0
C# documentation
C# samples
Python → v1.0.0
Python documentation
Python samples
The Azure.AI.Language.QuestionAnswering  namespace enables you to use the following
Language features:
Question answering
Authoring - Automate common tasks like adding new question answer pairs and
working with projects/knowledge bases.
Prediction - Answer questions based on passages of text.
As you use these features in your application, use the following documentation and code
samples for additional information.
provides the ability to analyze text using models you create. See the following
quickstarts for more information.
Conversational language understanding
Orchestration workflow
ﾉ
Expand table
Azure.AI.Language.QuestionAnswering
ﾉ
Expand table
\nLanguage → Latest GA version
Reference documentation
Samples
C#/.NET → v1.0.0
C# documentation
C# samples
Python → v1.0.0
Python documentation
Python samples
Azure AI Language overview
See also
\nLanguage role-based access control
06/30/2025
Azure AI Language supports Azure role-based access control (Azure RBAC), an authorization
system for managing individual access to Azure resources. Using Azure RBAC, you assign
different team members different levels of permissions for your projects authoring resources.
See the Azure RBAC documentation for more information.
To use Azure RBAC, you must enable Microsoft Entra authentication. You can create a new
resource with a custom subdomain or create a custom subdomain for your existing resource.
Azure RBAC can be assigned to a Language resource. To grant access to an Azure resource, you
add a role assignment.
1. In the Azure portal
, select All services.
2. Select Azure AI services, and navigate to your specific Language resource.
3. Select Access control (IAM) on the left pane.
4. Select Add, then select Add role assignment.
5. On the Role tab on the next screen, select a role you want to add.
6. On the Members tab, select a user, group, service principal, or managed identity.
7. On the Review + assign tab, select Review + assign to assign the role.
Within a few minutes, the target will be assigned the selected role at the selected scope. For
help with these steps, see Assign Azure roles using the Azure portal.
Enable Microsoft Entra authentication
Add role assignment to Language resource
７ Note
You can also set up Azure RBAC for whole resource groups, subscriptions, or
management groups. Do this by selecting the desired scope level and then
navigating to the desired item. For example, selecting Resource groups and then
navigating to a specific resource group.
\nUse the following table to determine access needs for your Language projects.
These custom roles only apply to Language resources.
A user that should only be validating and reviewing the Language apps, typically a tester to
ensure the application is performing well before deploying the project. They might want to
review the application’s assets to notify the app developers of any changes that need to be
made, but do not have direct access to make them. Readers will have access to view the
evaluation results.
Capabilities
API Access
Read
Test
All GET APIs under:
Language authoring conversational language understanding APIs
Language authoring text analysis APIs
Question answering projects Only TriggerExportProjectJob  POST operation under:
Language authoring conversational language understanding export API
Language authoring text analysis export API Only Export POST operation under:
Question Answering Projects All the Batch Testing Web APIs *Language Runtime CLU APIs
*Language Runtime Text Analysis APIs
Language role types
７ Note
All prebuilt capabilities are accessible to all roles
Owner and Contributor roles take priority over the custom language roles
Microsoft Entra ID is only used in case of custom Language roles
If you are assigned as a Contributor on Azure, your role will be shown as Owner in
Language studio portal.
Cognitive Services Language Reader
Cognitive Services Language Writer
\nA user that is responsible for building and modifying an application, as a collaborator in a
larger team. The collaborator can modify the Language apps in any way, train those changes,
and validate/test those changes in the portal. However, this user shouldn’t have access to
deploying this application to the runtime, as they might accidentally reflect their changes in
production. They also shouldn’t be able to delete the application or alter its prediction
resources and endpoint settings (assigning or unassigning prediction resources, making the
endpoint public). This restricts this role from altering an application currently being used in
production. They might also create new applications under this resource, but with the
restrictions mentioned.
Capabilities
API Access
All functionalities under Cognitive Services Language Reader.
Ability to:
Train
Write
All APIs under Language reader
All POST, PUT, and PATCH APIs under:
Language conversational language understanding APIs
Language text analysis APIs
question answering projects Except for
Delete deployment
Delete trained model
Delete Project
Deploy Model
These users are the gatekeepers for the Language applications in production environments.
They should have full access to any of the underlying functions and thus can view everything in
the application and have direct access to edit any changes for both authoring and runtime
environments
Functionality
Cognitive Services Language Owner
７ Note
If you are assigned as an Owner and Language Owner you will be shown as Cognitive
Services Language Owner in Language studio portal.
\nAPI Access
All functionalities under Cognitive Services Language Writer
Deploy
Delete
All APIs available under:
Language authoring conversational language understanding APIs
Language authoring text analysis APIs
question answering projects
\nMultilingual and emoji support in
Language service features
06/30/2025
Multilingual and emoji support has led to Unicode encodings that use more than one code
point
 to represent a single displayed character, called a grapheme. For example, emojis like
🌷 and 👍 may use several characters to compose the shape with additional characters for
visual attributes, such as skin tone. Similarly, the Hindi word अनुच्छेद is encoded as five letters
and three combining marks.
Because of the different lengths of possible multilingual and emoji encodings, Language
service features may return offsets in the response.
Whenever offsets are returned the API response, remember:
Elements in the response may be specific to the endpoint that was called.
HTTP POST/GET payloads are encoded in UTF-8
, which may or may not be the default
character encoding on your client-side compiler or operating system.
Offsets refer to grapheme counts based on the Unicode 8.0.0
 standard, not character
counts.
Offsets can cause problems when using character-based substring methods, for example the
.NET substring() method. One problem is that an offset may cause a substring method to end
in the middle of a multi-character grapheme encoding instead of the end.
In .NET, consider using the StringInfo class, which enables you to work with a string as a series
of textual elements, rather than individual character objects. You can also look for grapheme
splitter libraries in your preferred software environment.
The Language service features returns these textual elements as well, for convenience.
Endpoints that return an offset will support the stringIndexType  parameter. This parameter
adjusts the offset  and length  attributes in the API output to match the requested string
iteration scheme. Currently, we support three types:
Offsets in the API response
Extracting substrings from text with offsets
\ntextElement_v8  (default): iterates over graphemes as defined by the Unicode 8.0.0
standard
unicodeCodePoint : iterates over Unicode Code Points
, the default scheme for Python 3
utf16CodeUnit : iterates over UTF-16 Code Units
, the default scheme for JavaScript, Java,
and .NET
If the stringIndexType  requested matches the programming environment of choice, substring
extraction can be done using standard substring or slice methods.
Language service overview
See also