The following properties in the JSON response have been added:
SentenceText  in sentiment analysis
Warnings  for each document
The names of the following properties in the JSON response have been changed, where
applicable:
score  has been renamed to confidenceScore
confidenceScore  has two decimal points of precision.
type  has been renamed to category
subtype  has been renamed to subcategory
New sentiment analysis feature - opinion mining
New personal ( PII ) domain filter for protected health information ( PHI ).
Additional entity types are now available in the Named Entity Recognition (NER). This update
introduces model version 2020-02-01 , which includes:
Recognition of the following general entity types (English only):
PersonType
Product
Event
Geopolitical Entity (GPE) as a subtype under Location
Skill
Recognition of the following personal information entity types (English only):
Person
Organization
Age as a subtype under Quantity
Date as a subtype under DateTime
Email
Phone Number (US only)
URL
IP Address
February 2020
October 2019
\nIntroduction of PII feature
Model version 2019-10-01 , which includes:
Named entity recognition:
Expanded detection and categorization of entities found in text.
Recognition of the following new entity types:
Phone number
IP address
Sentiment analysis:
Significant improvements in the accuracy and detail of the API's text categorization
and scoring.
Automatic labeling for different sentiments in text.
Sentiment analysis and output on a document and sentence level.
This model version supports: English ( en ), Japanese ( ja ), Chinese Simplified ( zh-Hans ),
Chinese Traditional ( zh-Hant ), French ( fr ), Italian ( it ), Spanish ( es ), Dutch ( nl ),
Portuguese ( pt ), and German ( de ).
See What's new for current service updates.
Next steps
\nConfigure Azure AI services virtual
networks
05/19/2025
Azure AI services provide a layered security model. This model enables you to secure your
Azure AI services accounts to a specific subset of networks​. When network rules are configured,
only applications that request data over the specified set of networks can access the account.
You can limit access to your resources with request filtering, which allows requests that
originate only from specified IP addresses, IP ranges, or from a list of subnets in Azure Virtual
Networks.
An application that accesses an AI Foundry resource when network rules are in effect requires
authorization. Authorization is supported with Microsoft Entra ID credentials or with a valid API
key.
） Important
Turning on firewall rules for your Azure AI services account blocks incoming requests for
data by default. To allow requests through, one of the following conditions needs to be
met:
The request originates from a service that operates within an Azure Virtual Network
on the allowed subnet list of the target Azure AI services account. The endpoint
request that originated from the virtual network needs to be set as the custom
subdomain of your Azure AI services account.
The request originates from an allowed list of IP addresses.
Requests that are blocked include those from other Azure services, from the Azure portal,
and from logging and metrics services.
７ Note
We recommend that you use the Azure Az PowerShell module to interact with Azure. To
get started, see Install Azure PowerShell. To learn how to migrate to the Az PowerShell
module, see Migrate Azure PowerShell from AzureRM to Az.
Scenarios
\nTo secure your Azure AI services resource, you should first configure a rule to deny access to
traffic from all networks, including internet traffic, by default. Then, configure rules that grant
access to traffic from specific virtual networks. This configuration enables you to build a secure
network boundary for your applications. You can also configure rules to grant access to traffic
from select public internet IP address ranges and enable connections from specific internet or
on-premises clients.
Network rules are enforced on all network protocols to Azure AI services, including REST and
WebSocket. To access data by using tools such as the Azure test consoles, explicit network
rules must be configured. You can apply network rules to existing Azure AI services resources,
or when you create new Azure AI services resources. After network rules are applied, they're
enforced for all requests.
Virtual networks are supported in regions where Azure AI services are available
. Azure AI
services support service tags for network rules configuration. The services listed here are
included in the CognitiveServicesManagement  service tag.
Supported regions and service offerings
Anomaly Detector
＂
Azure OpenAI
＂
Content Moderator
＂
Custom Vision
＂
Face
＂
Language Understanding (LUIS)
＂
Personalizer
＂
Speech service
＂
Language
＂
QnA Maker
＂
Translator
＂
７ Note
If you use Azure OpenAI, LUIS, Speech Services, or Language services, the
CognitiveServicesManagement  tag only enables you to use the service by using the SDK or
REST API. To access and use the Azure AI Foundry portal
, LUIS portal, Speech Studio, or
Language Studio from a virtual network, you need to use the following tags:
AzureActiveDirectory
AzureFrontDoor.Frontend
AzureResourceManager
\nBy default, Azure AI services resources accept connections from clients on any network. To limit
access to selected networks, you must first change the default action.
You can manage default network access rules for Azure AI services resources through the Azure
portal, PowerShell, or the Azure CLI.
1. Go to the Azure AI services resource you want to secure.
2. Select Resource Management to expand it, then select Networking.
CognitiveServicesManagement
CognitiveServicesFrontEnd
Storage  (Speech Studio only)
For information on Azure AI Foundry portal
 configurations, see the Azure AI Foundry
documentation.
Change the default network access rule
２ Warning
Making changes to network rules can impact your applications' ability to connect to Azure
AI services. Setting the default network rule to deny blocks all access to the data unless
specific network rules that grant access are also applied.
Before you change the default rule to deny access, be sure to grant access to any allowed
networks by using network rules. If you allow listing for the IP addresses for your on-
premises network, be sure to add all possible outgoing public IP addresses from your on-
premises network.
Manage default network access rules
Azure portal
\n3. To deny access by default, under Firewalls and virtual networks, select Selected
Networks and Private Endpoints.
With this setting alone, unaccompanied by configured virtual networks or address
ranges, all access is effectively denied. When all access is denied, requests that
attempt to consume the Azure AI services resource aren't permitted. The Azure
portal, Azure PowerShell, or the Azure CLI can still be used to configure the Azure AI
services resource.
4. To allow traffic from all networks, select All networks.
5. Select Save to apply your changes.


\n![Image](images/page1766_image1.png)

![Image](images/page1766_image2.png)
\nYou can configure Azure AI services resources to allow access from specific subnets only. The
allowed subnets might belong to a virtual network in the same subscription or in a different
subscription. The other subscription can belong to a different Microsoft Entra tenant. When the
subnet belongs to a different subscription, the Microsoft.CognitiveServices resource provider
needs to be also registered for that subscription.
Enable a service endpoint for Azure AI services within the virtual network. The service endpoint
routes traffic from the virtual network through an optimal path to the Azure AI service. For
more information, see Virtual Network service endpoints.
The identities of the subnet and the virtual network are also transmitted with each request.
Administrators can then configure network rules for the Azure AI services resource to allow
requests from specific subnets in a virtual network. Clients granted access by these network
rules must continue to meet the authorization requirements of the Azure AI services resource
to access the data.
Each Azure AI services resource supports up to 100 virtual network rules, which can be
combined with IP network rules. For more information, see Grant access from an internet IP
range later in this article.
To apply a virtual network rule to an AI Foundry resource, you need the appropriate
permissions for the subnets to add. The required permission is the default Contributor role or
the Cognitive Services Contributor role. Required permissions can also be added to custom role
definitions.
The Azure AI services resource and the virtual networks that are granted access might be in
different subscriptions, including subscriptions that are part of a different Microsoft Entra
tenant.
Grant access from a virtual network
Set required permissions
７ Note
Configuration of rules that grant access to subnets in virtual networks that are a part of a
different Microsoft Entra tenant are currently supported only through PowerShell, the
Azure CLI, and the REST APIs. You can view these rules in the Azure portal, but you can't
configure them.
Configure virtual network rules
\nYou can manage virtual network rules for Azure AI services resources through the Azure portal,
PowerShell, or the Azure CLI.
To grant access to a virtual network with an existing network rule:
1. Go to the Azure AI services resource you want to secure.
2. Select Resource Management to expand it, then select Networking.
3. Confirm that you selected Selected Networks and Private Endpoints.
4. Under Allow access from, select Add existing virtual network.
5. Select the Virtual networks and Subnets options, and then select Enable.
Azure portal

\n![Image](images/page1768_image1.png)
\n７ Note
If a service endpoint for Azure AI services wasn't previously configured for the
selected virtual network and subnets, you can configure it as part of this
operation.
Currently, only virtual networks that belong to the same Microsoft Entra tenant
are available for selection during rule creation. To grant access to a subnet in a
virtual network that belongs to another tenant, use PowerShell, the Azure CLI, or
the REST APIs.
\n![Image](images/page1769_image1.png)
\n6. Select Save to apply your changes.
To create a new virtual network and grant it access:
1. On the same page as the previous procedure, select Add new virtual network.
2. Provide the information necessary to create the new virtual network, and then select
Create.

\n![Image](images/page1770_image1.png)