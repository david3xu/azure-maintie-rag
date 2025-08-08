3. Select Save to apply your changes.
To remove a virtual network or subnet rule:
1. On the same page as the previous procedures, select ...(More options) to open the
context menu for the virtual network or subnet, and select Remove.
\n![Image](images/page1771_image1.png)
\n2. Select Save to apply your changes.
You can configure Azure AI services resources to allow access from specific public internet IP
address ranges. This configuration grants access to specific services and on-premises networks,
which effectively block general internet traffic.
You can specify the allowed internet address ranges by using CIDR format (RFC 4632)
 in the
form 192.168.0.0/16  or as individual IP addresses like 192.168.0.1 .
IP network rules are only allowed for public internet IP addresses. IP address ranges reserved
for private networks aren't allowed in IP rules. Private networks include addresses that start
with 10.* , 172.16.*  - 172.31.* , and 192.168.* . For more information, see Private Address
Space (RFC 1918)
.
Currently, only IPv4 addresses are supported. Each Azure AI services resource supports up to
100 IP network rules, which can be combined with virtual network rules.

） Important
Be sure to set the default rule to deny, or network rules have no effect.
Grant access from an internet IP range
 Tip
Small address ranges that use /31  or /32  prefix sizes aren't supported. Configure these
ranges by using individual IP address rules.
\n![Image](images/page1772_image1.png)
\nTo grant access from your on-premises networks to your Azure AI services resource with an IP
network rule, identify the internet-facing IP addresses used by your network. Contact your
network administrator for help.
If you use Azure ExpressRoute on-premises for Microsoft peering, you need to identify the NAT
IP addresses. For more information, see What is Azure ExpressRoute.
For Microsoft peering, the NAT IP addresses that are used are either customer provided or
supplied by the service provider. To allow access to your service resources, you must allow
these public IP addresses in the resource IP firewall setting.
You can manage IP network rules for Azure AI services resources through the Azure portal,
PowerShell, or the Azure CLI.
1. Go to the Azure AI services resource you want to secure.
2. Select Resource Management to expand it, then select Networking.
3. Confirm that you selected Selected Networks and Private Endpoints.
4. Under Firewalls and virtual networks, locate the Address range option. To grant
access to an internet IP range, enter the IP address or address range (in CIDR
format
). Only valid public IP (nonreserved) addresses are accepted.
Configure access from on-premises networks
Managing IP network rules
Azure portal
\nTo remove an IP network rule, select the trash can  icon next to the address range.
5. Select Save to apply your changes.
You can use private endpoints for your Azure AI services resources to allow clients on a virtual
network to securely access data over Azure Private Link. The private endpoint uses an IP
address from the virtual network address space for your Azure AI services resource. Network
traffic between the clients on the virtual network and the resource traverses the virtual network
and a private link on the Microsoft Azure backbone network, which eliminates exposure from
the public internet.
Private endpoints for Azure AI services resources let you:
Secure your Azure AI services resource by configuring the firewall to block all connections
on the public endpoint for the Azure AI service.
Increase security for the virtual network, by enabling you to block exfiltration of data from
the virtual network.

） Important
Be sure to set the default rule to deny, or network rules have no effect.
Use private endpoints
\n![Image](images/page1774_image1.png)
\nSecurely connect to Azure AI services resources from on-premises networks that connect
to the virtual network by using Azure VPN Gateway or ExpressRoutes with private-
peering.
A private endpoint is a special network interface for an Azure resource in your virtual network.
Creating a private endpoint for your Azure AI services resource provides secure connectivity
between clients in your virtual network and your resource. The private endpoint is assigned an
IP address from the IP address range of your virtual network. The connection between the
private endpoint and the Azure AI service uses a secure private link.
Applications in the virtual network can connect to the service over the private endpoint
seamlessly. Connections use the same connection strings and authorization mechanisms that
they would use otherwise. The exception is Speech Services, which require a separate endpoint.
For more information, see Private endpoints with the Speech Services in this article. Private
endpoints can be used with all protocols supported by the Azure AI services resource,
including REST.
Private endpoints can be created in subnets that use service endpoints. Clients in a subnet can
connect to one Azure AI services resource using private endpoint, while using service
endpoints to access others. For more information, see Virtual Network service endpoints.
When you create a private endpoint for an AI Foundry resource in your virtual network, Azure
sends a consent request for approval to the Azure AI services resource owner. If the user who
requests the creation of the private endpoint is also an owner of the resource, this consent
request is automatically approved.
Azure AI services resource owners can manage consent requests and the private endpoints
through the Private endpoint connection tab for the Azure AI services resource in the Azure
portal
.
When you create a private endpoint, specify the Azure AI services resource that it connects to.
For more information on creating a private endpoint, see:
Create a private endpoint by using the Azure portal
Create a private endpoint by using Azure PowerShell
Create a private endpoint by using the Azure CLI
Understand private endpoints
Specify private endpoints
Connect to private endpoints
\nClients on a virtual network that use the private endpoint use the same connection string for
the Azure AI services resource as clients connecting to the public endpoint. The exception is
the Speech service, which requires a separate endpoint. For more information, see Use private
endpoints with the Speech service in this article. DNS resolution automatically routes the
connections from the virtual network to the Azure AI services resource over a private link.
By default, Azure creates a private DNS zone attached to the virtual network with the necessary
updates for the private endpoints. If you use your own DNS server, you might need to make
more changes to your DNS configuration. For updates that might be required for private
endpoints, see Apply DNS changes for private endpoints in this article.
See Use Speech service through a private endpoint.
When you create a private endpoint, the DNS CNAME  resource record for the Azure AI services
resource is updated to an alias in a subdomain with the prefix privatelink . By default, Azure
also creates a private DNS zone that corresponds to the privatelink  subdomain, with the DNS
A resource records for the private endpoints. For more information, see What is Azure Private
DNS.
When you resolve the endpoint URL from outside the virtual network with the private
endpoint, it resolves to the public endpoint of the Azure AI services resource. When it's
resolved from the virtual network hosting the private endpoint, the endpoint URL resolves to
the private endpoint's IP address.
This approach enables access to the Azure AI services resource using the same connection
string for clients in the virtual network that hosts the private endpoints and clients outside the
virtual network.
If you use a custom DNS server on your network, clients must be able to resolve the fully
qualified domain name (FQDN) for the Azure AI services resource endpoint to the private
７ Note
Azure OpenAI in Azure AI Foundry Models uses a different private DNS zone and public
DNS zone forwarder than other Azure AI services. For the correct zone and forwarder
names, see Azure services DNS zone configuration.
Use private endpoints with the Speech service
Apply DNS changes for private endpoints
\nendpoint IP address. Configure your DNS server to delegate your private link subdomain to the
private DNS zone for the virtual network.
For more information on configuring your own DNS server to support private endpoints, see
the following resources:
Name resolution that uses your own DNS server
DNS configuration
You can grant a subset of trusted Azure services access to Azure OpenAI, while maintaining
network rules for other apps. These trusted services will then use managed identity to
authenticate your Azure OpenAI service. The following table lists the services that can access
Azure OpenAI if the managed identity of those services have the appropriate role assignment.
Service
Resource provider name
Azure AI Services
Microsoft.CognitiveServices
Azure Machine Learning
Microsoft.MachineLearningServices
Azure AI Search
Microsoft.Search
You can grant networking access to trusted Azure services by creating a network rule exception
using the REST API or Azure portal:
Bash
 Tip
When you use a custom or on-premises DNS server, you should configure your DNS
server to resolve the Azure AI services resource name in the privatelink  subdomain to
the private endpoint IP address. Delegate the privatelink  subdomain to the private DNS
zone of the virtual network. Alternatively, configure the DNS zone on your DNS server and
add the DNS A records.
Grant access to trusted Azure services for Azure
OpenAI
ﾉ
Expand table
Using the Azure CLI
\nTo revoke the exception, set networkAcls.bypass  to None .
To verify if the trusted service has been enabled from the Azure portal,
1. Use the JSON View from the Azure OpenAI resource overview page
2. Choose your latest API version under API versions. Only the latest API version is
supported, 2023-10-01-preview  .
accessToken=$(az account get-access-token --resource https://management.azure.com 
--query "accessToken" --output tsv)
rid="/subscriptions/<your subscription id>/resourceGroups/<your resource 
group>/providers/Microsoft.CognitiveServices/accounts/<your Azure AI resource 
name>"
curl -i -X PATCH https://management.azure.com$rid?api-version=2023-10-01-preview \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $accessToken" \
-d \
'
{
    "properties":
    {
        "networkAcls": {
            "bypass": "AzureServices"
        }
    }
}
'

\n![Image](images/page1778_image1.png)
\n1. Navigate to your Azure OpenAI resource, and select Networking from the navigation
menu.
2. Under Exceptions, select Allow Azure services on the trusted services list to access this
cognitive services account.

Using the Azure portal
 Tip
You can view the Exceptions option by selecting either Selected networks and
private endpoints or Disabled under Allow access from.

\n![Image](images/page1779_image1.png)

![Image](images/page1779_image2.png)
\nFor pricing details, see Azure Private Link pricing
.
Explore the various Azure AI services
Learn more about Virtual Network service endpoints
Pricing
Next steps