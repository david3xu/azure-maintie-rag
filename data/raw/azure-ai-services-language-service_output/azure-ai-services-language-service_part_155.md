4. Specify the signed key Start and Expiry times.
When you create a shared access signature (SAS), the default duration is 48
hours. After 48 hours, you'll need to create a new token.
Consider setting a longer duration period for the time you're using your
storage account for Language Service operations.
The value of the expiry time is determined by whether you're using an
Account key or User delegation key Signing method:
Account key: No imposed maximum time limit; however, best practices
recommended that you configure an expiration policy to limit the interval
and minimize compromise. Configure an expiration policy for shared
access signatures.
User delegation key: The value for the expiry time is a maximum of seven
days from the creation of the SAS token. The SAS is invalid after the user
delegation key expires, so a SAS with an expiry time of greater than seven
days will still only be valid for seven days. For more information, see Use
Microsoft Entra credentials to secure a SAS.
5. The Allowed IP addresses field is optional and specifies an IP address or a range of
IP addresses from which to accept requests. If the request IP address doesn't
match the IP address or address range specified on the SAS token, authorization
fails. The IP address or a range of IP addresses must be public IPs, not private. For
more information, see, Specify an IP address or IP range.
6. The Allowed protocols field is optional and specifies the protocol permitted for a
request made with the SAS. The default value is HTTPS.
7. Review then select Generate SAS token and URL.
8. The Blob SAS token query string and Blob SAS URL are displayed in the lower area
of window.
9. Copy and paste the Blob SAS token and URL values in a secure location. They'll
only be displayed once and cannot be retrieved once the window is closed.
10. To construct a SAS URL, append the SAS token (URI) to the URL for a storage
service.
The SAS URL includes a special set of query parameters. Those parameters indicate how
the client accesses the resources.
Use your SAS URL to grant access
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
You can include your SAS URL with REST API requests in two ways:
Use the SAS URL as your sourceURL and targetURL values.
Append the SAS query string to your existing sourceURL and targetURL values.
Here's a sample REST API request:
JSON
That's it! You learned how to create SAS tokens to authorize how clients access your
data.
{
  "analysisInput": {
    "documents": [
      {
        "id": "doc_0",
        "language": "en",
        "source": {
          "location": "myaccount.blob.core.windows.net/sample-
input/input.pdf?{SAS-Token}"
        },
        "target": {
          "location": "https://myaccount.blob.core.windows.net/sample-
output?{SAS-Token}"
        }
      }
    ]
  }
}
Next steps
Learn more about native document support
Learn more about granting access with SAS
Yes
No
\nManaged identities for Language
resources
Article • 03/05/2025
Managed identities for Azure resources are service principals that create a Microsoft
Entra identity and specific permissions for Azure managed resources. Managed
identities are a safer way to grant access to storage data and replace the requirement
for you to include shared access signature tokens (SAS) with your source and target
container URLs.
You can use managed identities to grant access to any resource that supports
Microsoft Entra authentication, including your own applications.
To grant access to an Azure resource, assign an Azure role to a managed identity
using Azure role-based access control (Azure RBAC).
There's no added cost to use managed identities in Azure.
） Important
When using managed identities, don't include a SAS token URL with your
HTTP requests. Using managed identities replaces the requirement for you to
include shared access signature tokens (SAS) with your source and target
container URLs.
To use managed identities for Language operations, you must create your
Language resource
 in a specific geographic Azure region such as East US. If
your Language resource region is set to Global, then you can't use managed
identity authentication. You can, however, still use Shared Access Signature
(SAS) tokens.
Prerequisites
\n![Image](images/page1543_image1.png)
\nTo get started, you need the following resources:
An active Azure account
. If you don't have one, you can create a free account
.
An single-service Azure AI Language
 resource created in a regional location.
A brief understanding of Azure role-based access control ( Azure RBAC ) using the
Azure portal.
An Azure Blob Storage account
 in the same region as your Language resource.
You also need to create containers to store and organize your blob data within
your storage account.
If your storage account is behind a firewall, you must enable the following
configuration:
1. Go to the Azure portal
 and sign in to your Azure account.
2. Select your Storage account.
3. In the Security + networking group in the left pane, select Networking.
4. In the Firewalls and virtual networks tab, select Enabled from selected
virtual networks and IP addresses.
5. Deselect all check boxes.
6. Make sure Microsoft network routing is selected.
7. Under the Resource instances section, select
Microsoft.CognitiveServices/accounts as the resource type and select your
Language resource as the instance name.
8. Make certain that the Allow Azure services on the trusted services list to
access this storage account box is checked. For more information about
\n![Image](images/page1544_image1.png)
\nmanaging exceptions, see Configure Azure Storage firewalls and virtual
networks.
9. Select Save.
Although network access is now permitted, your Language resource is still unable
to access the data in your Storage account. You need to create a managed identity
for and assign a specific access role to your Language resource.
There are two types of managed identities: system-assigned and user-assigned.
Currently, Document Translation supports system-assigned managed identity:
A system-assigned managed identity is enabled directly on a service instance. It
isn't enabled by default; you must go to your resource and update the identity
setting.
The system-assigned managed identity is tied to your resource throughout its
lifecycle. If you delete your resource, the managed identity is deleted as well.
In the following steps, we enable a system-assigned managed identity and grant your
Language resource limited access to your Azure Blob Storage account.
You must grant the Language resource access to your storage account before it can
create, read, or delete blobs. Once you enabled the Language resource with a system-
７ Note
It may take up to 5 minutes for the network changes to propagate.
Managed identity assignments
Enable a system-assigned managed identity
\n![Image](images/page1545_image1.png)
\nassigned managed identity, you can use Azure role-based access control ( Azure RBAC ),
to give Language features access to your Azure storage containers.
1. Go to the Azure portal
 and sign in to your Azure account.
2. Select your Language resource.
3. In the Resource Management group in the left pane, select Identity. If your
resource was created in the global region, the Identity tab isn't visible. You can still
use Shared Access Signature (SAS) tokens for authentication.
4. Within the System assigned tab, turn on the Status toggle.
5. Select Save.
） Important
User assigned managed identities don't meet the requirements for the batch
processing storage account scenario. Be sure to enable system assigned
managed identity.
Grant storage account access for your
Language resource
） Important
To assign a system-assigned managed identity role, you need
Microsoft.Authorization/roleAssignments/write permissions, such as Owner or
User Access Administrator at the storage scope for the storage resource.
\n![Image](images/page1546_image1.png)
\n1. Go to the Azure portal
 and sign in to your Azure account.
2. Select your Language resource.
3. In the Resource Management group in the left pane, select Identity.
4. Under Permissions select Azure role assignments:
5. On the Azure role assignments page that opened, choose your subscription from
the drop-down menu then select + Add role assignment.
6. Next, assign a Storage Blob Data Contributor role to your Language service
resource. The Storage Blob Data Contributor role gives Language (represented by
the system-assigned managed identity) read, write, and delete access to the blob
container and data. In the Add role assignment pop-up window, complete the
fields as follows and select Save:
Field
Value
Scope
Storage.
ﾉ
Expand table
\n![Image](images/page1547_image1.png)

![Image](images/page1547_image2.png)
\nField
Value
Subscription
The subscription associated with your storage resource.
Resource
The name of your storage resource.
Role
Storage Blob Data Contributor.
7. After the Added Role assignment confirmation message appears, refresh the page
to see the added role assignment.
8. If you don't see the new role assignment right away, wait and try refreshing the
page again. When you assign or remove role assignments, it can take up to 30
minutes for changes to take effect.
A native document Language service operation request is submitted to your
Language service endpoint via a POST request.
With managed identity and Azure RBAC , you no longer need to include SAS URLs.
HTTP requests
\n![Image](images/page1548_image1.png)

![Image](images/page1548_image2.png)
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
If successful, the POST method returns a 202 Accepted  response code and the
service creates a request.
The processed documents appear in your target container.
Next steps
Native document support
Yes
No
\nCall center overview
05/22/2025
Azure AI Language and Azure AI Speech can help you realize partial or full automation of
telephony-based customer interactions, and provide accessibility across multiple channels.
With the Language and Speech services, you can further analyze call center transcriptions,
extract and redact conversation (PII), summarize the transcription, and detect the sentiment.
Some example scenarios for the implementation of Azure AI services in call and contact centers
are:
Virtual agents: Conversational AI-based telephony-integrated voice bots and voice-
enabled chatbots
Agent-assist: Real-time transcription and analysis of a call to improve the customer
experience by providing insights and suggest actions to agents
Post-call analytics: Post-call analysis to create insights into customer conversations to
improve understanding and support continuous improvement of call handling,
optimization of quality assurance and compliance control as well as other insight driven
optimizations.
A holistic call center implementation typically incorporates technologies from the Language
and Speech services.
Audio data typically used in call centers generated through landlines, mobile phones, and
radios are often narrowband, in the range of 8 KHz, which can create challenges when you're
converting speech to text. The Speech service recognition models are trained to ensure that
you can get high-quality transcriptions, however you choose to capture the audio.
Once you transcribe your audio with the Speech service, you can use the Language service to
perform analytics on your call center data such as: sentiment analysis, summarizing the reason
 Tip
Try the Language Studio
 or Speech Studio
 for a demonstration on how to use the
Language and Speech services to analyze call center conversations.
To deploy a call center transcription solution to Azure with a no-code approach, try the
Ingestion Client.
Azure AI services features for call centers