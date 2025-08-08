Security for Azure AI services
Article • 04/29/2025
Security should be considered a top priority in the development of all applications, and with
the growth of artificial-intelligence-enabled applications, security is even more important. This
article outlines various security features available for Azure AI services. Each feature addresses a
specific liability, so multiple features can be used in the same workflow.
For a comprehensive list of Azure service security recommendations, see the Azure AI services
security baseline article.
Feature
Description
Transport Layer
Security (TLS)
All of the Azure AI services endpoints exposed over HTTP enforce the TLS 1.2 protocol.
With an enforced security protocol, consumers attempting to call an Azure AI services
endpoint should follow these guidelines:
The client operating system (OS) needs to support TLS 1.2.
The language (and platform) used to make the HTTP call need to specify TLS 1.2
as part of the request. Depending on the language and platform, specifying TLS
is done either implicitly or explicitly.
For .NET users, consider the Transport Layer Security best practices
Authentication
options
Authentication is the act of verifying a user's identity. Authorization, by contrast, is the
specification of access rights and privileges to resources for a given identity. An
identity is a collection of information about a principal
, and a principal can be either
an individual user or a service.
By default, you authenticate your own calls to Azure AI services using the subscription
keys provided; this is the simplest method but not the most secure. The most secure
authentication method is to use managed roles in Microsoft Entra ID. To learn about
this and other authentication options, see Authenticate requests to Azure AI services.
Key rotation
Each Azure AI services resource has two API keys to enable secret rotation. This is a
security precaution that lets you regularly change the keys that can access your
service, protecting the privacy of your service if a key gets leaked. To learn about this
and other authentication options, see Rotate keys.
Environment
variables
Environment variables are name-value pairs that are stored within a specific
development environment. Environment variables are more secure than using
hardcoded values in your code. For instructions on how to use environment variables
in your code, see the Environment variables guide.
Security features
ﾉ
Expand table
\nFeature
Description
However, if your environment is compromised, the environment variables are
compromised as well, so this isn't the most secure approach. The most secure
authentication method is to use managed roles in Microsoft Entra ID. To learn about
this and other authentication options, see Authenticate requests to Azure AI services.
Customer-
managed keys
(CMK)
This feature is for services that store customer data at rest (longer than 48 hours).
While this data is already double-encrypted on Azure servers, users can get extra
security by adding another layer of encryption, with keys they manage themselves. You
can link your service to Azure Key Vault and manage your data encryption keys there.
Check to see if CMK is supported by the service that you want to use in the Customer-
managed keys documentation.
Virtual networks
Virtual networks allow you to specify which endpoints can make API calls to your
resource. The Azure service rejects API calls from devices outside of your network. You
can set a formula-based definition of the allowed network, or you can define an
exhaustive list of endpoints to allow. This is another layer of security that can be used
in combination with others.
Data loss
prevention
The data loss prevention feature lets an administrator decide what types of URIs their
Azure resource can take as inputs (for those API calls that take URIs as input). This can
be done to prevent the possible exfiltration of sensitive company data: If a company
stores sensitive information (such as a customer's private data) in URL parameters, a
bad actor inside that company could submit the sensitive URLs to an Azure service,
which surfaces that data outside the company. Data loss prevention lets you configure
the service to reject certain URI forms on arrival.
Customer
Lockbox
The Customer Lockbox feature provides an interface for customers to review and
approve or reject data access requests. It's used in cases where a Microsoft engineer
needs to access customer data during a support request. For information on how
Customer Lockbox requests are initiated, tracked, and stored for later reviews and
audits, see the Customer Lockbox guide.
Customer Lockbox is available for the following services:
Azure OpenAI
Translator
Conversational language understanding
Custom text classification
Custom named entity recognition
Orchestration workflow
Bring your own
storage (BYOS)
The Speech service doesn't currently support Customer Lockbox. However, you can
arrange for your service-specific data to be stored in your own storage resource using
bring-your-own-storage (BYOS). BYOS allows you to achieve similar data controls to
Customer Lockbox. Keep in mind that Speech service data stays and is processed in
the Azure region where the Speech resource was created. This applies to any data at
rest and data in transit. For customization features like Custom Speech and Custom
\nFeature
Description
Voice, all customer data is transferred, stored, and processed in the same region where
the Speech service resource and BYOS resource (if used) reside.
To use BYOS with Speech, follow the Speech encryption of data at rest guide.
Microsoft doesn't use customer data to improve its Speech models. Additionally, if
endpoint logging is disabled and no customizations are used, then no customer data
is stored by Speech.
Explore Azure AI services and choose a service to get started.
Next step
\nLanguage service encryption of data at rest
06/30/2025
The Language service automatically encrypts your data when it is persisted to the cloud. The
Language service encryption protects your data and helps you meet your organizational
security and compliance commitments.
Data is encrypted and decrypted using FIPS 140-2
 compliant 256-bit AES
 encryption.
Encryption and decryption are transparent, meaning encryption and access are managed for
you. Your data is secure by default and you don't need to modify your code or applications to
take advantage of encryption.
By default, your subscription uses Microsoft-managed encryption keys. There is also the option
to manage your subscription with your own keys called customer-managed keys (CMK). CMK
offers greater flexibility to create, rotate, disable, and revoke access controls. You can also audit
the encryption keys used to protect your data.
There is also an option to manage your subscription with your own keys. Customer-managed
keys (CMK), also known as Bring your own key (BYOK), offer greater flexibility to create, rotate,
disable, and revoke access controls. You can also audit the encryption keys used to protect
your data.
You must use Azure Key Vault to store your customer-managed keys. You can either create
your own keys and store them in a key vault, or you can use the Azure Key Vault APIs to
generate keys. The Azure AI Foundry resource and the key vault must be in the same region
and in the same Microsoft Entra tenant, but they can be in different subscriptions. For more
information about Azure Key Vault, see What is Azure Key Vault?.
A new Azure AI Foundry resource is always encrypted using Microsoft-managed keys. It's not
possible to enable customer-managed keys at the time that the resource is created. Customer-
managed keys are stored in Azure Key Vault, and the key vault must be provisioned with access
About Azure AI services encryption
About encryption key management
Customer-managed keys with Azure Key Vault
Enable customer-managed keys
\npolicies that grant key permissions to the managed identity that is associated with the Azure AI
Foundry resource. The managed identity is available only after the resource is created using the
Pricing Tier for CMK.
To learn how to use customer-managed keys with Azure Key Vault for Azure AI services
encryption, see:
Configure customer-managed keys with Key Vault for Azure AI services encryption from
the Azure portal
Enabling customer managed keys will also enable a system assigned managed identity, a
feature of Microsoft Entra ID. Once the system assigned managed identity is enabled, this
resource will be registered with Microsoft Entra ID. After being registered, the managed
identity will be given access to the Key Vault selected during customer managed key setup. You
can learn more about Managed Identities.
To enable customer-managed keys, you must use an Azure Key Vault to store your keys. You
must enable both the Soft Delete and Do Not Purge properties on the key vault.
Only RSA keys of size 2048 are supported with Azure AI services encryption. For more
information about keys, see Key Vault keys in About Azure Key Vault keys, secrets and
certificates.
） Important
If you disable system assigned managed identities, access to the key vault will be removed
and any data encrypted with the customer keys will no longer be accessible. Any features
depended on this data will stop working.
） Important
Managed identities do not currently support cross-directory scenarios. When you
configure customer-managed keys in the Azure portal, a managed identity is
automatically assigned under the covers. If you subsequently move the subscription,
resource group, or resource from one Microsoft Entra directory to another, the managed
identity associated with the resource is not transferred to the new tenant, so customer-
managed keys may no longer work. For more information, see Transferring a subscription
between Microsoft Entra directories in FAQs and known issues with managed identities
for Azure resources.
Store customer-managed keys in Azure Key Vault
\nYou can rotate a customer-managed key in Azure Key Vault according to your compliance
policies. When the key is rotated, you must update the Azure AI Foundry resource to use the
new key URI. To learn how to update the resource to use a new version of the key in the Azure
portal, see the section titled Update the key version in Configure customer-managed keys for
Azure AI services by using the Azure portal.
Rotating the key does not trigger re-encryption of data in the resource. There is no further
action required from the user.
To revoke access to customer-managed keys, use PowerShell or Azure CLI. For more
information, see Azure Key Vault PowerShell
 or Azure Key Vault CLI. Revoking access
effectively blocks access to all data in the Azure AI Foundry resource, as the encryption key is
inaccessible by Azure AI services.
Learn more about Azure Key Vault
Rotate customer-managed keys
Revoke access to customer-managed keys
Next steps