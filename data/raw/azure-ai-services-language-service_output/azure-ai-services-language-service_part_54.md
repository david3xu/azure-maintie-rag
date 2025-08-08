Quickstart: Detecting named entities (NER)
05/23/2025
Create a Project in Foundry in the Azure AI Foundry Portal
Using the left side pane, select Playgrounds. Then select the Try the Language Playground
button.
Prerequisites
Navigate to the Azure AI Foundry Playground

Use NER in the Azure AI Foundry Playground
\n![Image](images/page531_image1.png)
\nThe Language Playground consists of four sections:
Top banner: You can select any of the currently available Language services here.
Right pane: This pane is where you can find the Configuration options for the service,
such as the API and model version, along with features specific to the service.
Center pane: This pane is where you enter your text for processing. After the operation is
run, some results are shown here.
Right pane: This pane is where Details of the run operation are shown.
Here you can select the Named Entity Recognition capability by choosing the top banner tile,
Extract Named Entities.
Extract Named Entities is designed to identify named entities in text.
In Configuration there are the following options:
Option
Description
Select API version
Select which version of the API to use.
Select model version
Select which version of the model to use.
Select text language
Select which language the language is input in.
Select types to include
Select they types of information you want to extract.
Overlap policy
Select the policy for overlapping entities.
Inference options
Additional options to customize the return of the processed data.
After your operation is completed, the type of entity is displayed beneath each entity in the
center pane and the Details section contains the following fields for each entity:
Field
Description
Entity
The detected entity.
Category
The type of entity that was detected.
Offset
The number of characters that the entity was detected from the beginning of the line.
Use Extract Named Entities
ﾉ
Expand table
ﾉ
Expand table
\nField
Description
Length
The character length of the entity.
Confidence
How confident the model is in the correctness of identification of entity's type.
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other resources
associated with it.
Azure portal
Azure CLI
NER overview

Clean up resources
Next steps
\n![Image](images/page533_image1.png)
\nNamed Entity Recognition (NER) language
support
06/21/2025
Use this article to learn which natural languages are supported by the NER feature of Azure AI
Language.
Language
Language Code
Notes
Afrikaans
af
Albanian
sq
Amharic
am
Arabic
ar
Armenian
hy
Assamese
as
Azerbaijani
az
Basque
eu
Bengali
bn
Bosnian
bs
Bulgarian
bg
Burmese
my
７ Note
You can additionally find the language support for the Preview API in the second tab.
NER language support
Generally Available API
ﾉ
Expand table
\nLanguage
Language Code
Notes
Catalan
ca
Chinese (Simplified)
zh-Hans
zh  also accepted
Chinese (Traditional)
zh-Hant
Croatian
hr
Czech
cs
Danish
da
Dutch
nl
English
en
Estonian
et
Finnish
fi
French
fr
Galician
gl
Georgian
ka
German
de
Greek
el
Gujarati
gu
Hebrew
he
Hindi
hi
Hungarian
hu
Indonesian
id
Irish
ga
Italian
it
Japanese
ji
Kannada
kn
Kazakh
kk
Khmer
km
\nLanguage
Language Code
Notes
Korean
ko
Kurdish (Kurmanji)
ku
Kyrgyz
ky
Lao
lo
Latvian
lv
Lithuanian
lt
Macedonian
mk
Malagasy
mg
Malay
ms
Malayalam
ml
Marathi
mr
Mongolian
mn
Nepali
ne
Norwegian (Bokmal)
no
nb  also accepted
Odia
or
Pashto
ps
Persian
fa
Polish
pl
Portuguese (Brazil)
pt-BR
Portuguese (Portugal)
pt-PT
pt  also accepted
Punjabi
pa
Romanian
ro
Russian
ru
Serbian
sr
Slovak
sk
Slovenian
sl
\nLanguage
Language Code
Notes
Somali
so
Spanish
es
Swahili
sw
Swazi
ss
Swedish
sv
Tamil
ta
Telugu
te
Thai
th
Turkish
tr
Ukrainian
uk
Urdu
ur
Uyghur
ug
Uzbek
uz
Vietnamese
vi
Welsh
cy
NER feature overview
Next steps
\nTransparency note for Named Entity
Recognition including Personally
Identifiable Information (PII)
06/24/2025
An AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Creating a system that
is fit for its intended purpose requires an understanding of how the technology works, its
capabilities and limitations, and how to achieve the best performance. Microsoft's Transparency
Notes are intended to help you understand how our AI technology works, the choices system
owners can make that influence system performance and behavior, and the importance of
thinking about the whole system, including the technology, the people, and the environment.
You can use Transparency Notes when developing or deploying your own system, or share
them with the people who will use or be affected by your system.
Microsoft's Transparency notes are part of a broader effort at Microsoft to put our AI principles
into practice. To find out more, see Responsible AI principles from Microsoft.
Azure AI Language supports named entity recognition to identify and categorize information in
your text. These include general entities such as Product and Event and Personally Identifiable
Information (PII) entities. A wide variety of personal entities such as names, organizations,
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a transparency note?
） Important
This article assumes that you're familiar with guidelines and best practices for Azure AI
Language. For more information, see Transparency note for Azure AI Language.
Introduction to Named Entity Recognition and
Personally Identifiable Information (PII)
\naddresses, phone numbers, financial account numbers or codes and government and country
or region specific identification numbers can be recognized. A subset of these personal entities
is protected health information (PHI). If you specify domain=phi in your request, you will only
get the PHI entities returned. The full list of PII and PHI entity categories can be found in the
table here. In addition, the PII recognition supports the ability to specify specific entity
categories you want in the response and redact PII entities in the response. The PII entities will
be replaced by asterisks in the redactedText  property of the response.
Read example NER request and example response to see how to send text to the service and
what to expect back.
Customers may want to recognize various categories of named entities two main reasons:
Enhance search capabilities - Customers can build knowledge graphs based on entities
detected in documents to enhance document search.
Enhance or automate business processes - For example, when reviewing insurance
claims, recognized entities like name and location could be highlighted to facilitate the
review. Or a support ticket could be generated with a customer's name and company
automatically from an email.
Customers may want to recognize various categories of PII entities specifically for several
reasons:
Apply sensitivity labels - For example, based on the results from the PII service, a public
sensitivity label might be applied to documents where no PII entities are detected. For
documents where US addresses and phone numbers are recognized, a confidential label
might be applied. A highly confidential label might be used for documents where bank
routing numbers are recognized.
Redact some categories of personal information from documents to protect privacy -
For example, if customer contact records are accessible to first line support
representatives, the company may want to redact unnecessary customer's personal
information from customer history to preserve the customer's privacy.
Redact personal information in order to reduce unconscious bias - For example, during
a company's resume review process, they may want to block name, address and phone
number to help reduce unconscious gender or other biases.
Replace personal information in source data for machine learning to reduce unfairness
– For example, if you want to remove names that might reveal gender when training a
machine learning model, you could use the service to identify them and you could
replace them with generic placeholders for model training.
Example use cases
\nDo not use
PII only - Do not use for automatic redaction or information classification scenarios – Any
scenario where failures to redact personal information could expose people to the risk of
identity theft and physical or psychological harms should include careful human
oversight.
NER and PII - Do not use for scenarios that use personal information for a purpose that
consent was not obtained for - For example, a company has resumes from past job
applicants. The applicants did not give their consent to be contacted for promotional
events when they submitted their resumes. Based on this scenario, both NER and PII
services should not be used to identify contact information for the purpose of inviting the
past applicants to a trade show.
NER and PII - Customers are prohibited from using of this service to harvest personal
information from publicly available content without consent from person(s) whom are the
subject of the personal information.
NER and PII - Do not use for scenarios that replace personal information in text with the
intent to mislead people.
Legal and regulatory considerations: Organizations need to evaluate potential specific legal
and regulatory obligations when using any AI services and solutions, which may not be
appropriate for use in every industry or scenario. Additionally, AI services or solutions are not
designed for and may not be used in ways prohibited in applicable terms of service and
relevant codes of conduct.
Depending on your scenario, input data and the entities you wish to extract, you could
experience different levels of performance. The following sections are designed to help you
understand key concepts about performance as they apply to using the Azure AI Language
NER and PII services.
Since both false positive and false negative errors can occur, it is important to understand how
both types of errors might affect your overall system. With Named Entity Recognition (NER), a
false positive occurs when an entity is not present in the text, but is recognized and returned
by the system. A false negative is when an entity is present in the text, but is not recognized
and returned by the system.
Considerations when choosing a use case
Characteristics and limitations
Understand and measure performance of NER