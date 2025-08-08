Extract PII from conversation is designed to identify and mask personally identifying
information in conversational text.
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
Select they types of information you want to redact.
Specify redaction
policy
Select the method of redaction.
Specify redaction
character
Select which character is used for redaction. Only available with the
CharacterMask redaction policy.
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
Length
The character length of the entity.
Confidence
How confident the model is in the correctness of identification of entity's type.
ﾉ
Expand table
ﾉ
Expand table
\nExtract PII from text is designed to identify and mask personally identifying information in text.
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
Select they types of information you want to redact.
Specify redaction
policy
Select the method of redaction.
Specify redaction
character
Select which character is used for redaction. Only available with the
CharacterMask redaction policy.
After your operation is completed, the type of entity is displayed beneath each entity in the
center pane and the Details section contains the following fields for each entity:

Extract PII from text
ﾉ
Expand table
ﾉ
Expand table
\n![Image](images/page892_image1.png)
\nField
Description
Entity
The detected entity.
Category
The type of entity that was detected.
Offset
The number of characters that the entity was detected from the beginning of the line.
Length
The character length of the entity.
Confidence
How confident the model is in the correctness of identification of entity's type.
Tags
How confident the model is in the correctness for each identified entity type.
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other resources
associated with it.
Azure portal
Azure CLI
Overview

Clean up resources
Next steps
\n![Image](images/page893_image1.png)
\nPersonally Identifiable Information (PII)
detection language support
06/04/2025
Use this article to learn which natural languages are supported by the text PII, document PII,
and conversation PII features of Azure AI Language Service.
Language
Language code
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
Catalan
ca
Chinese-Simplified
zh-hans
zh  also accepted
Chinese-Traditional
zh-hant
Text PII
Text PII language support
ﾉ
Expand table
\nLanguage
Language code
Notes
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
ja
Kannada
kn
Kazakh
kk
Khmer
km
Korean
ko
Kurdish(Kurmanji)
ku
Kyrgyz
ky
\nLanguage
Language code
Notes
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
Norwegian (Bokmål)
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
Somali
so
Spanish
es
Swahili
sw
\nLanguage
Language code
Notes
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
PII feature overview
Next steps
\nTransparency note for Personally
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
your text. The PII feature supports the detection of personal (PII) categories of entities. A wide
variety of personal entities such as names, organizations, addresses, phone numbers, financial
account numbers or codes and government and country or region specific identification
numbers can be recognized. A subset of these personal entities is protected health information
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a transparency note?
） Important
This article assumes that you're familiar with guidelines and best practices for Azure AI
Language. For more information, see Transparency note for Azure AI Language.
Introduction to the Personally Identifiable
Information (PII) feature
\n(PHI). If you specify domain=phi in your request, you will only get the PHI entities returned. The
full list of PII and PHI entity categories can be found in the table here.
Read example NER request and example response to see how to send text to the service and
what to expect back.
Customers may want to recognize various categories of PII for several reasons:
Apply sensitivity labels - For example, based on the results from the PII service, a public
sensitivity label might be applied to documents where no PII entities are detected. For
documents where US addresses and phone numbers are recognized, a confidential label
might be applied. A highly confidential label might be used for documents where bank
routing numbers are recognized.
Redact some categories of personal information from documents that get wider
circulation - For example, if customer contact records are accessible to first line support
representatives, the company may want to redact the customer's personal information
besides their name from the version of the customer history to preserve the customer's
privacy.
Redact personal information in order to reduce unconscious bias - For example, during
a company's resume review process, they may want to block name, address and phone
number to help reduce unconscious gender or other biases.
Replace personal information in source data for machine learning to reduce unfairness
– For example, if you want to remove names that might reveal gender when training a
machine learning model, you could use the service to identify them and you could
replace them with generic placeholders for model training.
Remove personal information from call center transcription – For example, if you want
to remove names or other PII data that happen between the agent and the customer in a
call center scenario. You could use the service to identify and remove them.
Avoid high-risk automatic redaction or information classification scenarios – Any
scenario where failures to redact personal information could expose people to the risk of
identity theft and physical or psychological harms should include careful human
oversight.
Avoid scenarios that use personal information for a purpose that consent was not
obtained for - For example, a company has resumes from past job applicants. The
applicants did not give their consent to be contacted for promotional events when they
submitted their resumes. Based on this scenario, the PII service should not be used to
Example use cases
Considerations when choosing a use case
\nidentify contact information for the purpose of inviting the past applicants to a trade
show.
Avoid scenarios that use the service to harvest personal information from publicly
available content.
Avoid scenarios that replace personal information in text with the intent to mislead
people.
Legal and regulatory considerations: Organizations need to evaluate potential specific
legal and regulatory obligations when using any AI services and solutions, which may not
be appropriate for use in every industry or scenario. Additionally, AI services or solutions
are not designed for and may not be used in ways prohibited in applicable terms of
service and relevant codes of conduct.
Depending on your scenario, input data and the entities you wish to extract, you could
experience different levels of performance. The following sections are designed to help you
understand key concepts about performance as they apply to using the Azure AI Language PII
service.
Since both false positive and false negative errors can occur, it is important to understand how
both types of errors might affect your overall system. In redaction scenarios, for example, false
negatives could lead to personal information leakage. For redaction scenarios, consider a
process for human review to account for this type of error. For sensitivity label scenarios, both
false positives and false negatives could lead to misclassification of documents. The audience
may unnecessarily limited for documents labelled as confidential where a false positive
occurred. PII could be leaked where a false negative occurred and a public label was applied.
You can adjust the threshold for confidence score your system uses to tune your system. If it is
more important to identify all potential instances of PII, you can use a lower threshold. This
means that you may get more false positives (non- PII data being recognized as PII entities),
but fewer false negatives (PII entities not recognized as PII). If it is more important for your
system to recognize only true PII data, you can use a higher threshold. Threshold values may
not have consistent behavior across individual categories of PII entities. Therefore, it is critical
that you test your system with real data it will process in production.
Characteristics and limitations
Understand and measure performance
System limitations and best practices for enhancing
performance