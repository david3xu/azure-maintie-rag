What is language detection in Azure AI
Language?
Article • 02/21/2025
Language detection is one of the features offered by Azure AI Language, a collection of
machine learning and AI algorithms in the cloud for developing intelligent applications
that involve written language. Language detection is able to detect more than 100
languages in their primary script. In addition, it offers script detection to detect
supported scripts for each detected language according to the ISO 15924 standard
for a select number of languages supported by Azure AI Language Service.
This documentation contains the following types of articles:
Quickstarts are getting-started instructions to guide you through making requests
to the service.
How-to guides contain instructions for using the service in more specific or
customized ways.
Language detection: Returns one predominant language for each document you
submit, along with its ISO 639-1 name, a human-readable name, confidence score,
script name and script code according to ISO 15924 standard.
Script detection: To distinguish between multiple scripts used to write certain
languages, such as Kazakh, language detection returns a script name and script
code according to the ISO 15924 standard.
Ambiguous content handling: To help disambiguate language based on the input,
you can specify an ISO 3166-1 alpha-2 country/region code. For example, the word
"communication" is common to both English and French. Specifying the origin of
the text as France can help the language detection model determine the correct
language.
To use this feature, you submit data for analysis and handle the API output in your
application. Analysis is performed as-is, with no added customization to the model used
on your data.
Language detection features
Typical workflow
\n1. Create an Azure AI Language resource, which grants you access to the features
offered by Azure AI Language. It generates a password (called a key) and an
endpoint URL that you use to authenticate API requests.
2. Create a request using either the REST API or the client library for C#, Java,
JavaScript, and Python. You can also send asynchronous calls with a batch request
to combine API requests for multiple features into a single call.
3. Send the request containing your text data. Your key and endpoint are used for
authentication.
4. Stream or store the response locally.
To use language detection, you submit raw unstructured text for analysis and handle the
API output in your application. Analysis is performed as-is, with no additional
customization to the model used on your data. There are three ways to use language
detection:
Development
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use entity linking
with text examples with your own data when you sign up. For more
information, see the Azure AI Foundry website
 or Azure AI Foundry
documentation.
REST API or Client
library (Azure SDK)
Integrate language detection into your applications using the REST API, or
the client library available in a variety of languages. For more information,
see the language detection quickstart.
Docker container
Use the available Docker container to deploy this feature on-premises.
These docker containers enable you to bring the service closer to your
data for compliance, security, or other operational reasons.
An AI system includes not only the technology, but also the people who will use it, the
people who will be affected by it, and the environment in which it's deployed. Read the
transparency note for language detection to learn about responsible AI use and
Get started with language detection
ﾉ
Expand table
Responsible AI
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
deployment in your systems. You can also see the following articles for more
information:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
There are two ways to get started using the entity linking feature:
Azure AI Foundry is a web-based platform that lets you use several Language
service features without needing to write code.
The quickstart article for instructions on making requests to the service using the
REST API and client library SDK.
Next steps
Yes
No
\nQuickstart: using the Language
Detection client library and REST API
Article • 02/17/2025
Create a Project in Foundry in the Azure AI Foundry Portal
Using the left side pane, select Playgrounds. Then select the Try the Language
Playground button.
Prerequisites
Navigate to the Azure AI Foundry Playground

\n![Image](images/page394_image1.png)
\nThe Language Playground consists of four sections:
Top banner: You can select any of the currently available Language services here.
Right pane: This pane is where you can find the Configuration options for the
service, such as the API and model version, along with features specific to the
service.
Center pane: This pane is where you enter your text for processing. After the
operation is run, some results are shown here.
Right pane: This pane is where Details of the run operation are shown.
Here you can select the Language Detection capability by choosing the top banner tile,
Detect language.
Detect language is designed to identify the language typed in text.
In Configuration there are the following options:
Option
Description
Select API version
Select which version of the API to use.
Select model version
Select which version of the model to use.
Select country hint
Select the origin country/region of the input text.
After your operation is completed, the Details section contains the following fields for
the most detected language and script:
Field
Description
Sentence
Iso 639-1 Code
The ISE 639-1 code for the most detected language.
Confidence Score
How confident the model is in the correctness of identification of the most
Use Language Detection in the Azure AI
Foundry Playground
Use Detect language
ﾉ
Expand table
ﾉ
Expand table
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Field
Description
typed language.
Script Name
The name of the most detected script in the text.
Iso 15924 Script
Code
The ISO 15924 script code for the most detected script.
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other
resources associated with it.
Azure portal
Azure CLI
Language detection overview

Clean up resources
Next steps
Yes
No
\n![Image](images/page396_image1.png)
\nLanguage support for Language Detection
06/30/2025
Use this article to learn which natural languages that language detection supports.
The Language Detection feature can detect a wide range of languages, variants, dialects, and
some regional/cultural languages, and return detected languages with their name and code.
The returned language code parameters conform to BCP-47
 standard with most of them
conforming to ISO-639-1
 identifiers.
If you have content expressed in a less frequently used language, you can try Language
Detection to see if it returns a code. The response for languages that can't be detected is
unknown .
Language
Language Code
Supported Script Code
Afrikaans
af
Latn
Albanian
sq
Latn
Amharic
am
Ethi
Arabic
ar
Arab
Armenian
hy
Armn
Assamese
as
Beng , Latn
Azerbaijani
az
Latn
Bashkir
ba
Cyrl
Basque
eu
Latn
Belarusian
be
Cyrl
Bengali
bn
Beng , Latn
Bhojpuri
bho
Deva
Bodo
brx
Deva
Bosnian
bs
Latn
Languages supported by Language Detection
ﾉ
Expand table
\nLanguage
Language Code
Supported Script Code
Bulgarian
bg
Cyrl
Burmese
my
Mymr
Catalan
ca
Latn
Central Khmer
km
Khmr
Checheni
ce
Cyrl
Chhattisgarhi
hne
Deva
Chinese Literal
lzh
Hani
Chinese Simplified
zh_chs
Hans
Chinese Traditional
zh_cht
Hant
Chuvash
cv
Cyrl
Corsican
co
Latn
Croatian
hr
Latn
Czech
cs
Latn
Danish
da
Latn
Dari
prs
Arab
Divehi
dv
Thaa
Dogri
dgo
Deva
Dutch
nl
Latn
English
en
Latn
Esperanto
eo
Latn
Estonian
et
Latn
Faroese
fo
Latn
Fijian
fj
Latn
Finnish
fi
Latn
French
fr
Latn
Galician
gl
Latn
\nLanguage
Language Code
Supported Script Code
Georgian
ka
Gujr
German
de
Latn
Greek
el
Grek
Gujarati
gu
Gujr , Latn
Haitian
ht
Latn
Hausa
ha
Latn
Hebrew
he
Hebr
Hindi
hi
Deva , Latn
Hmong Daw
mww
Latn
Hungarian
hu
Latn
Icelandic
is
Latn
Igbo
ig
Latn
Indonesian
id
Latn
Inuktitut
iu
Cans , Latn
Inuinnaqtun
ikt
Latn
Irish
ga
Latn
Italian
it
Latn
Japanese
ja
Jpan
Javanese
jv
Latn
Kannada
kn
Knda , Latn
Kashmiri
ks
Arab , Deva , Shrd
Kazakh
kk
Cyrl
Kinyarwanda
rw
Latn
Kirghiz
ky
Cyrl
Konkani
gom
Deva
Korean
ko
Hang
\nLanguage
Language Code
Supported Script Code
Kurdish
ku
Arab
Kurdish (Northern)
kmr
Latn
Lao
lo
Laoo
Latin
la
Latn
Latvian
lv
Latn
Lithuanian
lt
Latn
Lower Siberian
dsb
Latn
Luxembourgish
lb
Latn
Macedonian
mk
Cyrl
Maithili
mai
Deva
Malagasy
mg
Latn
Malay
ms
Latn
Malayalam
ml
Mlym , Latn
Maltese
mt
Latn
Maori
mi
Latn
Marathi
mr
Deva , Latn
Meitei
mni
Mtei
Mongolian
mn
Cyrl , Mong
Nepali
ne
Deva
Norwegian
no
Latn
Norwegian Nynorsk
nn
Latn
Odia
or
Orya , Latn
Pashto
ps
Arab
Persian
fa
Arab
Polish
pl
Latn
Portuguese
pt
Latn