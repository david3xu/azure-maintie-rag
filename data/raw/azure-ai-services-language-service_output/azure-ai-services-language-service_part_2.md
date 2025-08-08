Conversational language understanding (CLU) enables users to build custom natural language
understanding models to predict the overall intention of an incoming utterance and extract
important information from it.
Orchestration workflow is a custom feature that enables you to connect Conversational
Language Understanding (CLU), question answering, and LUIS applications.
Orchestration workflow

Question answering
\n![Image](images/page11_image1.png)
\nQuestion answering is a custom feature that identifies the most suitable answer for user inputs.
This feature is typically utilized to develop conversational client applications, including social
media platforms, chat bots, and speech-enabled desktop applications.
This section helps you decide which Language service feature you should use for your
application:
What do you want to do?
Document
format
Your best solution
Is this solution
customizable?*
Detect and/or redact sensitive
information such as PII  and PHI .
Unstructured
text,
transcribed
conversations
PII detection

Which Language service feature should I use?
ﾉ
Expand table
\n![Image](images/page12_image1.png)
\nWhat do you want to do?
Document
format
Your best solution
Is this solution
customizable?*
Extract categories of information
without creating a custom model.
Unstructured text
The preconfigured
NER feature
Extract categories of information
using a model specific to your data.
Unstructured text
Custom NER
✓
Extract main topics and important
phrases.
Unstructured text
Key phrase extraction
Determine the sentiment and
opinions expressed in text.
Unstructured text
Sentiment analysis
and opinion mining
Summarize long chunks of text or
conversations.
Unstructured
text,
transcribed
conversations.
Summarization
Disambiguate entities and get links
to Wikipedia.
Unstructured text
Entity linking
Classify documents into one or more
categories.
Unstructured text
Custom text
classification
✓
Extract medical information from
clinical/medical documents, without
building a model.
Unstructured text
Text analytics for
health
Build a conversational application
that responds to user inputs.
Unstructured
user inputs
Question answering
✓
Detect the language that a text was
written in.
Unstructured text
Language detection
Predict the intention of user inputs
and extract information from them.
Unstructured
user inputs
Conversational
language
understanding
✓
Connect apps from conversational
language understanding, LUIS, and
question answering.
Unstructured
user inputs
Orchestration
workflow
✓
* If a feature is customizable, you can train an AI model using our tools to fit your data
specifically. Otherwise a feature is preconfigured, meaning the AI models it uses can't be
changed. You just send your data, and use the feature's output in your applications.
\nAzure AI Language unifies three individual language services in Azure AI services - Text
Analytics, QnA Maker, and Language Understanding (LUIS). If you have been using these three
services, you can easily migrate to the new Azure AI Language. For instructions see Migrating
to Azure AI Language.
After you get started with the Language service quickstarts, try our tutorials that show you how
to solve various scenarios.
Extract key phrases from text stored in Power BI
Use Power Automate to sort information in Microsoft Excel
Use Flask to translate text, analyze sentiment, and synthesize speech
Use Azure AI services in canvas apps
Create an FAQ Bot
You can find more code samples on GitHub for the following languages:
C#
Java
JavaScript
Python
Use Language service containers to deploy API features on-premises. These Docker containers
enable you to bring the service closer to your data for compliance, security, or other
operational reasons. The Language service offers the following containers:
Sentiment analysis
Language detection
Key phrase extraction
Custom Named Entity Recognition
Text Analytics for health
Summarization
Migrate from Text Analytics, QnA Maker, or
Language Understanding (LUIS)
Tutorials
Code samples
Deploy on premises using Docker containers
\nAn AI system includes not only the technology, but also the people who use it, the people
affected by it, and the deployment environment. Read the following articles to learn about
responsible AI use and deployment in your systems:
Transparency note for the Language service
Integration and responsible use
Data, privacy, and security
Responsible AI
\nLanguage support for Language features
06/04/2025
Use this article to learn about the languages currently supported by different features.
Language
Language
code
Custom text
classification
Custom named
entity
recognition(NER)
Conversational
language
understanding
Entity
linking
Language
detection
Key
phrase
extraction
Named entity
recognition(NER)
Orchestration
workflow
Person
Identifia
Informa
(PII)
Afrikaans
af
✓
✓
✓
✓
✓
✓
Albanian
sq
✓
✓
✓
✓
✓
✓
Amharic
am
✓
✓
✓
✓
✓
✓
Arabic
ar
✓
✓
✓
✓
✓
✓
✓
Armenian
hy
✓
✓
✓
✓
✓
✓
Assamese
as
✓
✓
✓
✓
✓
✓
Azerbaijani
az
✓
✓
✓
✓
✓
✓
Basque
eu
✓
✓
✓
✓
✓
✓
Belarusian
be
✓
✓
✓
✓
✓
Bengali
bn
✓
✓
✓
✓
✓
✓
Bosnian
bs
✓
✓
✓
✓
✓
✓
Breton
br
✓
✓
✓
✓
✓
Bulgarian
bg
✓
✓
✓
✓
✓
✓
Burmese
my
✓
✓
✓
✓
✓
✓
Catalan
ca
✓
✓
✓
✓
✓
✓
Central Khmer
km
✓
Chinese
(Simplified)
zh-hans
✓
✓
✓
✓
✓
✓
✓
Chinese
(Traditional)
zh-hant
✓
✓
✓
✓
✓
✓
✓
Corsican
co
✓
Croatian
hr
✓
✓
✓
✓
✓
✓
Czech
cs
✓
✓
✓
✓
✓
✓
✓
Danish
da
✓
✓
✓
✓
✓
✓
✓
Dari
prs
✓
Divehi
dv
✓
Dutch
nl
✓
✓
✓
✓
✓
✓
✓
English (UK)
en-gb
✓
✓
✓
✓
English (US)
en-us
✓
✓
✓
✓
✓
✓
✓
✓
✓
Esperanto
eo
✓
✓
✓
✓
✓
Estonian
et
✓
✓
✓
✓
✓
✓
Fijian
fj
✓
Filipino
tl
✓
✓
✓
✓
✓
ﾉ
Expand table
\nLanguage
Language
code
Custom text
classification
Custom named
entity
recognition(NER)
Conversational
language
understanding
Entity
linking
Language
detection
Key
phrase
extraction
Named entity
recognition(NER)
Orchestration
workflow
Person
Identifia
Informa
(PII)
Finnish
fi
✓
✓
✓
✓
✓
✓
✓
French
fr
✓
✓
✓
✓
✓
✓
✓
✓
Galician
gl
✓
✓
✓
✓
✓
✓
Georgian
ka
✓
✓
✓
✓
✓
✓
German
de
✓
✓
✓
✓
✓
✓
✓
✓
Greek
el
✓
✓
✓
✓
✓
✓
Gujarati
gu
✓
✓
✓
✓
✓
✓
Haitian
ht
✓
Hausa
ha
✓
✓
✓
✓
✓
Hebrew
he
✓
✓
✓
✓
✓
✓
✓
Hindi
hi
✓
✓
✓
✓
✓
✓
✓
Hmong Daw
mww
✓
Hungarian
hu
✓
✓
✓
✓
✓
✓
✓
Icelandic
is
✓
Igbo
ig
✓
Indonesian
id
✓
✓
✓
✓
✓
✓
Inuktitut
iu
✓
Irish
ga
✓
✓
✓
✓
✓
✓
Italian
it
✓
✓
✓
✓
✓
✓
✓
✓
Japanese
ja
✓
✓
✓
✓
✓
✓
✓
Javanese
jv
✓
✓
✓
✓
✓
Kannada
kn
✓
✓
✓
✓
✓
✓
Kazakh
kk
✓
✓
✓
✓
✓
✓
Khmer
km
✓
✓
✓
✓
✓
✓
Kinyarwanda
rw
✓
Korean
ko
✓
✓
✓
✓
✓
✓
✓
Kurdish
(Kurmanji)
ku
✓
✓
✓
✓
✓
✓
Kyrgyz
ky
✓
✓
✓
✓
✓
✓
Lao
lo
✓
✓
✓
✓
✓
✓
Latin
la
✓
✓
✓
✓
✓
Latvian
lv
✓
✓
✓
✓
✓
✓
Lithuanian
lt
✓
✓
✓
✓
✓
✓
Luxembourgish
lb
✓
Macedonian
mk
✓
✓
✓
✓
✓
✓
Malagasy
mg
✓
✓
✓
✓
✓
✓
Malay
ms
✓
✓
✓
✓
✓
✓
Malayalam
ml
✓
✓
✓
✓
✓
✓
\nLanguage
Language
code
Custom text
classification
Custom named
entity
recognition(NER)
Conversational
language
understanding
Entity
linking
Language
detection
Key
phrase
extraction
Named entity
recognition(NER)
Orchestration
workflow
Person
Identifia
Informa
(PII)
Maltese
mt
✓
Maori
mi
✓
Marathi
mr
✓
✓
✓
✓
✓
✓
Mongolian
mn
✓
✓
✓
✓
✓
✓
Nepali
ne
✓
✓
✓
✓
✓
✓
Norwegian
(Bokmal)
nb
✓
✓
✓
✓
✓
✓
✓
Norwegian
no
✓
Norwegian
Nynorsk
nn
✓
Odia
or
✓
✓
✓
✓
✓
✓
Oromo
om
✓
Pashto
ps
✓
✓
✓
✓
✓
✓
Persian
fa
✓
✓
✓
✓
✓
✓
Polish
pl
✓
✓
✓
✓
✓
✓
✓
Portuguese
(Brazil)
pt-br
✓
✓
✓
✓
✓
✓
✓
✓
Portuguese
(Portugal)
pt-pt
✓
✓
✓
✓
✓
✓
✓
Punjabi
pa
✓
✓
✓
✓
✓
✓
Queretaro
Otomi
otq
✓
Romanian
ro
✓
✓
✓
✓
✓
✓
Russian
ru
✓
✓
✓
✓
✓
✓
✓
Samoan
sm
✓
Sanskrit
sa
✓
✓
✓
✓
✓
Scottish Gaelic
gd
✓
✓
✓
✓
✓
Serbian
sr
✓
✓
✓
✓
✓
✓
Shona
sn
✓
Sindhi
sd
✓
✓
✓
✓
✓
Sinhala
si
✓
✓
✓
✓
✓
Slovak
sk
✓
✓
✓
✓
✓
✓
Slovenian
sl
✓
✓
✓
✓
✓
✓
Somali
so
✓
✓
✓
✓
✓
✓
Spanish
es
✓
✓
✓
✓
✓
✓
✓
✓
✓
Sundanese
su
✓
✓
✓
✓
✓
Swahili
sw
✓
✓
✓
✓
✓
✓
Swati
ss
✓
Swedish
sv
✓
✓
✓
✓
✓
✓
✓
Tahitian
ty
✓
\nLanguage
Language
code
Custom text
classification
Custom named
entity
recognition(NER)
Conversational
language
understanding
Entity
linking
Language
detection
Key
phrase
extraction
Named entity
recognition(NER)
Orchestration
workflow
Person
Identifia
Informa
(PII)
Tajik
tg
✓
Tamil
ta
✓
✓
✓
✓
✓
✓
Tatar
tt
✓
Telugu
te
✓
✓
✓
✓
✓
✓
Thai
th
✓
✓
✓
✓
✓
✓
Tibetan
bo
✓
Tigrinya
ti
✓
Tongan
to
✓
Turkish
tr
✓
✓
✓
✓
✓
✓
✓
Turkmen
tk
✓
Ukrainian
uk
✓
✓
✓
✓
✓
✓
Urdu
ur
✓
✓
✓
✓
✓
✓
Uyghur
ug
✓
✓
✓
✓
✓
✓
Uzbek
uz
✓
✓
✓
✓
✓
✓
Vietnamese
vi
✓
✓
✓
✓
✓
✓
Welsh
cy
✓
✓
✓
✓
✓
✓
Western Frisian
fy
✓
✓
✓
✓
✓
Xhosa
xh
✓
✓
✓
✓
✓
Yiddish
yi
✓
✓
✓
✓
✓
Yoruba
yo
✓
Yucatec Maya
yua
✓
Zulu
zu
✓
✓
✓
✓
See the following service-level language support articles for more information on language support:
Custom text classification
Custom named entity recognition(NER)
Conversational language understanding
Entity linking
Language detection
Key phrase extraction
Named entity recognition(NER)
Orchestration workflow
Personally Identifiable Information (PII)
Conversation PII
Question answering
Sentiment analysis
Opinion mining
Text Analytics for health
Summarization
Conversation summarization
See also
\nLanguage service supported regions
Article • 05/09/2025
The Language service is available for use in several Azure regions. Use this article to learn
about the regional support and limitations.
Typically you can refer to the region support
 for details, and most Language service
capabilities are available in all supported regions. Some Language service capabilities, however,
are only available in select regions which are listed below.
Conversational language understanding and orchestration workflow are only available in some
Azure regions. Some regions are available for both authoring and prediction, while other
regions are prediction only. Language resources in authoring regions allow you to create, edit,
train, and deploy your projects. Language resources in prediction regions allow you to get
predictions from a deployment.
Region
Authoring
Prediction
Australia East
✓
✓
Brazil South
✓
Canada Central
✓
✓
Canada East
✓
Central India
✓
✓
Central US
✓
Region support overview
７ Note
Language service doesn't store or process customer data outside the region you deploy
the service instance in.
Conversational language understanding and
orchestration workflow
ﾉ
Expand table