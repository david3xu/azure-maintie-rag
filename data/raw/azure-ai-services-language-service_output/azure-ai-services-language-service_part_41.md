Language
Language Code
Supported Script Code
Punjabi
pa
Guru , Latn
Queretaro Otomi
otq
Latn
Romanian
ro
Latn
Russian
ru
Cyrl
Samoan
sm
Latn
Sanscrit
sa
Deva
Santali
sat
Olck
Serbian
sr
Latn , Cyrl
Shona
sn
Latn
Sindhi
sd
Arab
Sinhala
si
Sinh
Slovak
sk
Latn
Slovenian
sl
Latn
Somali
so
Latn
Spanish
es
Latn
Sundanese
su
Latn
Swahili
sw
Latn
Swedish
sv
Latn
Tagalog
tl
Latn
Tahitian
ty
Latn
Tajik
tg
Cyrl
Tamil
ta
Taml , Latn
Tatar
tt
Cyrl
Telugu
te
Telu , Latn
Thai
th
Thai
Tibetan
bo
Tibt
\nLanguage
Language Code
Supported Script Code
Tigrinya
ti
Ethi
Tongan
to
Latn
Turkish
tr
Latn
Turkmen
tk
Latn
Upper Sorbian
hsb
Latn
Uyghur
ug
Arab
Ukrainian
uk
Latn
Urdu
ur
Arab , Latn
Uzbek
uz
Latn
Vietnamese
vi
Latn
Welsh
cy
Latn
Xhosa
xh
Latn
Yiddish
yi
Hebr
Yoruba
yo
Latn
Yucatec Maya
yua
Latn
Zulu
zu
Latn
Language
Language Code
Assamese
as
Bengali
bn
Gujarati
gu
Hindi
hi
Romanized Indic Languages supported by
Language Detection
ﾉ
Expand table
\nLanguage
Language Code
Kannada
kn
Malayalam
ml
Marathi
mr
Odia
or
Punjabi
pa
Tamil
ta
Telugu
te
Urdu
ur
Language
Script code
Scripts
Assamese
as
Latn , Beng
Bengali
bn
Latn , Beng
Gujarati
gu
Latn , Gujr
Hindi
hi
Latn , Deva
Kannada
kn
Latn , Knda
Malayalam
ml
Latn , Mlym
Marathi
mr
Latn , Deva
Odia
or
Latn , Orya
Punjabi
pa
Latn , Guru
Tamil
ta
Latn , Taml
Telugu
te
Latn , Telu
Urdu
ur
Latn , Arab
Tatar
tt
Latn , Cyrl
Script detection
ﾉ
Expand table
\nLanguage
Script code
Scripts
Serbian
sr
Latn , Cyrl
Inuktitut
iu
Latn , Cans
Language detection overview
Next steps
\nTransparency note for Language Detection
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
The language detection feature of Azure AI Language detects the language an input text is
written in and reports a single language code for every document submitted on the request in
a wide range of languages, variants, dialects, and some regional/cultural languages. The
language code is paired with a confidence score.
Be sure to check the list of supported languages to ensure the languages you need are
supported.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a transparency note?
） Important
This article assumes that you're familiar with guidelines and best practices for Azure AI
Language. For more information, see Transparency note for Azure AI Language.
Introduction to language detection
\nLanguage detection is used in multiple scenarios across a variety of industries. Some examples
include:
Preprocessing text of other Azure AI Language features. Other Azure AI Language
features require a language code to be sent in the request to identify the source
language. If you don't know the source language of your text, you can use language
detection as a pre-processor to obtain the language code.
Detect languages for business workflow. For example, if a company receives email in
various languages from customers, they could use language detection to route the emails
by language to native speakers that can communicate best with those customers.
Do not use
Do not use for automatic actions without human intervention for high risk scenarios. A
person should always review source data when another person's economic situation,
health or safety is affected.
Legal and regulatory considerations: Organizations need to evaluate potential specific legal
and regulatory obligations when using any AI services and solutions, which may not be
appropriate for use in every industry or scenario. Additionally, AI services or solutions are not
designed for and may not be used in ways prohibited in applicable terms of service and
relevant codes of conduct.
Depending on your scenario and input data, you could experience different levels of
performance. The following information is designed to help you understand key concepts
about performance as they apply to using Azure AI Language's language detection.
For inputs that include mixed-language content only a single language is returned. In
general the language with the largest representation in the content is returned, but with a
lower confidence score.
Example use cases
Considerations when choosing a use case
Characteristics and limitations
System limitations and best practices for enhancing
performance
\nThe service does not yet support the romanized versions of all languages that do not use
the Latin script. For example, Pinyin is not supported for Chinese and Franco-Arabic is not
supported for Arabic.
Some words exist in multiple languages. For example, "impossible" is common to both
English and French. For short samples that include ambiguous words, you may not get
the right language.
If you have some idea about the country or region of origin of your text, and you
encounter mixed languages, you can use the countryHint  parameter to pass in a 2 letter
country/region code.
In general longer inputs are more likely to be correctly recognized. Full phrases or
sentences are more likely to be correctly recognized than single words or sentence
fragments.
Not all languages will be recognized. Be sure to check the list of supported languages
and scripts.
To distinguish between multiple scripts used to write certain languages such as Kazakh,
the language detection feature returns a script name and script code according to the ISO
15924 standard
 for a limited set of scripts.
The service supports language detection of text only if it is in native script. For example,
Pinyin is not supported for Chinese and Franco-Arabic is not supported for Arabic.
Due to unknown gaps in our training data, certain dialects and language varieties less
represented in web data may not be properly recognized.
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for Health
Transparency note for Key Phrase Extraction
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Data Privacy and Security for Azure AI Language
Guidance for integration and responsible use with Azure AI Language
See also
\nGuidance for integration and responsible
use with Azure AI Language
06/24/2025
Microsoft wants to help you responsibly develop and deploy solutions that use Azure AI
Language. We are taking a principled approach to upholding personal agency and dignity by
considering the fairness, reliability & safety, privacy & security, inclusiveness, transparency, and
human accountability of our AI systems. These considerations are in line with our commitment
to developing Responsible AI.
This article discusses Azure AI Language features and the key considerations for making use of
this technology responsibly. Consider the following factors when you decide how to use and
implement AI-powered products and features.
When you're getting ready to deploy AI-powered products or features, the following activities
help to set you up for success:
Understand what it can do: Fully assess the capabilities of any AI model you are using to
understand its capabilities and limitations. Understand how it will perform in your
particular scenario and context.
Test with real, diverse data: Understand how your system will perform in your scenario by
thoroughly testing it with real life conditions and data that reflects the diversity in your
users, geography and deployment contexts. Small datasets, synthetic data and tests that
don't reflect your end-to-end scenario are unlikely to sufficiently represent your
production performance.
Respect an individual's right to privacy: Only collect data and information from
individuals for lawful and justifiable purposes. Only use data and information that you
have consent to use for this purpose.
Legal review: Obtain appropriate legal advice to review your solution, particularly if you
will use it in sensitive or high-risk applications. Understand what restrictions you might
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
General guidelines
\nneed to work within and your responsibility to resolve any issues that might come up in
the future. Do not provide any legal advice or guidance.
System review: If you're planning to integrate and responsibly use an AI-powered
product or feature into an existing system of software, customers or organizational
processes, take the time to understand how each part of your system will be affected.
Consider how your AI solution aligns with Microsoft's Responsible AI principles.
Human in the loop: Keep a human in the loop, and include human oversight as a
consistent pattern area to explore. This means constant human oversight of the AI-
powered product or feature and maintaining the role of humans in decision-making.
Ensure you can have real-time human intervention in the solution to prevent harm. This
enables you to manage where the AI model doesn't perform as required.
Security: Ensure your solution is secure and has adequate controls to preserve the
integrity of your content and prevent unauthorized access.
Customer feedback loop: Provide a feedback channel that allows users and individuals to
report issues with the service once it's been deployed. Once you've deployed an AI-
powered product or feature it requires ongoing monitoring and improvement – be ready
to implement any feedback and suggestions for improvement.
Microsoft Responsible AI principles
Microsoft Responsible AI resources
Microsoft Azure Learning courses on Responsible AI
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for the health feature
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Data Privacy and Security for Azure AI Language
Learn more about Responsible AI
See also
\nData, privacy, and security for Azure AI
Language
06/24/2025
This article provides details regarding how Azure AI Language processes your data. Azure AI
Language is designed with compliance, privacy, and security in mind. However, you are
responsible for its use and the implementation of this technology. It's your responsibility to
comply with all applicable laws and regulations in your jurisdiction.
Azure AI Language processes text data that is sent by the customer to the system for the
purposes of getting a response from one of the available features.
All results of the requested feature are sent back to the customer in the API response as
specified in the API reference. For example, if Language Detection is requested, the
language code is returned along with a confidence score for each text record.
Azure AI Language uses aggregate telemetry such as which APIs are used and the
number of calls from each subscription and resource for service monitoring purposes.
Azure AI Language doesn't store or process customer data outside the region where the
customer deploys the service instance.
Azure AI Language encrypts all content, including customer data, at rest.
Data sent in synchronous or asynchronous calls may be temporarily stored by Azure AI
Language for up to 48 hours only and is purged thereafter. This data is encrypted and is
only accessible to authorized on call engineers when service support is needed for
debugging purposes in the event of a catastrophic failure. To prevent this temporary
storage of input data, the LoggingOptOut query parameter can be set accordingly. By
default, this parameter is set to false for Language Detection, Key Phrase Extraction,
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What data does Azure AI Language process and
how does it process it?
How is data retained and what customer controls
are available?