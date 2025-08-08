Feedback
The Azure AI containers are required to submit metering information for billing
purposes. Failure to allowlist various network channels that the Azure AI containers rely
on will prevent the container from working.
The host should allowlist port 443 and the following domains:
*.cognitive.microsoft.com
*.cognitiveservices.azure.com
Deep packet inspection (DPI)
 is a type of data processing that inspects in detail the
data sent over a computer network, and usually takes action by blocking, rerouting, or
logging it accordingly.
Disable DPI on the secure channels that the Azure AI containers create to Microsoft
servers. Failure to do so will prevent the container from functioning correctly.
Developer samples are available at our GitHub repository
.
Learn about container recipes you can use with the Azure AI services.
Install and explore the functionality provided by containers in Azure AI services:
Anomaly Detector containers
Azure AI Vision containers
Language Understanding (LUIS) containers
Speech Service API containers
Language service containers
Translator containers
Container networking
Allowlist Azure AI services domains and ports
Disable deep packet inspection
Developer samples
Next steps
\nWas this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Yes
No
\nSupported Named Entity Recognition
(NER) entity categories and entity types
05/23/2025
Use this article to find the entity categories that can be returned by Named Entity Recognition
(NER). NER runs a predictive model to identify and categorize named entities from an input
document.
This type contains the following entity:
Entity
Person
Details
Names of people.
Supported document languages
ar , cs , da , nl , en , fi , fr , de , he ,
hu , it , ja , ko , no , pl , pt-br , pt - pt , ru , es , sv , tr
This type contains the following entity:
Entity
PersonType
７ Note
Starting from API version 2023-04-15-preview, the category and subcategory fields
are replaced with entity types and tags to introduce better flexibility.
Generally Available API
Type: Person
Type: PersonType
\nDetails
Job types or roles held by a person
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt , pt-br
This type contains the following entity:
Entity
Location
Details
Natural and human-made landmarks, structures, geographical features, and geopolitical
entities.
Supported document languages
ar , cs , da , nl , en , fi , fr , de , he , hu , it , ja , ko , no , pl , pt-br , pt-pt , ru , es , sv , tr
The entity in this type can have the following subtypes.
Entity subtype
Geopolitical Entity (GPE)
Details
Cities, countries/regions, states.
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt , pt-br
Structural
Manmade structures.
Type: Location
Subtype
\nen
Geographical
Geographic and natural features such as rivers, oceans, and deserts.
en
This type contains the following entity:
Entity
Organization
Details
Companies, political groups, musical bands, sport clubs, government bodies, and public
organizations. Nationalities and religions are not included in this entity type.
Supported document languages
ar , cs , da , nl , en , fi , fr , de , he , hu , it , ja , ko , no , pl , pt-br , pt-pt , ru , es , sv , tr
The entity in this type can have the following subtype.
Entity subtype
Medical
Details
Medical companies and groups.
Supported document languages
en
Stock exchange
Stock exchange groups.
Type: Organization
Subtype
\nen
Sports
Sports-related organizations.
en
This type contains the following entity:
Entity
Event
Details
Historical, social, and naturally occurring events.
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt  and pt-br
The entity in this type can have the following subtype.
Entity subtype
Cultural
Details
Cultural events and holidays.
Supported document languages
en
Natural
Naturally occurring events.
Type: Event
Subtypes
\nen
Sports
Sporting events.
en
This type contains the following entity:
Entity
Product
Details
Physical objects of various types.
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt , pt-br
The entity in this type can have the following subtype.
Entity subtype
Computing products
Details
Computing products.
Supported document languages
en
This type contains the following entity:
Type: Product
Subtype
Type: Skill
\nEntity
Skill
Details
A capability, skill, or expertise.
Supported document languages
en  , es , fr , de , it , pt-pt , pt-br
This type contains the following entity:
Entity
Address
Details
Full mailing address.
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt , pt-br
This type contains the following entity:
Entity
PhoneNumber
Details
Phone numbers (US and EU phone numbers only).
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt  pt-br
Type: Address
Type: PhoneNumber
\nThis type contains the following entity:
Entity
Email
Details
Email addresses.
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt , pt-br
This type contains the following entity:
Entity
URL
Details
URLs to websites.
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt , pt-br
This type contains the following entity:
Entity
IP
Details
network IP addresses.
Supported document languages
Type: Email
Type: URL
Type: IP
\nen , es , fr , de , it , zh-hans , ja , ko , pt-pt , pt-br
This type contains the following entities:
Entity
DateTime
Details
Dates and times of day.
Supported document languages
en , es , fr , de , it , zh-hans , ja , ko , pt-pt , pt-br
Entities in this type can have the following subtypes
The entity in this type can have the following subtypes.
Entity subtype
Date
Details
Calender dates.
Supported document languages
en , es , fr , de , it , zh-hans , pt-pt , pt-br
Time
Times of day.
en , es , fr , de , it , zh-hans , pt-pt , pt-br
DateRange
Type: DateTime
Subtypes