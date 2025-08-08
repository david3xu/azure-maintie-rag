Bushel
Hogshead
For Weight:
Kilogram
Gram
Milligram
Microgram
Gallon
MetricTon
Ton
Pound
Ounce
Grain
Pennyweight
LongTonBritish
ShortTonUS
ShortHundredweightUS
Stone
Dram
Examples: "3rd", "first", "last"
JSON
Possible values for "relativeTo":
Start
End
Examples: "88 deg fahrenheit", "twenty three degrees celsius"
Ordinal
"metadata": {
                "offset": "3",
                "relativeTo": "Start",
                "value": "3"
            }
Temperature
\nJSON
Possible values for "unit":
Celsius
Fahrenheit
Kelvin
Rankine
"metadata": {
                "unit": "Fahrenheit",
                "value": 88
            }
\nEntity types and tags
Article • 04/29/2025
Use this article to get an overview of the new API changes starting from version 2024-11-01 .
This API change mainly introduces two new concepts ( entity types  and entity tags )
replacing the category  and subcategory  fields in the current Generally Available API. A detailed
overview of each API parameter and the supported API versions it corresponds to can be found
on the [Skill Parameters][../how-to/skill-parameters.md] page.
Since an entity like “Seattle” could be classified as a City, GPE (Geo Political Entity), and a
Location, the type  attribute is used to define the most granular classification, in this case City.
The tags  attribute in the service output is a list all possible classifications (City, GPE, and
Location) and their respective confidence score. A full mapping of possible tags for each type
can be found below. The metadata  attributes in the service output contains additional
information about the entity, such as the integer value associated with the entity.
Entity types represent the lowest (or finest) granularity at which the entity has been detected
and can be considered to be the base class that has been detected.
Entity tags are used to further identify an entity where a detected entity is tagged by the entity
type and additional tags to differentiate the identified entity. The entity tags list could be
considered to include categories, subcategories, sub-subcategories, and so on.
The changes introduce better flexibility for the named entity recognition service, including:
Updates to the structure of input formats: • InclusionList • ExclusionList • Overlap policy
Updates to the handling of output formats:
More granular entity recognition outputs through introducing the tags list where an
entity could be tagged by more than one entity tag.
Overlapping entities where entities could be recognized as more than one entity type and
if so, this entity would be returned twice. If an entity was recognized to belong to two
Entity types
Entity tags
Changes from versions 2022-05-01  and 2023-04-01
to version 2024-11-01  API
\nentity tags under the same entity type, both entity tags are returned in the tags list.
Filtering entities using entity tags, you can learn more about this by navigating to this
article.
Metadata Objects which contain additional information about the entity but currently
only act as a wrapper for the existing entity resolution feature. You can learn more about
this new feature here.
You can see a comparison between the structure of the entity categories/types in the
Supported Named Entity Recognition (NER) entity categories and entity types article. Below is a
table describing the mappings between the results you would expect to see from versions
2022-05-01  and 2023-04-01  and the current version API.
Type
Tags
Date
Temporal, Date
DateRange
Temporal, DateRange
DateTime
Temporal, DateTime
DateTimeRange
Temporal, DateTimeRange
Duration
Temporal, Duration
SetTemporal
Temporal, SetTemporal
Time
Temporal, Time
TimeRange
Temporal, TimeRange
City
GPE, Location, City
State
GPE, Location, State
CountryRegion
GPE, Location, CountryRegion
Continent
GPE, Location, Continent
GPE
Location, GPE
Location
Location
Versions 2022-05-01  and 2023-04-01  to current
version API entity mappings
ﾉ
Expand table
\nType
Tags
Airport
Structural, Location
Structural
Location, Structural
Geological
Location, Geological
Age
Numeric, Age
Currency
Numeric, Currency
Number
Numeric, Number
PhoneNumber
PhoneNumber
NumberRange
Numeric, NumberRange
Percentage
Numeric, Percentage
Ordinal
Numeric, Ordinal
Temperature
Numeric, Dimension, Temperature
Speed
Numeric, Dimension, Speed
Weight
Numeric, Dimension, Weight
Height
Numeric, Dimension, Height
Length
Numeric, Dimension, Length
Volume
Numeric, Dimension, Volume
Area
Numeric, Dimension, Area
Information
Numeric, Dimension, Information
Address
Address
Person
Person
PersonType
PersonType
Organization
Organization
Product
Product
ComputingProduct
Product, ComputingProduct
IP
IP
Email
Email
\nType
Tags
URL
URL
Skill
Skill
Event
Event
CulturalEvent
Event, CulturalEvent
SportsEvent
Event, SportsEvent
NaturalEvent
Event, NaturalEvent
\nExtract information in Excel using
Named Entity Recognition(NER) and
Power Automate
Article • 03/24/2025
In this tutorial, you create a Power Automate flow to extract text in an Excel spreadsheet
without having to write code.
This flow takes a spreadsheet of issues reported about an apartment complex, and
classify them into two categories: plumbing and other. It also extracts the names and
phone numbers of the tenants who sent them. Lastly, the flow appends this information
to the Excel sheet.
In this tutorial, you learn how to:
A Microsoft Azure account. Create a free account
 or sign in
.
A Language resource. If you don't have one, you can create one in the Azure
portal
 and use the free tier to complete this tutorial.
The key and endpoint that was generated for you when you created the resource.
A spreadsheet containing tenant issues. Example data for this tutorial is available
on GitHub
.
Microsoft 365, with OneDrive for business
.
Download the example Excel file from GitHub
. This file must be stored in your
OneDrive for Business account.
Use Power Automate to create a flow
＂
Upload Excel data from OneDrive for Business
＂
Extract text from Excel, and send it for Named Entity Recognition(NER)
＂
Use the information from the API to update an Excel sheet.
＂
Prerequisites
Add the Excel file to OneDrive for Business

\n![Image](images/page617_image1.png)
\nThe issues are reported in raw text. We use the NER feature to extract the person name
and phone number. Then the flow looks for the word "plumbing" in the description to
categorize the issues.
Go to the Power Automate site
, and log in. Then select Create and Scheduled flow.
On the Build a scheduled cloud flow page, initialize your flow with the following fields:
Field
Value
Flow name
Scheduled Review or another name.
Starting
Enter the current date and time.
Repeat every
1 hour
Create a new Power Automate workflow

ﾉ
Expand table
Add variables to the flow
\n![Image](images/page618_image1.png)
\nCreate variables representing the information that is added to the Excel file. Select New
Step and search for Initialize variable. Do this four times, to create four variables.
Add the following information to the variables you created. They represent the columns
of the Excel file. If any variables are collapsed, you can select them to expand them.
Action
Name
Type
Value
Initialize variable
var_person
String
Person
Initialize variable 2
var_phone
String
Phone Number
Initialize variable 3
var_plumbing
String
plumbing
Initialize variable 4
var_other
String
other

ﾉ
Expand table
\n![Image](images/page619_image1.png)
\nSelect New Step and type Excel, then select List rows present in a table from the list of
actions.

Read the excel file
\n![Image](images/page620_image1.png)