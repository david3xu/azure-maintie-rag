Next steps
\nImprove quality of response with synonyms
06/30/2025
In this tutorial, you learn how to:
This tutorial shows you how you can improve the quality of your responses by using synonyms.
Let's assume that users aren't getting an accurate response to their queries, when they use
alternate forms, synonyms or acronyms of a word. So, they decide to improve the quality of the
response by using Authoring API to add synonyms for keywords.
Let improve the results by adding the following words and their alterations:
Word
Alterations
fix problems
troubleshoot , diagnostic
whiteboard
white board , white canvas
bluetooth
blue tooth , BT
JSON
Add synonyms to improve the quality of your responses
＂
Evaluate the response quality via the inspect option of the Test pane
＂
Add synonyms using Authoring API
ﾉ
Expand table
{
    "synonyms": [
        {
            "alterations": [
                "fix problems",
                "troubleshoot",
                "diagnostic",
                ]
        },
        {
            "alterations": [
                "whiteboard",
                "white board",
                "white canvas"
            ]
        },
        {
            "alterations": [
\nFor the question and answer pair “Fix problems with Surface Pen,” we compare the response
for a query made using its synonym “troubleshoot.”
                "bluetooth",
                "blue tooth",
                "BT"
            ]
        }
    ]
}
Response before addition of synonym

Response after addition of synonym
\n![Image](images/page1173_image1.png)
\nAs you can see, when troubleshoot  was not added as a synonym, we got a low confidence
response to the query “How to troubleshoot your surface pen.” However, after we add
troubleshoot  as a synonym to “fix problems”, we received the correct response to the query
with a higher confidence score. Once these synonyms were added, the relevance of results is
improved.
Synonyms can be added in any order. The ordering is not considered in any
computational logic.
Synonyms can only be added to a project that has at least one question and answer pair.

） Important
Synonyms are case insensitive. Synonyms also might not work as expected if you add stop
words as synonyms. The list of stop words can be found here: List of stop words
. For
instance, if you add the abbreviation IT for Information technology, the system might not
be able to recognize Information Technology because IT is a stop word and is filtered
when a query is processed.
Notes
\n![Image](images/page1174_image1.png)
\nSynonyms can be added only when there is at least one question and answer pair present
in a project.
In case of overlapping synonym words between two sets of alterations, it can have
unexpected results and it isn't recommended to use overlapping sets.
Special characters are not allowed for synonyms. For hyphenated words like "COVID-19,"
they are treated the same as "COVID 19," and "space" can be used as a term separator.
Following is the list of special characters not allowed:
Special character
Symbol
Comma
,
Question mark
?
Colon
:
Semicolon
;
Double quotation mark
"
Single quotation mark
'
Open parenthesis
(
Close parenthesis
)
Open brace
{
Close brace
}
Open bracket
[
Close bracket
]
Hyphen/dash
-
Plus sign
+
Period
.
Forward slash
/
Exclamation mark
!
Asterisk
*
Underscore
_
Ampersand
@
ﾉ
Expand table
\nSpecial character
Symbol
Hash
#
Next steps
\nCreate projects in multiple languages
06/21/2025
In this tutorial, you learn how to:
This tutorial will walk through the process of creating projects in multiple languages. We use
the Surface Pen FAQ
 URL to create projects in German and English. We then deploy the
project and use the custom question answering REST API to query and get answers to FAQs in
the desired language.
To be able to create a project in more than one language, the multiple language setting must
be set at the creation of the first project that is associated with the language resource.
Create a project that supports English
＂
Create a project that supports German
＂
Create project in German

\n![Image](images/page1177_image1.png)
\n1. From the Language Studio
 home page, select open custom question answering. Select
Create new project > I want to select the language when I create a project in this
resource > Next.
2. Fill out enter basic information page and select Next > Create project.
Setting
Value
Name
Unique name for your project
Description
Unique description to help identify the project
Source language
For this tutorial, select German
Default answer
Default answer when no answer is returned
3. Add source > URLs > Add url > Add all.
ﾉ
Expand table

ﾉ
Expand table
\n![Image](images/page1178_image1.png)
\nSetting
Value
Url Name
Surface Pen German
URL
https://support.microsoft.com/de-de/surface/how-to-use-your-surface-pen-
8a403519-cd1f-15b2-c9df-faa5aa924e98
Classify file
structure
Auto-detect
Custom question answering reads the document and extracts question answer pairs from
the source URL to create the project in the German language. If you select the link to the
source, the project page opens where we can edit the contents.
We now repeat the above steps from before but this time select English and provide an English
URL as a source.
1. From the Language Studio
 open the custom question answering page > Create new
project.
2. Fill out enter basic information page and select Next > Create project.
Setting
Value
Name
Unique name for your project

Create project in English
ﾉ
Expand table
\n![Image](images/page1179_image1.png)
\nSetting
Value
Description
Unique description to help identify the project
Source language
For this tutorial, select English
Default answer
Default answer when no answer is returned
3. Add source > URLs > Add url > Add all.
Setting
Value
Url Name
Surface Pen German
URL
https://support.microsoft.com/en-us/surface/how-to-use-your-surface-pen-
8a403519-cd1f-15b2-c9df-faa5aa924e98
Classify file
structure
Auto-detect
We are now ready to deploy the two project and query them in the desired language using the
custom question answering REST API. Once a project is deployed, the following page is shown
which provides details to query the project.
ﾉ
Expand table
Deploy and query project