Tutorial: Create an FAQ bot
Next steps
\nWhat is sentiment analysis and opinion
mining?
Article • 02/21/2025
Sentiment analysis and opinion mining are features offered by the Language service, a
collection of machine learning and AI algorithms in the cloud for developing intelligent
applications that involve written language. These features help you find out what people
think of your brand or topic by mining text for clues about positive or negative
sentiment, and can associate them with specific aspects of the text.
Both sentiment analysis and opinion mining work with various written languages.
The sentiment analysis feature provides sentiment labels (such as "negative", "neutral"
and "positive") based on the highest confidence score found by the service at a
sentence and document-level. This feature also returns confidence scores between 0
and 1 for each document & sentences within it for positive, neutral, and negative
sentiment.
Opinion mining is a feature of sentiment analysis, also known as aspect-based sentiment
analysis in Natural Language Processing (NLP). This feature provides more granular
information about the opinions related to words (such as the attributes of products or
services) in text.
To use this feature, you submit data for analysis and handle the API output in your
application. Analysis is performed as-is, with no added customization to the model used
on your data.
1. Create an Azure AI Language resource, which grants you access to the features
offered by Azure AI Language. It generates a password (called a key) and an
endpoint URL that you use to authenticate API requests.
2. Create a request using either the REST API or the client library for C#, Java,
JavaScript, and Python. You can also send asynchronous calls with a batch request
Sentiment analysis
Opinion mining
Typical workflow
\nto combine API requests for multiple features into a single call.
3. Send the request containing your text data. Your key and endpoint are used for
authentication.
4. Stream or store the response locally.
To use sentiment analysis, you submit raw unstructured text for analysis and handle the
API output in your application. Analysis is performed as-is, with no additional
customization to the model used on your data. There are two ways to use sentiment
analysis:
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
Integrate sentiment analysis into your applications using the REST API, or
the client library available in a variety of languages. For more information,
see the sentiment analysis quickstart.
Docker container
Use the available Docker container to deploy this feature on-premises.
These docker containers enable you to bring the service closer to your
data for compliance, security, or other operational reasons.
As you use this feature in your applications, see the following reference documentation
and samples for Azure AI Language:
Development option / language
Reference documentation
Samples
REST API
REST API documentation
C#
C# documentation
C# samples
Get started with sentiment analysis
ﾉ
Expand table
Reference documentation and code samples
ﾉ
Expand table
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Development option / language
Reference documentation
Samples
Java
Java documentation
Java Samples
JavaScript
JavaScript documentation
JavaScript samples
Python
Python documentation
Python samples
As you use sentiment analysis, see the following reference documentation and samples
for the Language service:
Development option / language
Reference documentation
Samples
REST APIs (Authoring)
REST API documentation
REST APIs (Runtime)
REST API documentation
An AI system includes not only the technology, but also the people who use it, the
people who are affected by it, and the environment in which it's deployed. Read the
transparency note for sentiment analysis to learn about responsible AI use and
deployment in your systems. You can also see the following articles for more
information:
The quickstart articles with instructions on using the service for the first time.
Use sentiment analysis and opinion mining
Reference documentation
ﾉ
Expand table
Responsible AI
Next steps
Yes
No
\nQuickstart: Sentiment analysis and
opinion mining
Article • 02/17/2025
Create a Project in Foundry in the Azure AI Foundry Portal
Using the left side pane, select Playgrounds. Then select the Try the Language
Playground button.
Prerequisites
Navigate to the Azure AI Foundry Playground

\n![Image](images/page1225_image1.png)
\nThe Language Playground consists of four sections:
Top banner: You can select any of the currently available Language services here.
Right pane: This pane is where you can find the Configuration options for the
service, such as the API and model version, along with features specific to the
service.
Center pane: This pane is where you enter your text for processing. After the
operation is run, some results are shown here.
Right pane: This pane is where Details of the run operation are shown.
Here you can select the Sentiment Analysis capability by choosing the top banner tile,
Analyze sentiment.
Analyze sentiment is designed to identify positive, negative and neutral sentiment in
text.
In Configuration there are the following options:
Option
Description
Select API version
Select which version of the API to use.
Select model version
Select which version of the model to use.
Select text language
Select the language of the input text.
Enable opinion mining
Enables or disables the opinion mining skill.
After your operation is completed, in the center pane, each sentence will be numbered
and opinions will be labeled if Enable opinion mining was checked and the Details
section contains the following fields for the overall sentiment and the sentiment of each
sentence:
Use Sentiment Analysis in the Azure AI Foundry
Playground
Use Analyze sentiment
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
Sentence
number
The number of the sentence in the order it was typed. This field is not present
for Overall sentiment.
Sentiment
The detected overall sentiment for the segment of text.
Scores
The amount of positive, neutral and negative sentiment detected in the text
segment.
The following fields are only present if opinion mining is enabled:
Field
Description
Target
The target of the detected opinion.
Assessments
The detected opinion and the detected persuasion (positive, neutral, negative), as
well as the percent of detected persuasion.
ﾉ
Expand table

Yes
No
\n![Image](images/page1227_image1.png)
\nSentiment Analysis and Opinion Mining
language support
06/21/2025
Use this article to learn which languages are supported by Sentiment Analysis and Opinion
Mining. Both the cloud-based API and Docker containers support the same languages.
Total supported language codes: 94
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
Belarusian (new)
be
Bengali
bn
Bosnian
bs
Breton (new)
br
Bulgarian
bg
Burmese
my
Catalan
ca
Chinese (Simplified)
zh-hans
zh  also accepted
Sentiment Analysis language support
ﾉ
Expand table
\nLanguage
Language code
Notes
Chinese (Traditional)
zh-hant
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
Esperanto (new)
eo
Estonian
et
Filipino
fil
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
Hausa (new)
ha
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
Javanese (new)
jv
Kannada
kn
\nLanguage
Language code
Notes
Kazakh
kk
Khmer
km
Korean
ko
Kurdish (Kurmanji)
ku
Kyrgyz
ky
Lao
lo
Latin (new)
la
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
Norwegian
no
Odia
or
Oromo (new)
om
Pashto
ps
Persian
fa
Polish
pl
Portuguese (Portugal)
pt-PT
pt  also accepted
Portuguese (Brazil)
pt-BR
Punjabi
pa
Romanian
ro