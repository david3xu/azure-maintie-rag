After you close the Invoke Custom Function dialog, a banner may appear asking you to specify
how to connect to the Key Phrases API.
Select Edit Credentials, make sure Anonymous  is selected in the dialog, then select Connect.
If you see the Edit Credentials banner even after choosing anonymous access, you might have
forgotten to paste your Language resource key into the code in the KeyPhrases  custom
function.
Next, a banner may appear asking you to provide information about your data sources' privacy.
Select Continue and choose Public  for each of the data sources in the dialog. Then select
Save.
７ Note
You select Anonymous  because Key Phrase Extraction authenticates requests using your
access key, so Power BI does not need to provide credentials for the HTTP request itself.
\n![Image](images/page521_image1.png)

![Image](images/page521_image2.png)

![Image](images/page521_image3.png)
\nOnce you have dealt with any banners that appear, select Close & Apply in the Home ribbon to
close the Query Editor.
Power BI Desktop takes a moment to make the necessary HTTP requests. For each row in the
table, the new keyphrases  column contains the key phrases detected in the text by the Key
Phrases API.
Now you'll use this column to generate a word cloud. To get started, click the Report button in
the main Power BI Desktop window, to the left of the workspace.
If you don't already have the Word Cloud custom visual installed, install it. In the Visualizations
panel to the right of the workspace, click the three dots (...) and choose Import From Market. If
the word "cloud" is not among the displayed visualization tools in the list, you can search for
"cloud" and click the Add button next the Word Cloud visual. Power BI installs the Word Cloud
visual and lets you know that it installed successfully.
Create the word cloud
７ Note
Why use extracted key phrases to generate a word cloud, rather than the full text of every
comment? The key phrases provide us with the important words from our customer
comments, not just the most common words. Also, word sizing in the resulting cloud isn't
skewed by the frequent use of a word in a relatively small number of comments.
\n![Image](images/page522_image1.png)
\nFirst, click the Word Cloud icon in the Visualizations panel.
A new report appears in the workspace. Drag the keyphrases  field from the Fields panel to the
Category field in the Visualizations panel. The word cloud appears inside the report.
Now switch to the Format page of the Visualizations panel. In the Stop Words category, turn on
Default Stop Words to eliminate short, common words like "of" from the cloud. However,
because we're visualizing key phrases, they might not contain stop words.
\n![Image](images/page523_image1.png)

![Image](images/page523_image2.png)
\nDown a little further in this panel, turn off Rotate Text and Title.
Select the Focus Mode tool in the report to get a better look at our word cloud. The tool
expands the word cloud to fill the entire workspace, as shown below.
Using other features
\n![Image](images/page524_image1.png)

![Image](images/page524_image2.png)

![Image](images/page524_image3.png)
\nAzure AI Language also provides sentiment analysis and language detection. The language
detection in particular is useful if your customer feedback isn't all in English.
Both of these other APIs are similar to the Key Phrases API. That means you can integrate them
with Power BI Desktop using custom functions that are nearly identical to the one you created
in this tutorial. Just create a blank query and paste the appropriate code below into the
Advanced Editor, as you did earlier. (Don't forget your access key!) Then, as before, use the
function to add a new column to the table.
The Sentiment Analysis function below returns a label indicating how positive the sentiment
expressed in the text is.
F#
Here are two versions of a Language Detection function. The first returns the ISO language
code (for example, en  for English), while the second returns the "friendly" name (for example,
English ). You may notice that only the last line of the body differs between the two versions.
F#
// Returns the sentiment label of the text, for example, positive, negative or 
mixed.
(text) => let
    apikey = "YOUR_API_KEY_HERE",
    endpoint = "<your-custom-subdomain>.cognitiveservices.azure.com" & 
"/text/analytics/v3.1/sentiment",
    jsontext = Text.FromBinary(Json.FromValue(Text.Start(Text.Trim(text), 5000))),
    jsonbody = "{ documents: [ { language: ""en"", id: ""0"", text: " & jsontext & 
" } ] }",
    bytesbody = Text.ToBinary(jsonbody),
    headers = [#"Ocp-Apim-Subscription-Key" = apikey],
    bytesresp = Web.Contents(endpoint, [Headers=headers, Content=bytesbody]),
    jsonresp = Json.Document(bytesresp),
    sentiment   = jsonresp[documents]{0}[sentiment] 
    in sentiment
// Returns the two-letter language code (for example, 'en' for English) of the 
text
(text) => let
    apikey      = "YOUR_API_KEY_HERE",
    endpoint    = "https://<your-custom-subdomain>.cognitiveservices.azure.com" & 
"/text/analytics/v3.1/languages",
    jsontext    = Text.FromBinary(Json.FromValue(Text.Start(Text.Trim(text), 
5000))),
    jsonbody    = "{ documents: [ { id: ""0"", text: " & jsontext & " } ] }",
    bytesbody   = Text.ToBinary(jsonbody),
    headers     = [#"Ocp-Apim-Subscription-Key" = apikey],
    bytesresp   = Web.Contents(endpoint, [Headers=headers, Content=bytesbody]),
\nF#
Finally, here's a variant of the Key Phrases function already presented that returns the phrases
as a list object, rather than as a single string of comma-separated phrases.
F#
    jsonresp    = Json.Document(bytesresp),
    language    = jsonresp [documents]{0}[detectedLanguage] [name] in language 
// Returns the name (for example, 'English') of the language in which the text is 
written
(text) => let
    apikey      = "YOUR_API_KEY_HERE",
    endpoint    = "https://<your-custom-subdomain>.cognitiveservices.azure.com" & 
"/text/analytics/v3.1/languages",
    jsontext    = Text.FromBinary(Json.FromValue(Text.Start(Text.Trim(text), 
5000))),
    jsonbody    = "{ documents: [ { id: ""0"", text: " & jsontext & " } ] }",
    bytesbody   = Text.ToBinary(jsonbody),
    headers     = [#"Ocp-Apim-Subscription-Key" = apikey],
    bytesresp   = Web.Contents(endpoint, [Headers=headers, Content=bytesbody]),
    jsonresp    = Json.Document(bytesresp),
    language    =jsonresp [documents]{0}[detectedLanguage] [name] in language 
７ Note
Returning a single string simplified our word cloud example. A list, on the other hand, is a
more flexible format for working with the returned phrases in Power BI. You can
manipulate list objects in Power BI Desktop using the Structured Column group in the
Query Editor's Transform ribbon.
// Returns key phrases from the text as a list object
(text) => let
    apikey      = "YOUR_API_KEY_HERE",
    endpoint    = "https://<your-custom-subdomain>.cognitiveservices.azure.com" & 
"/text/analytics/v3.1/keyPhrases",
    jsontext    = Text.FromBinary(Json.FromValue(Text.Start(Text.Trim(text), 
5000))),
    jsonbody    = "{ documents: [ { language: ""en"", id: ""0"", text: " & 
jsontext & " } ] }",
    bytesbody   = Text.ToBinary(jsonbody),
    headers     = [#"Ocp-Apim-Subscription-Key" = apikey],
    bytesresp   = Web.Contents(endpoint, [Headers=headers, Content=bytesbody]),
    jsonresp    = Json.Document(bytesresp),
    keyphrases  = jsonresp[documents]{0}[keyPhrases]
in  keyphrases
\nLearn more about Azure AI Language, the Power Query M formula language, or Power BI.
Azure AI Language overview
Power Query M reference
Power BI documentation
Next steps
\nWhat is Named Entity Recognition
(NER) in Azure AI Language?
Article • 02/21/2025
Named Entity Recognition (NER) is one of the features offered by Azure AI Language, a
collection of machine learning and AI algorithms in the cloud for developing intelligent
applications that involve written language. The NER feature can identify and categorize
entities in unstructured text. For example: people, places, organizations, and quantities.
The prebuilt NER feature has a preset list of recognized entities. The custom NER feature
allows you to train the model to recognize specialized entities specific to your use case.
Quickstarts are getting-started instructions to guide you through making requests
to the service.
How-to guides contain instructions for using the service in more specific or
customized ways.
The conceptual articles provide in-depth explanations of the service's functionality
and features.
To use this feature, you submit data for analysis and handle the API output in your
application. Analysis is performed as-is, with no added customization to the model used
on your data.
1. Create an Azure AI Language resource, which grants you access to the features
offered by Azure AI Language. It generates a password (called a key) and an
endpoint URL that you use to authenticate API requests.
2. Create a request using either the REST API or the client library for C#, Java,
JavaScript, and Python. You can also send asynchronous calls with a batch request
to combine API requests for multiple features into a single call.
７ Note
Entity Resolution was upgraded to the Entity Metadata starting in API version
2023-04-15-preview. If you're calling the preview version of the API equal or newer
than 2023-04-15-preview, check out the Entity Metadata article to use the
resolution feature.
Typical workflow
\n3. Send the request containing your text data. Your key and endpoint are used for
authentication.
4. Stream or store the response locally.
To use named entity recognition, you submit raw unstructured text for analysis and
handle the API output in your application. Analysis is performed as-is, with no additional
customization to the model used on your data. There are two ways to use named entity
recognition:
Development
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use named entity
recognition with text examples with your own data when you sign up. For
more information, see the Azure AI Foundry website
 or Azure AI Foundry
documentation.
REST API or Client
library (Azure SDK)
Integrate named entity recognition into your applications using the REST
API, or the client library available in a variety of languages. For more
information, see the named entity recognition quickstart.
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
Java
Java documentation
Java Samples
JavaScript
JavaScript documentation
JavaScript samples
Python
Python documentation
Python samples
Get started with named entity recognition
ﾉ
Expand table
Reference documentation and code samples
ﾉ
Expand table
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
An AI system includes not only the technology, but also the people who use it, the
people who are affected by it, and the environment in which it's deployed. Read the
transparency note for NER to learn about responsible AI use and deployment in your
systems. You can also see the following articles for more information:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
Enhance search capabilities and search indexing - Customers can build knowledge
graphs based on entities detected in documents to enhance document search as
tags.
Automate business processes - For example, when reviewing insurance claims,
recognized entities like name and location could be highlighted to facilitate the
review. Or a support ticket could be generated with a customer's name and
company automatically from an email.
Customer analysis – Determine the most popular information conveyed by
customers in reviews, emails, and calls to determine the most relevant topics that
get brought up and determine trends over time.
There are two ways to get started using the Named Entity Recognition (NER) feature:
Azure AI Foundry is a web-based platform that lets you use several Language
service features without needing to write code.
The quickstart article for instructions on making requests to the service using the
REST API and client library SDK.
Responsible AI
Scenarios
Next steps
Yes
No