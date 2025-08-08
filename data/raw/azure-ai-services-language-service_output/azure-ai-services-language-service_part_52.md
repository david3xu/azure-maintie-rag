Security should be a primary focus whenever you're developing applications. The
importance of security is a metric for success. When you're architecting a software
solution that includes Azure AI containers, it's vital to understand the limitations and
capabilities available to you. For more information about network security, see
Configure Azure AI services virtual networks.
The following diagram illustrates the default and non-secure approach:
As an example of an alternative and secure approach, consumers of Azure AI containers
could augment a container with a front-facing component, keeping the container
endpoint private. Let's consider a scenario where we use Istio
 as an ingress gateway.
Istio supports HTTPS/TLS and client-certificate authentication. In this scenario, the Istio
frontend exposes the container access, presenting the client certificate that is approved
beforehand with Istio.
Nginx
 is another popular choice in the same category. Both Istio and Nginx act as a
service mesh and offer additional features including things like load-balancing, routing,
and rate-control.
） Important
By default there is no security on the Azure AI services container API. The reason for
this is that most often the container will run as part of a pod which is protected
from the outside by a network bridge. However, it is possible for users to construct
their own authentication infrastructure to approximate the authentication methods
used when accessing the cloud-based Azure AI services.
\nFeedback
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
\nTutorial: Extract key phrases from text
stored in Power BI
06/04/2025
Microsoft Power BI Desktop is a free application that lets you connect to, transform, and
visualize your data. Key phrase extraction, one of the features of Azure AI Language, provides
natural language processing. Given raw unstructured text, it can extract the most important
phrases, analyze sentiment, and identify well-known entities such as brands. Together, these
tools can help you quickly see what your customers are talking about and how they feel about
it.
In this tutorial, you'll learn how to:
Microsoft Power BI Desktop. Download at no charge
.
A Microsoft Azure account. Create a free account
 or sign in
.
A Language resource. If you don't have one, you can create one
.
The Language resource key that was generated for you when you created the resource.
Customer comments. You can use our example data
 or your own data. This tutorial
assumes you're using our example data.
To get started, open Power BI Desktop and load the comma-separated value (CSV) file that you
downloaded as part of the prerequisites. This file represents a day's worth of hypothetical
activity in a fictional small company's support forum.
Use Power BI Desktop to import and transform data
＂
Create a custom function in Power BI Desktop
＂
Integrate Power BI Desktop with the Key Phrase Extraction feature of Azure AI Language
＂
Use Key Phrase Extraction to get the most important phrases from customer feedback
＂
Create a word cloud from customer feedback
＂
Prerequisites
Load customer data
７ Note
Power BI can use data from a wide variety of web-based sources, such as SQL databases.
See the Power Query documentation for more information.
\nIn the main Power BI Desktop window, select the Home ribbon. In the External data group of
the ribbon, open the Get Data drop-down menu and select Text/CSV.
The Open dialog appears. Navigate to your Downloads folder, or to the folder where you
downloaded the CSV file. Select the name of the file, then the Open button. The CSV import
dialog appears.
The CSV import dialog lets you verify that Power BI Desktop has correctly detected the
character set, delimiter, header rows, and column types. This information is all correct, so select
Load.
To see the loaded data, click the Data View button on the left edge of the Power BI workspace.
A table opens that contains the data, like in Microsoft Excel.
\n![Image](images/page515_image1.png)

![Image](images/page515_image2.png)
\nYou might need to transform your data in Power BI Desktop before it's ready to be processed
by Key Phrase Extraction.
The sample data contains a subject  column and a comment  column. With the Merge Columns
function in Power BI Desktop, you can extract key phrases from the data in both these columns,
rather than just the comment  column.
In Power BI Desktop, select the Home ribbon. In the External data group, select Edit Queries.
Select FabrikamComments  in the Queries list at the left side of the window if it isn't already
selected.
Now select both the subject  and comment  columns in the table. You might need to scroll
horizontally to see these columns. First click the subject  column header, then hold down the
Control key and click the comment  column header.
Prepare the data
\n![Image](images/page516_image1.png)

![Image](images/page516_image2.png)
\nSelect the Transform ribbon. In the Text Columns group of the ribbon, select Merge Columns.
The Merge Columns dialog appears.
In the Merge Columns dialog, choose Tab  as the separator, then select OK.
You might also consider filtering out blank messages using the Remove Empty filter, or
removing unprintable characters using the Clean transformation. If your data contains a
column like the spamscore  column in the sample file, you can skip "spam" comments using a
Number Filter.
Key Phrase Extraction
 can process up to a thousand text documents per HTTP request. Power
BI prefers to deal with records one at a time, so in this tutorial your calls to the API will include
Understand the API
\n![Image](images/page517_image1.png)

![Image](images/page517_image2.png)
\nonly a single document each. The Key Phrases API requires the following fields for each
document being processed.
Field
Description
id
A unique identifier for this document within the request. The response also contains this field.
That way, if you process more than one document, you can easily associate the extracted key
phrases with the document they came from. In this tutorial, because you're processing only
one document per request, you can hard-code the value of id  to be the same for each
request.
text
The text to be processed. The value of this field comes from the Merged  column you created
in the previous section, which contains the combined subject line and comment text. The Key
Phrases API requires this data be no longer than about 5,120 characters.
language
The code for the natural language the document is written in. All the messages in the sample
data are in English, so you can hard-code the value en  for this field.
Now you're ready to create the custom function that will integrate Power BI and Key Phrase
Extraction. The function receives the text to be processed as a parameter. It converts data to
and from the required JSON format and makes the HTTP request to the Key Phrases API. The
function then parses the response from the API and returns a string that contains a comma-
separated list of the extracted key phrases.
In Power BI Desktop, make sure you're still in the Query Editor window. If you aren't, select the
Home ribbon, and in the External data group, select Edit Queries.
Now, in the Home ribbon, in the New Query group, open the New Source drop-down menu
and select Blank Query.
A new query, initially named Query1 , appears in the Queries list. Double-click this entry and
name it KeyPhrases .
ﾉ
Expand table
Create a custom function
７ Note
Power BI Desktop custom functions are written in the Power Query M formula language,
or just "M" for short. M is a functional programming language based on F#. You don't
need to be a programmer to finish this tutorial, though; the required code is included
below.
\nNow, in the Home ribbon, in the Query group, select Advanced Editor to open the Advanced
Editor window. Delete the code that's already in that window and paste in the following code.
F#
Replace YOUR_API_KEY_HERE  with your Language resource key. You can also find this key by
signing in to the Azure portal
, navigating to your Language resource, and selecting the Key
and endpoint page. Be sure to leave the quotation marks before and after the key. Then select
Done.
Now you can use the custom function to extract the key phrases from each of the customer
comments and store them in a new column in the table.
In Power BI Desktop, in the Query Editor window, switch back to the FabrikamComments  query.
Select the Add Column ribbon. In the General group, select Invoke Custom Function.
７ Note
Replace the example endpoint below (containing <your-custom-subdomain> ) with the
endpoint generated for your Language resource. You can find this endpoint by signing in
to the Azure portal
, navigating to your resource, and selecting Key and endpoint.
// Returns key phrases from the text in a comma-separated list
(text) => let
    apikey      = "YOUR_API_KEY_HERE",
    endpoint    = "https://<your-custom-
subdomain>.cognitiveservices.azure.com/text/analytics" & "/v3.0/keyPhrases",
    jsontext    = Text.FromBinary(Json.FromValue(Text.Start(Text.Trim(text), 
5000))),
    jsonbody    = "{ documents: [ { language: ""en"", id: ""0"", text: " & 
jsontext & " } ] }",
    bytesbody   = Text.ToBinary(jsonbody),
    headers     = [#"Ocp-Apim-Subscription-Key" = apikey],
    bytesresp   = Web.Contents(endpoint, [Headers=headers, Content=bytesbody]),
    jsonresp    = Json.Document(bytesresp),
    keyphrases  = Text.Lower(Text.Combine(jsonresp[documents]{0}[keyPhrases], ", 
"))
in  keyphrases
Use the custom function
\nThe Invoke Custom Function dialog appears. In New column name, enter keyphrases . In
Function query, select the custom function you created, KeyPhrases .
A new field appears in the dialog, text (optional). This field is asking which column we want to
use to provide values for the text  parameter of the Key Phrases API. (Remember that you
already hard-coded the values for the language  and id  parameters.) Select Merged  (the
column you created previously by merging the subject and message fields) from the drop-
down menu.
Finally, select OK.
If everything is ready, Power BI calls your custom function once for each row in the table. It
sends the queries to the Key Phrases API and adds a new column to the table to store the
results. But before that happens, you might need to specify authentication and privacy settings.
Authentication and privacy
\n![Image](images/page520_image1.png)

![Image](images/page520_image2.png)