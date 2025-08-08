Add the Excel file to the flow by filling in the fields in this action. This tutorial requires
the file to have been uploaded to OneDrive for Business.

\n![Image](images/page621_image1.png)
\nSelect New Step and add an Apply to each action.

\n![Image](images/page622_image1.png)
\nSelect Select an output from previous step. In the Dynamic content box that appears,
select value.
If you haven't already, you need to create a Language resource
 in the Azure portal.


Send a request for entity recognition
\n![Image](images/page623_image1.png)

![Image](images/page623_image2.png)
\nIn the Apply to each, select Add an action. Go to your Language resource's key and
endpoint page in the Azure portal, and get the key and endpoint for your Language
resource.
In your flow, enter the following information to create a new Language connection.
Field
Value
Connection
Name
A name for the connection to your Language resource. For example,
TAforPowerAutomate .
Account key
The key for your Language resource.
Site URL
The endpoint for your Language resource.
Create a Language service connection
７ Note
If you already have created a Language connection and want to change your
connection details, Select the ellipsis on the top right corner, and select + Add new
connection.
ﾉ
Expand table
\nAfter the connection is created, search for Text Analytics and select Named Entity
Recognition. This extracts information from the description column of the issue.

Extract the excel content
\n![Image](images/page625_image1.png)
\nSelect in the Text field and select Description from the Dynamic content windows that
appears. Enter en  for Language, and a unique name as the document ID (you might
need to select Show advanced options).

\n![Image](images/page626_image1.png)
\nWithin the Apply to each, select Add an action and create another Apply to each
action. Select inside the text box and select documents in the Dynamic Content window
that appears.


\n![Image](images/page627_image1.png)

![Image](images/page627_image2.png)
\nNext, we find the person entity type in the NER output. Within the Apply to each 2,
select Add an action, and create another Apply to each action. Select inside the text box
and select Entities in the Dynamic Content window that appears.
Within the newly created Apply to each 3 action, select Add an action, and add a
Condition control.
Extract the person name

\n![Image](images/page628_image1.png)
\nIn the Condition window, select the first text box. In the Dynamic content window,
search for Category and select it.
Make sure the second box is set to is equal to. Then select the third box, and search for
var_person  in the Dynamic content window.


\n![Image](images/page629_image1.png)

![Image](images/page629_image2.png)
\nIn the If yes condition, type in Excel then select Update a Row.
Enter the Excel information, and update the Key Column, Key Value and PersonName
fields. This appends the name detected by the API to the Excel sheet.


\n![Image](images/page630_image1.png)

![Image](images/page630_image2.png)