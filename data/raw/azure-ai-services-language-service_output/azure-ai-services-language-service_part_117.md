If you're not going to continue to use this application, delete the associate custom question
answering and bot service resources.
Advance to the next article to learn how to customize your FAQ bot with multi-turn prompts.
Clean up resources
Next steps
\n![Image](images/page1161_image1.png)
\nAdd guided conversations with multi-turn
prompts
06/05/2025
In this tutorial, you learn how to:
In this tutorial, we use Surface Pen FAQ
 to create a project.
If you have never created a custom question answering project before we recommend starting
with the getting started article, which will take you step-by-step through the process.
For this example, let's assume that users are asking for additional details about the Surface Pen
product, particularly how to troubleshoot their Surface Pen, but they are not getting the correct
answers. So, we add more prompts to support additional scenarios and guide the users to the
correct answers using multi-turn prompts.
Multi-turn prompts that are associated with question and answer pairs, can be viewed by
selecting Show columns > Context. By default this should already be enabled on the Edit
project page in the Language Studio custom question answering interface.
This displays the context tree where all follow-up prompts linked to a QnA pair are shown:
Add new question and answer pairs to your existing project
＂
Add follow-up prompts to create guided conversations
＂
Test your multi-turn prompts
＂
Prerequisites
View question answer pair context

\n![Image](images/page1162_image1.png)
\nTo help users solve issues with their Surface Pen, we add follow-up prompts:
Add a new question pair with two follow-up prompts
Add a follow-up prompt to one of the newly added prompts
1. Add a new QnA pair with two follow-up prompts Check compatibility and Check Pen
Settings Using the editor, we add a new QnA pair with a follow-up prompt by clicking on
Add question pair
A new row in Editorial is created where we enter the question answer pair as shown
below:
Field
Value
Questions
Fix problems with Surface
Answers and
prompts
Here are some things to try first if your Surface Pen won't write, open apps, or
connect to Bluetooth.
2. We then add a follow-up prompt to the newly created question pair by choosing Add
follow-up prompts. Fill the details for the prompt as shown:

Add question pair with follow-up prompts
ﾉ
Expand table
\n![Image](images/page1163_image1.png)

![Image](images/page1163_image2.png)
\nWe provide Check Compatibility as the "Display text" for the prompt and try to link it to a
QnA. Since, no related QnA pair is available to link to the prompt, when we search “Check
your Surface Pen Compatibility”, we create a new question pair by clicking on Create link
to new pair and select Done. Then select Save changes.
3. Similarly, we add another prompt Check Pen Settings to help the user troubleshoot the
Surface Pen and add question pair to it.


\n![Image](images/page1164_image1.png)

![Image](images/page1164_image2.png)
\n4. Add another follow-up prompt to the newly created prompt. We now add “Replace Pen
tips’ as a follow-up prompt to the previously created prompt “Check Pen Settings”.


\n![Image](images/page1165_image1.png)

![Image](images/page1165_image2.png)
\n5. Finally, save the changes and test these prompts in the Test pane:
For a user query Issues with Surface Pen, the system returns an answer and presents the
newly added prompts to the user. The user then selects one of the prompts Check Pen
Settings and the related answer is returned to the user with another prompt Replace Pen
Tips, which when selected further provides the user with more information. So, multi-turn
is used to help and guide the user to the desired answer.


\n![Image](images/page1166_image1.png)

![Image](images/page1166_image2.png)
\nNext steps
\nEnrich your project with active learning
06/04/2025
In this tutorial, you learn how to:
This tutorial shows you how to enhance your Custom question answering project with active
learning. If you notice that customers are asking questions that are not covered in your project,
they may be paraphrased variations of questions.
These variations, when added as alternate questions to the relevant question answer pair, help
to optimize the project to answer real world user queries. You can manually add alternate
questions to question answer pairs through the editor. At the same time, you can also use the
active learning feature to generate active learning suggestions based on user queries. The
active learning feature, however, requires that the project receives regular user traffic to
generate suggestions.
Active learning is turned on by default for Custom question answering enabled resources.
To try out active learning suggestions, you can import the following file as a new project:
SampleActiveLearning.tsv
.
Run the following command from the command prompt to download a local copy of the
SampleActiveLearning.tsv  file.
Windows Command Prompt
Download an active learning test file
＂
Import the test file to your existing project
＂
Accept/reject active learning suggestions
＂
Add alternate questions
＂
Use active learning
Download file
curl "https://github.com/Azure-Samples/cognitive-services-sample-data-
files/blob/master/qna-maker/knowledge-bases/SampleActiveLearning.tsv" --output 
SampleActiveLearning.tsv
Import file
\nFrom the edit project pane for your project, select the ...  (ellipsis) icon from the menu >
Import questions and answers > Import as TSV. Then, select Choose file to browse to the
copy of SampleActiveLearning.tsv  that you downloaded to your computer in the previous
step, and then select done.
Once the import of the test file is complete, active learning suggestions can be viewed on the
review suggestions pane:

View and add/reject active learning suggestions

７ Note
\n![Image](images/page1169_image1.png)

![Image](images/page1169_image2.png)
\nWe can now either accept these suggestions or reject them using the options on the menu bar
to Accept all suggestions or Reject all suggestions.
Alternatively, to accept or reject individual suggestions, select the checkmark (accept) symbol
or trash can (reject) symbol that appears next to individual questions in the Review
suggestions page.
While active learning automatically suggests alternate questions based on the user queries
hitting the project, we can also add variations of a question on the edit project page by
selecting Add alternate phrase to question answer pairs.
By adding alternate questions along with active learning, we further enrich the project with
variations of a question that helps to provide consistent answers to user queries.
Active learning suggestions are not real time. There is an approximate delay of 30 minutes
before the suggestions can show on this pane. This delay is to ensure that we balance the
high cost involved for real time updates to the index and service performance.

Add alternate questions
７ Note
When alternate questions have many stop words, they might negatively impact the
accuracy of responses. So, if the only difference between alternate questions is in the stop
words, these alternate questions are not required. To examine the list of stop words
consult the stop words article
.
\n![Image](images/page1170_image1.png)