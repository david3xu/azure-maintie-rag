
\n![Image](images/page1201_image1.png)
\n4. To correctly set the input variable to the QnA Maker action, select Select a variable, then
select bot.UnrecognizedTriggerPhrase.
\n
\n![Image](images/page1203_image1.png)
\n5. To correctly set the output variable to the custom Question Answering action, in the
Message action, select UnrecognizedTriggerPhrase, then select the icon to insert a
variable, {x}, then select FinalAnswer.
6. From the context toolbar, select Save, to save the authoring canvas details for the topic.
Here's what the final bot canvas looks like:
\n
\n![Image](images/page1205_image1.png)
\nAs you design your bot in Power Virtual Agents, you can use the Test bot pane to see how the
bot leads a customer through the bot conversation.
1. In the test pane, toggle Track between topics. This allows you to watch the progression
between topics, as well as within a single topic.
2. Test the bot by entering the user text in the following order. The authoring canvas reports
the successful steps with a green check mark.
Question
order
Test questions
Purpose
1
Hello
Begin conversation
2
Store hours
Sample topic. This is configured for you without any
additional work on your part.
3
Yes
In reply to "Did that answer your question?"
4
Excellent
In reply to "Please rate your experience."
5
Yes
In reply to "Can I help with anything else?"
6
How can I improve the
throughput performance
for query predictions?
This question triggers the fallback action, which sends the
text to your project to answer. Then the answer is shown. the
green check marks for the individual actions indicate success
for each action.
Test the bot
ﾉ
Expand table
\n
Publish your bot
\n![Image](images/page1207_image1.png)
\nTo make the bot available to all members of your organization, you need to publish it.
Publish your bot by following the steps in Publish your bot.
To make your bot available to others, you first need to publish it to a channel. For this tutorial
we'll use the demo website.
Configure the demo website by following the steps in Configure a chatbot for a live or demo
website.
Then you can share your website URL with your school or organization members.
When you are done with the project, remove the QnA Maker resources in the Azure portal.
Tutorial: Create an FAQ bot
Share your bot
Clean up resources
See also
\nMarkdown format supported in answer
text
06/21/2025
Custom question answering stores answer text as markdown. There are many flavors of
markdown. In order to make sure the answer text is returned and displayed correctly, use this
reference.
Use the CommonMark
 tutorial to validate your markdown. The tutorial has a Try it feature
for quick copy/paste validation.
Rich-text editing of answers allows you, as the author, to use a formatting toolbar to quickly
select and format text.
Markdown is a better tool when you need to autogenerate content to create projects to be
imported as part of a CI/CD pipeline or for batch testing.
Following is the list of markdown formats that you can use in your answer text.
Purpose
Format
Example markdown
A new line
between 2
sentences.
\n\n
How can I create a bot with \n\n custom
question answering?
Headers from h1
to h6, the
number of #
denotes which
header. 1 #  is
the h1.
\n# text \n## text \n### text
\n####text \n#####text
## Creating a bot \n ...text.... \n###
Important news\n ...text... \n### Related
Information\n ....text...
\n# my h1 \n## my h2\n### my h3 \n#### my
h4 \n##### my h5
Italics
*text*
How do I create a bot with *custom question
answering*?
Strong (bold)
**text**
How do I create a bot with **custom
question answering***?
When to use rich-text editing versus markdown
Supported markdown format
ﾉ
Expand table
\nPurpose
Format
Example markdown
URL for link
[text](https://www.my.com)
How do I create a bot with [custom question
answering]
(https://language.cognitive.azure.com/)?
*URL for public
image
![text]
(https://www.my.com/image.png)
How can I create a bot with ![custom
question answering](path-to-your-image.png)
Strikethrough
~~text~~
some ~~questions~~ questions need to be
asked
Bold and italics
***text***
How can I create a ***custom question
answering**** bot?
Bold URL for link
[**text**](https://www.my.com)
How do I create a bot with [**custom
question answering**]
(https://language.cognitive.azure.com/)?
Italics URL for
link
[*text*](https://www.my.com)
How do I create a bot with [*custom
question answering*]
(https://language.cognitive.azure.com/)?
Escape
markdown
symbols
\*text\*
How do I create a bot with \*custom
question answering*\*?
Ordered list
\n 1. item1 \n 1. item2
This is an ordered list: \n 1. List item 1
\n 1. List item 2
The preceding example uses automatic
numbering built into markdown.
This is an ordered list: \n 1. List item 1
\n 2. List item 2
The preceding example uses explicit
numbering.
Unordered list
\n * item1 \n * item2
or
\n - item1 \n - item2
This is an unordered list: \n * List item 1
\n * List item 2
Nested lists
\n * Parent1 \n\t * Child1 \n\t *
Child2 \n * Parent2
\n * Parent1 \n\t 1. Child1 \n\t *
Child2 \n 1. Parent2
You can nest ordered and unordered
lists together. The tab, \t , indicates
the indentation level of the child
element.
This is an unordered list: \n * List item 1
\n\t * Child1 \n\t * Child2 \n * List item
2
This is an ordered nested list: \n 1.
Parent1 \n\t 1. Child1 \n\t 1. Child2 \n 1.
Parent2