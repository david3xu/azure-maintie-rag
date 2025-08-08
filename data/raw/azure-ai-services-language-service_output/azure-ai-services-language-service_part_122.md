Custom question answering doesn't process the image in any way. It is the client
application's role to render the image.
If you want to add content using update/replace project APIs and the content/file contains
html tags, you can preserve the HTML in your file by ensuring that opening and closing of the
tags are converted in the encoded format.
Preserve HTML
Representation in the API request
Representation in KB
Yes
&lt;br&gt;
<br>
Yes
&lt;h3&gt;header&lt;/h3&gt;
<h3>header</h3>
Additionally, CR LF(\r\n)  are converted to \n  in the KB. LF(\n)  is kept as is. If you want to
escape any escape sequence like a \t or \n you can use backslash, for example: '\\r\\n' and '\\t'
Import a project
ﾉ
Expand table
Next steps
\nFormat guidelines for custom question
answering
06/21/2025
Review these formatting guidelines to get the best results for your content.
After importing a file or URL, custom question answering converts and stores your content in
the markdown format
. The conversion process adds new lines in the text, such as \n\n . A
knowledge of the markdown format helps you to understand the converted content and
manage your project content.
If you add or edit your content directly in your project, use markdown formatting to create
rich text content or change the markdown format content that is already in the answer. Custom
question answering supports much of the markdown format to bring rich text capabilities to
your content. However, the client application, such as a chat bot may not support the same set
of markdown formats. It is important to test the client application's display of answers.
Custom question answering identifies sections and subsections and relationships in the file
based on visual clues like:
font size
font style
numbering
colors
A manual is typically guidance material that accompanies a product. It helps the user to set up,
use, maintain, and troubleshoot the product. When custom question answering processes a
Formatting considerations
Basic document formatting
７ Note
We don't support extraction of images from uploaded documents currently.
Product manuals
\nmanual, it extracts the headings and subheadings as questions and the subsequent content as
answers. See an example here
.
Below is an example of a manual with an index page, and hierarchical content
Many other types of documents can also be processed to generate question answer pairs,
provided they have a clear structure and layout. These include: Brochures, guidelines, reports,
white papers, scientific papers, policies, books, etc. See an example here
.
Below is an example of a semi-structured doc, without an index:
７ Note
Extraction works best on manuals that have a table of contents and/or an index page, and
a clear structure with hierarchical headings.
Brochures, guidelines, papers, and other files
\n![Image](images/page1213_image1.png)
\nCustom question answering now supports unstructured documents. A document that does not
have its content organized in a well-defined hierarchical manner, is missing a set structure or
has its content free flowing can be considered as an unstructured document.
Below is an example of an unstructured PDF document:
Unstructured document support
\n![Image](images/page1214_image1.png)
\n７ Note
QnA pairs are not extracted in the "Edit sources" tab for unstructured sources.
\n![Image](images/page1215_image1.png)
\nThe format for structured question-answers in DOC files, is in the form of alternating questions
and answers per line, one question per line followed by its answer in the following line, as
shown below:
text
Below is an example of a structured custom question answering word document:
） Important
Support for unstructured file/content is available only in custom question answering.
Structured custom question answering document
Question1
Answer1
Question2
Answer2
\nCustom question answering in the form of structured .txt, .tsv or .xls files can also be uploaded
to custom question answering to create or augment a project. These can either be plain text, or
can have content in RTF or HTML. Question answer pairs have an optional metadata field that
can be used to group question answer pairs into categories.
Structured TXT, TSV and XLS Files
ﾉ
Expand table
\n![Image](images/page1217_image1.png)
\nQuestion
Answer
Metadata (1 key: 1 value)
Question1
Answer1
Key1:Value1 | Key2:Value2
Question2
Answer2
Key:Value
Any additional columns in the source file are ignored.
Importing a project replaces the content of the existing project. Import requires a structured
.tsv file that contains data source information. This information helps group the question-
answer pairs and attribute them to a particular data source. Question answer pairs have an
optional metadata field that can be used to group question answer pairs into categories. The
import format needs to be similar to the exported knowledgebase format.
Question
Answer
Source
Metadata (1 key: 1 value)
QnaId
Question1
Answer1
Url1
Key1:Value1 | Key2:Value2
QnaId 1
Question2
Answer2
Editorial
Key:Value
QnaId 2
Use headings and subheadings to denote hierarchy. For example, You can h1 to denote
the parent question answer and h2 to denote the question answer that should be taken
as prompt. Use small heading size to denote subsequent hierarchy. Do not use style,
color, or some other mechanism to imply structure in your document, custom question
answering will not extract the multi-turn prompts.
First character of heading must be capitalized.
Do not end a heading with a question mark, ? .
Sample documents:
Surface Pro (docx)
Contoso Benefits (docx)
Contoso Benefits (pdf)
Custom question answering can support FAQ web pages in three different forms:
Structured data format through import
ﾉ
Expand table
Multi-turn document formatting
FAQ URLs
\nPlain FAQ pages
FAQ pages with links
FAQ pages with a Topics Homepage
This is the most common type of FAQ page, in which the answers immediately follow the
questions in the same page.
In this type of FAQ page, questions are aggregated together and are linked to answers that are
either in different sections of the same page, or in different pages.
Below is an example of an FAQ page with links in sections that are on the same page:
This type of FAQ has a Topics page where each topic is linked to a corresponding set of
questions and answers on a different page. Question answer crawls all the linked pages to
extract the corresponding questions & answers.
Plain FAQ pages
FAQ pages with links
Parent Topics page links to child answers pages
\n![Image](images/page1219_image1.png)
\nBelow is an example of a Topics page with links to FAQ sections in different pages.
Custom question answering can process semi-structured support web pages, such as web
articles that would describe how to perform a given task, how to diagnose and resolve a given
problem, and what are the best practices for a given process. Extraction works best on content
that has a clear structure with hierarchical headings.
TSV and XLS files, from exported projects, can only be used by importing the files from the
Settings page in Language Studio. They cannot be used as data sources during project creation
or from the + Add file or + Add URL feature on the Settings page.
When you import the project through these TSV and XLS files, the question answer pairs get
added to the editorial source and not the sources from which the question and answers were
extracted in the exported project.
Support URLs
７ Note
Extraction for support articles is a new feature and is in early stages. It works best for
simple pages, that are well structured, and do not contain complex headers/footers.
Import and export project
\n![Image](images/page1220_image1.png)