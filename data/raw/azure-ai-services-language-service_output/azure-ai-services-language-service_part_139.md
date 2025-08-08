Abstractive summarization: Generates a summary with concise, coherent
sentences or words that aren't verbatim extract sentences from the original
source.
Summary texts: Abstractive summarization returns a summary for each
contextual input range. A long input can be segmented so multiple groups
of summary texts can be returned with their contextual input range.
Contextual input range: The range within the input that was used to
generate the summary text.
As an example, consider the following paragraph of text:
"At Microsoft, we are on a quest to advance AI beyond existing techniques, by taking a
more holistic, human-centric approach to learning and understanding. As Chief
Technology Officer of Azure AI services, I have been working with a team of amazing
scientists and engineers to turn this quest into a reality. In my role, I enjoy a unique
perspective in viewing the relationship among three attributes of human cognition:
monolingual text (X), audio or visual sensory signals, (Y) and multilingual (Z). At the
intersection of all three, there's magic—what we call XYZ-code as illustrated in Figure
1—a joint representation to create more powerful AI that can speak, hear, see, and
understand humans better. We believe XYZ-code enables us to fulfill our long-term
vision: cross-domain transfer learning, spanning modalities and languages. The goal
is to have pretrained models that can jointly learn representations to support a broad
range of downstream AI tasks, much in the way humans do today. Over the past five
years, we achieve human performance on benchmarks in conversational speech
recognition, machine translation, conversational question answering, machine
reading comprehension, and image captioning. These five breakthroughs provided us
with strong signals toward our more ambitious aspiration to produce a leap in AI
capabilities, achieving multi-sensory and multilingual learning that is closer in line
with how humans learn and understand. I believe the joint XYZ-code is a
foundational component of this aspiration, if grounded with external knowledge
sources in the downstream AI tasks."
The text summarization API request is processed upon receipt of the request by
creating a job for the API backend. If the job succeeded, the output of the API is
returned. The output is available for retrieval for 24 hours. After this time, the
output is purged. Due to multilingual and emoji support, the response can contain
text offsets. For more information, see how to process offsets.
If we use the preceding example, the API might return these summaries:
Extractive summarization:
\n"At Microsoft, we are on a quest to advance AI beyond existing techniques, by
taking a more holistic, human-centric approach to learning and
understanding."
"We believe XYZ-code enables us to fulfill our long-term vision: cross-domain
transfer learning, spanning modalities and languages."
"The goal is to have pretrained models that can jointly learn representations
to support a broad range of downstream AI tasks, much in the way humans do
today."
Abstractive summarization:
"Microsoft is taking a more holistic, human-centric approach to learning and
understanding. We believe XYZ-code enables us to fulfill our long-term vision:
cross-domain transfer learning, spanning modalities and languages. Over the
past five years, we achieved human performance on benchmarks in
conversational speech recognition."
To use summarization, you submit for analysis and handle the API output in your
application. Analysis is performed as-is, with no added customization to the model used
on your data. There are two ways to use summarization:
Development
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use entity
linking with text examples with your own data when you sign up. For
more information, see the Azure AI Foundry website
 or Azure AI
Foundry documentation.
REST API or Client
library (Azure SDK)
Integrate text summarization into your applications using the REST
API, or the client library available in various languages. For more
information, see the summarization quickstart.
Get started with summarization
Text summarization
ﾉ
Expand table
Input requirements and service limits
\nFeedback
Was this page helpful?
Summarization takes text for analysis. For more information, see Data and
service limits in the how-to guide.
Summarization works with various written languages. For more information,
see language support.
As you use text summarization in your applications, see the following reference
documentation and samples for Azure AI Language:
Development option / language
Reference documentation
Samples
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
An AI system includes not only the technology, but also the people who use it, the
people affected by it, and the deployment environment. Read the transparency note for
summarization to learn about responsible AI use and deployment in your systems. For
more information, see the following articles:
Transparency note for Azure AI Language
Integration and responsible use
Characteristics and limitations of summarization
Data, privacy, and security
Text summarization
Reference documentation and code samples
ﾉ
Expand table
Responsible AI
Yes
No
\nProvide product feedback 
| Get help at Microsoft Q&A
\nQuickstart: using text, document and
conversation summarization
Article • 02/21/2025
Create a Project in Foundry in the Azure AI Foundry Portal
Using the left side pane, select Playgrounds. Then select the Try the Language
Playground button.
） Important
Our preview region, Sweden Central, showcases our latest and continually evolving
LLM fine tuning techniques based on GPT models. You are welcome to try them out
with a Language resource in the Sweden Central region.
Conversation summarization is only available using:
REST API
Python
C#
Prerequisites
Navigate to the Azure AI Foundry Playground
\nThe Language Playground consists of four sections:
Top banner: You can select any of the currently available Language services here.
Right pane: This pane is where you can find the Configuration options for the
service, such as the API and model version, along with features specific to the
service.
Center pane: This pane is where you enter your text for processing. After the
operation is run, some results are shown here.
Right pane: This pane is where Details of the run operation are shown.
Here you can select the Summarization capability that you want to use by choosing one
of these top banner tiles: Summarize conversation, Summarize for call center, or
Summarize text.

Use Summarization in the Azure AI Foundry
Playground
\n![Image](images/page1386_image1.png)
\nSummarize conversation is designed to recap conversations and segment long
meetings into timestamped chapters.
In Configuration there are the following options:
Option
Description
Select API version
Select which version of the API to use.
Select text language
Select the language of the input text.
Summarization
Aspects
Different methods of summarization that are returned. At least one must
be selected.
After your operation is completed, the Details section contains the following fields for
the selected methods of summarization:
Field
Description
Sentence
Recap
A recap of the processed text. The Recap Summarization aspect must be toggled on
for this to appear.
Chapter
Title
A list of titles for semantically segmented chapters with corresponding timestamps.
The Chapter title Summarization aspect must be toggled on for this to appear.
Narrative
A list of narrative summaries for semantically segmented chapters with
corresponding timestamps. The Narrative Summarization aspect must be toggled on
for this to appear.
Use Summarize conversation
ﾉ
Expand table
ﾉ
Expand table
\nSummarize for call center is designed to recap calls and summarize them for customer
issues and resolutions.
In Configuration there are the following options:
Option
Description
Select API version
Select which version of the API to use.
Select text language
Select the language of the input text.
Summarization
Aspects
Different methods of summarization that are returned. At least one must
be selected.
After your operation is completed, the Details section contains the following fields for
the selected methods of summarization:
Field
Description
Sentence
Recap
A recap of the processed text. The Recap Summarization aspect must be toggled on
for this to appear.
Issue
A summary of the customer issue in the customer-and-agent conversation. The Issue

Use Summarize for call center
ﾉ
Expand table
ﾉ
Expand table
\n![Image](images/page1388_image1.png)
\nField
Description
Summarization aspect must be toggled on for this to appear.
Resolution
A summary of the solutions tried in the customer-and-agent conversation. The
Resolution Summarization aspect must be toggled on for this to appear.
Summarize text is designed to summarize and extract key information at scale from
text.
In Configuration there are the following options:
Option
Description
Extractive summarization
The service will produce a summary by extracting salient
sentences.
Number of sentences
The number of sentences that Extractive summarization
will extract.
Abstractive summarization
the service will generate a summary with novel
sentences.
Summary length
The length of the summary generated by Abstractive
summarization.
Define keywords for summary focus
(preview)
Helps focus summarization on a particular set of
keywords.

Use Summarize text
ﾉ
Expand table
\n![Image](images/page1389_image1.png)
\nAfter your operation is completed, the Details section contains the following fields for
the selected methods of summarization:
Field
Description
Extractive
summary
Extracted sentences from the input text, ranked by detected relevance and
prioritized for words in the Defined keywords for summary focus field, if any.
Sentences are sorted by rank score of detected relevance (default) or order of
appearance in the input text.
Abstractive
summary
A summary of the input text of the length chosen in the Summary length field
and prioritized for words in the Defined keywords for summary focus field, if
any.
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other
resources associated with it.
Azure portal
Azure CLI
How to call text summarization
How to call conversation summarization
ﾉ
Expand table

Clean up resources
Next steps
\n![Image](images/page1390_image1.png)