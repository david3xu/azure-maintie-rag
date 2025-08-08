Use Conversational Language Understanding to build end-to-end conversational bots.
Use CLU to build and train a custom natural language understanding model based on a
specific domain and the expected users' utterances. Integrate it with any end-to-end
conversational bot so that it can process and analyze incoming text in real time to identify
the intention of the text and extract important information from it. Have the bot perform
the desired action based on the intention and extracted information. An example would
be a customized retail bot for online shopping or food ordering.
Use Question Answering for customer support. In most customer support scenarios,
common questions are asked frequently. Question Answering lets you instantly create a
chat bot from existing support content, and this bot can act as the front-line system for
handling customer queries. If the questions can't be answered by the bot, then additional
components can help identify and flag the question for human intervention.
Azure AI Language features only process text. The fidelity and formatting of the incoming text
will affect the performance of the system. Make sure you consider the following:
Speech transcription quality may affect the quality of the results. If your source data is
voice, make sure you use the highest quality combination of automatic and human
transcription to ensure the best performance. Consider using custom speech models for
better quality results.
Lack of standard punctuation or casing may affect the quality of your results. If you are
using a speech system, like Azure AI Speech to Text, be sure to select the option to
include punctuation.
Optical character recognition (OCR) quality may affect the quality of the system. If your
source data is images and you use OCR technology to generate the text, incorrectly
generated text may affect the performance of the system. Consider using custom OCR
models to help improve the quality of results.
If your data includes frequent misspellings, consider using Bing Spell Check to correct
misspellings.
Tabular data may not be identified correctly depending on how you send the table text to
the system. Assess how you send text from tables in source documents to the service. For
tables in documents, consider using Azure AI Document Intelligence or a similar service.
Limitations
The quality of the incoming text to the system will affect your
results.
\nThis will allow you to get the appropriate keys and values to send to Azure AI Language
with contextual keys that are close enough to the values for the system to properly
recognize the entities.
Microsoft trained its Azure AI Language feature models (with the exception of language
detection) using natural language text data that is comprised primarily of fully formed
sentences and paragraphs. Therefore, using this service for data that most closely
resembles this type of text will yield the best performance. We recommend avoiding use
of this service to evaluate incomplete sentences and phrases where possible, as the
performance may be reduced.
The service only supports single language text. If your text includes multiple languages
for example "the sandwich was bueno", the output may not be accurate.
The language code must match the input text language to get accurate results. If you are
unsure about the input language you can use the language detection feature.
Some features of Azure AI Language return confidence scores and can be evaluated using the
approach described in the following sections. Other features which do not return a confidence
score (such as key word extraction and summarization) will need to be evaluated using
different methods.
The sentiment, named entity recognition, language detection and health functions all return a
confidence score as a part of the system response. This is an indicator of how confident the
service is with the system's response. A higher value indicates that the service is more
confident that the result is accurate. For example, the system recognizes entity of category U.S.
Driver's License Number on the text 555 555 555 when given the text "My NY driver's license
number is 555 555 555" with a score of .75 and might recognize category U.S. Driver's License
Number on the text 555 555 555 with a score of .65 when given the text "My NY DL number is
555 555 555". Given the more specific context in the first example, the system is more
confident in its response. In many cases, the system response can be used without examining
the confidence score. In other cases, you can choose to use a response only if its confidence
score is above a specified confidence score threshold.
Best practices for improving system performance
Understand confidence scores for sentiment analysis, named
entity recognition, language detection, and health functions
Understand and measuring performance
\nThe performance of Azure AI Language features is measured by examining how well the
system recognizes the supported NLP concepts (at a given threshold value in comparison with
a human judge.) For named entity extraction (NER), for example, one might count the true
number of phone number entities in some text based on human judgement, and then compare
with the output of the system from processing the same text. Comparing human judgement
with the system recognized entities would allow you to classify the events into two kinds of
correct (or "true") events and two kinds of incorrect (or "false") events.
Outcome
Correct/Incorrect
Definition
Example
True
Positive
Correct
The system returns the
same result that would be
expected from a human
judge.
The system correctly recognizes PII entity
of category Phone Number on the text 1-
234-567-8910 when given the text: "You
can reach me at my office number 1-234-
567-9810."
True
Negative
Correct
The system does not return
a result, and this aligns
with what would be
expected from human
judge.
The system does not recognize any PII
entity when given the text: "You can
reach me at my office number."
False
Positive
Incorrect
The system returns a result
where a human judge
would not.
The system incorrectly recognizes PII
entity of category Phone Number for the
text office number when given the text:
"You can reach me at my office number."
False
Negative
Incorrect
The system does not return
a result when a human
judge would.
The system incorrectly misses a Phone
Number PII entity on the text 1-234-567-
8910 when given the text: "You can reach
me at my office number 1-234-567-
9810."
Azure AI Language features will not always be correct. You'll likely experience both false
negative and false positive errors. It's important to consider how each type of error will affect
your system. Carefully think through scenarios where true events won't be recognized and
where incorrect events will be recognized and what the downstream effects might be in your
implementation. Make sure to build in ways to identify, report and respond to each type of
error. Plan to periodically review the performance of your deployed system to ensure errors are
being handled appropriately.
ﾉ
Expand table
How to set confidence score thresholds
\nYou can choose to make decisions in your system based on the confidence score the system
returns. You can adjust the confidence score threshold your system uses to meet your needs. If
it is more important to identify all potential instances of the NLP concepts you want, you can
use a lower threshold. This means that you may get more false positives but fewer false
negatives. If it is more important for your system to recognize only true instances of the feature
you're calling, you can use a higher threshold. If you use a higher threshold, you may get fewer
false positives but more false negatives. Different scenarios call for different approaches. In
addition, threshold values may not have consistent behavior across individual features of Azure
AI Language and categories of entities. For example, do not make assumptions that using a
certain threshold for NER category Phone Number would be sufficient for another NER
category, or that a threshold you use in NER would work similarly for Sentiment Analysis.
Therefore, it is critical that you test your system with any thresholds you are considering using
with real data to determine the effects of various threshold values of your system in the
context that it will be used.
At Microsoft, we strive to empower every person on the planet to achieve more. An essential
part of this goal is working to create technologies and products that are fair and inclusive.
Fairness is a multi-dimensional, sociotechnical topic and impacts many different aspects of our
product development. You can learn more about Microsoft’s approach to fairness here
.
One dimension we need to consider is how well the system performs for different groups of
people. This may include looking at the accuracy of the model as well as measuring the
performance of the complete system. Research has shown that without conscious effort
focused on improving performance for all groups, it is often possible for the performance of an
AI system to vary across groups based on factors such as race, ethnicity, language, gender, and
age.
Each service and feature is different, and our testing may not perfectly match your context or
cover all scenarios required for your use case. We encourage developers to thoroughly
evaluate error rates for the service with real-world data that reflects your use case, including
testing with users from different demographic groups.
For Azure AI Language, certain dialects and language varieties within our supported languages
and text from some demographic groups may not yet have enough representation in our
current training datasets. We encourage you to review our responsible use guidelines, and if
you encounter performance differences, we encourage you to let us know.
Fairness
Performance varies across features and languages
\nVarious languages are supported for each Azure AI Language feature. You may find that
performance for a particular feature is not consistent with another feature. Also, you may find
that for a particular feature that performance is not consistent across various languages.
If you are using any of the features below, be sure to review the specific information for that
feature.
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for text analytics for health
Transparency note for key phrase extraction
Transparency note for language detection
Transparency note for question answering
Transparency note for summarization
Transparency note for sentiment analysis
Transparency note for custom Named Entity Recognition (NER)
Transparency note for custom text classification
Transparency note for conversational language understanding
Also, make sure to review:
Guidance for integration and responsible use with Azure AI Language
Data Privacy for Azure AI Language
Next steps
See also
\nGuidance for integration and responsible
use with Azure AI Language
06/24/2025
Microsoft wants to help you responsibly develop and deploy solutions that use Azure AI
Language. We are taking a principled approach to upholding personal agency and dignity by
considering the fairness, reliability & safety, privacy & security, inclusiveness, transparency, and
human accountability of our AI systems. These considerations are in line with our commitment
to developing Responsible AI.
This article discusses Azure AI Language features and the key considerations for making use of
this technology responsibly. Consider the following factors when you decide how to use and
implement AI-powered products and features.
When you're getting ready to deploy AI-powered products or features, the following activities
help to set you up for success:
Understand what it can do: Fully assess the capabilities of any AI model you are using to
understand its capabilities and limitations. Understand how it will perform in your
particular scenario and context.
Test with real, diverse data: Understand how your system will perform in your scenario by
thoroughly testing it with real life conditions and data that reflects the diversity in your
users, geography and deployment contexts. Small datasets, synthetic data and tests that
don't reflect your end-to-end scenario are unlikely to sufficiently represent your
production performance.
Respect an individual's right to privacy: Only collect data and information from
individuals for lawful and justifiable purposes. Only use data and information that you
have consent to use for this purpose.
Legal review: Obtain appropriate legal advice to review your solution, particularly if you
will use it in sensitive or high-risk applications. Understand what restrictions you might
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
General guidelines
\nneed to work within and your responsibility to resolve any issues that might come up in
the future. Do not provide any legal advice or guidance.
System review: If you're planning to integrate and responsibly use an AI-powered
product or feature into an existing system of software, customers or organizational
processes, take the time to understand how each part of your system will be affected.
Consider how your AI solution aligns with Microsoft's Responsible AI principles.
Human in the loop: Keep a human in the loop, and include human oversight as a
consistent pattern area to explore. This means constant human oversight of the AI-
powered product or feature and maintaining the role of humans in decision-making.
Ensure you can have real-time human intervention in the solution to prevent harm. This
enables you to manage where the AI model doesn't perform as required.
Security: Ensure your solution is secure and has adequate controls to preserve the
integrity of your content and prevent unauthorized access.
Customer feedback loop: Provide a feedback channel that allows users and individuals to
report issues with the service once it's been deployed. Once you've deployed an AI-
powered product or feature it requires ongoing monitoring and improvement – be ready
to implement any feedback and suggestions for improvement.
Microsoft Responsible AI principles
Microsoft Responsible AI resources
Microsoft Azure Learning courses on Responsible AI
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for the health feature
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Data Privacy and Security for Azure AI Language
Learn more about Responsible AI
See also
\nData, privacy, and security for Azure AI
Language
06/24/2025
This article provides details regarding how Azure AI Language processes your data. Azure AI
Language is designed with compliance, privacy, and security in mind. However, you are
responsible for its use and the implementation of this technology. It's your responsibility to
comply with all applicable laws and regulations in your jurisdiction.
Azure AI Language processes text data that is sent by the customer to the system for the
purposes of getting a response from one of the available features.
All results of the requested feature are sent back to the customer in the API response as
specified in the API reference. For example, if Language Detection is requested, the
language code is returned along with a confidence score for each text record.
Azure AI Language uses aggregate telemetry such as which APIs are used and the
number of calls from each subscription and resource for service monitoring purposes.
Azure AI Language doesn't store or process customer data outside the region where the
customer deploys the service instance.
Azure AI Language encrypts all content, including customer data, at rest.
Data sent in synchronous or asynchronous calls may be temporarily stored by Azure AI
Language for up to 48 hours only and is purged thereafter. This data is encrypted and is
only accessible to authorized on call engineers when service support is needed for
debugging purposes in the event of a catastrophic failure. To prevent this temporary
storage of input data, the LoggingOptOut query parameter can be set accordingly. By
default, this parameter is set to false for Language Detection, Key Phrase Extraction,
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What data does Azure AI Language process and
how does it process it?
How is data retained and what customer controls
are available?
\nSentiment Analysis and Named Entity Recognition endpoints. The LoggingOptOut
parameter is true by default for the PII and health feature endpoints. More information on
the LoggingOptOut query parameter is available in the API reference.
To learn more about Microsoft's privacy and security commitments, visit the Microsoft Trust
Center
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for the health feature
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Guidance for integration and responsible use with Azure AI Language
See also
\nMigrating to Azure AI Language
06/30/2025
On November 2nd 2021, Azure AI Language was released into public preview. This language
service unifies the Text Analytics, QnA Maker, and LUIS service offerings, and provides several
new features as well.
Text Analytics has been incorporated into the language service, and its features are still
available. If you were using Text Analytics features, your applications should continue to work
without breaking changes. If you are using Text Analytics API (v2.x or v3), see the Text Analytics
migration guide to migrate your applications to the unified Language endpoint and the latest
client library.
Consider using one of the available quickstart articles to see the latest information on service
endpoints, and API calls.
If you're using Language Understanding (LUIS), you can import your LUIS JSON file to the new
Conversational language understanding feature.
If you're using QnA Maker, see the migration guide for information on migrating knowledge
bases from QnA Maker to question answering.
Azure AI Language overview
Conversational language understanding overview
Question answering overview
Do I need to migrate to the language service if I
am using Text Analytics?
How do I migrate to the language service if I am
using LUIS?
How do I migrate to the language service if I am
using QnA Maker?
See also