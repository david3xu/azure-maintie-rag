Guidance for integration and responsible
use with summarization
06/24/2025
Microsoft wants to help you develop and deploy solutions that use the summarization feature
in a responsible manner. Microsoft takes a principled approach to uphold personal agency and
dignity, by considering the following aspects of an AI system: fairness, reliability and safety,
privacy and security, inclusiveness, transparency, and human accountability. These
considerations reflect our commitment to developing Responsible AI
.
When you're getting ready to integrate and responsibly use AI-powered products or features,
the following activities help to set you up for success. Although each guideline is unlikely to
apply to all scenarios, consider them as a starting point for mitigating possible risks:
Understand what it can do, and how it might be misused. Fully assess the capabilities of
any AI system you're using to understand its capabilities and limitations. The particular
testing that Microsoft conducts might not reflect your scenario. Understand how it will
perform in your particular scenario, by thoroughly testing it with real-life conditions and
diverse data that reflect your context. Include fairness considerations.
Test with real, diverse data. Understand how your system will perform in your scenario.
Test it thoroughly with real-life conditions, and data that reflects the diversity in your
users, geography, and deployment contexts. Small datasets, synthetic data, and tests that
don't reflect your end-to-end scenario are unlikely to sufficiently represent your
production performance.
Evaluate the system. Consider using adversarial testing, where trusted testers attempt to
find system failures, poor performance, or undesirable behaviors. This information helps
you to understand risks and how to mitigate them. Communicate the capabilities and
limitations to stakeholders. To help you evaluate your system, you might find some of
these resources useful: Checklist on GitHub
, Stereotyping Norwegian Salmon: An
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
General guidelines
\nInventory of Pitfalls in Fairness Benchmark Datasets
 (Blodgett et al., 2021), and On the
Dangers of Stochastic Parrots: Can Language Models Be Too Big?
 (Bender et al., 2021).
Learn about fairness. AI systems can behave unfairly for a variety of reasons. Some are
social, some are technical, and some are a combination of the two. There are seldom
clear-cut solutions. Mitigation methods are usually context dependent. Learning about
fairness can help you learn what to expect and how you might mitigate potential harms.
To learn more about the approach that Microsoft uses, see Responsible AI resources
,
the AI fairness checklist
, and resources from Microsoft Research
.
Respect an individual's right to privacy. Only collect data and information from
individuals for lawful and justifiable purposes. Only use data and information that you
have consent to use for this purpose.
Conduct user testing during development, and solicit feedback after deployment.
Consider using value-sensitive design to identify stakeholders. Work with them to identify
their values to design systems that support those values. Seek feedback from a diverse
group of people during the development and evaluation process. Use strategies like
Community Jury.
Undertake user testing with diverse stakeholders. Then analyze the results broken down
by stakeholder groups. Include stakeholders from different demographic groups.
Consider conducting online experiments, ring testing, dogfooding, field trials, or pilots in
deployment contexts.
Limit the length, structure, rate, and source of inputs and outputs. Restricting input and
output length can reduce the likelihood of risks. Such risks include producing undesirable
content, misuse for an overly general-purpose beyond the intended application use cases,
or other harmful, disallowed, or unintended scenarios.
Consider requiring prompts to be structured a certain way. They can be confined to a
particular topic, or drawn from validated sources, like a dropdown field. Consider
structuring the output so it isn't overly open-ended. Consider returning outputs from
validated, reliable source materials (such as existing support articles), rather than
connecting to the internet. This restriction can help your application stay on task and
mitigate unfair, unreliable, or offensive behavior. Putting rate limits in place can further
reduce misuse.
Implement blocklists and content moderation. Keep your application on topic. Consider
blocklists and content moderation strategies to check inputs and outputs for undesirable
content. The definition of undesired content depends on your scenario, and can change
over time. It might include hate speech, text that contains profane words or phrases,
misinformation, and text that relates to sensitive or emotionally charged topics. Checking
\ninputs can help keep your application on topic, even if a malicious user tries to produce
undesired content. Checking API outputs allows you to detect undesired content
produced by the system. You can then replace it, report it, ask the user to enter different
input, or provide input examples.
Authenticate users. To make misuse more difficult, consider requiring that customers sign
in and, if appropriate, link a valid payment method. Consider working only with known,
trusted customers in the early stages of development.
Ensure human oversight. Especially in higher-stakes scenarios, maintain the role of
humans in decision making. Disclose what the AI has done versus what a human has
done.
Based on your scenario, there are various stages in the lifecycle in which you can add
human oversight. Ensure you can have real-time human intervention in the solution to
prevent harm. For example, when generating summaries, editors should review the
summaries before publication. Ideally, assess the effectiveness of human oversight prior
to deployment, through user testing, and after deployment.
Have a customer feedback loop. Provide a feedback channel that allows users and
individuals to report issues with the service after deployment. Issues might include unfair
or undesirable behaviors. After you've deployed an AI-powered product or feature, it
requires ongoing monitoring and improvement. Establish channels to collect questions
and concerns from stakeholders who might be directly or indirectly affected by the
system, such as employees, visitors, and the general public. Examples include:
Feedback features built into app experiences.
An easy-to-remember email address for feedback.
Conduct a legal review. Obtain appropriate legal advice to review your solution,
particularly if you'll use it in sensitive or high-risk applications. Know what restrictions you
might need to work within. Understand your responsibility to resolve any issues that
might come up in the future. Ensure the appropriate use of datasets.
Conduct a system review. You might plan to integrate and responsibly use an AI-
powered product or feature into an existing system of software, customers, and
organizational processes. If so, take the time to understand how each part of your system
will be affected. Consider how your AI solution aligns with the principles of responsible AI
that Microsoft uses.
Security. Ensure that your solution is secure, and has adequate controls to preserve the
integrity of your content and prevent any unauthorized access.
\nAssess your application for alignment with Responsible AI principles
.
Use the Microsoft HAX Toolkit
. The toolkit recommends best practices for how AI
systems should behave on initial interaction, during regular interaction, when they're
inevitably wrong, and over time.
Follow the Microsoft guidelines for responsible development of conversational AI
systems
. Use the guidelines when you develop and deploy language models that
power chat bots, or other conversational AI systems.
Use Microsoft Inclusive Design Guidelines
 to build inclusive solutions.
Recommended content
\nCharacteristics and limitations for
Summarization
06/24/2025
Large-scale, natural language models are trained with publicly available text data which
typically contain societal biases. Such data can potentially behave in ways that are unfair,
unreliable, or offensive. This behavior, in turn, may cause harms of varying severities. These
types of harms aren't mutually exclusive. A single model can exhibit more than one type of
harm, potentially relating to multiple groups of people. For example:
Allocation: It's possible to use language models in ways that lead to unfair allocation of
resources or opportunities. For example, automated systems that screen resumes can
withhold employment opportunities from women, if these systems are trained on resume
data that reflects the existing gender imbalance in the technology industries.
Quality of service: Language models can fail to provide the same quality of service to
some people as they do to others. For example, summary generation can work less well
for some dialects or language varieties, because of their lack of representation in the
training data. The models are trained primarily on English text. English language varieties
less well represented in the training data might experience worse performance.
Stereotyping: Language models can reinforce stereotypes. For example, when translating
He is a nurse and She is a doctor into a genderless language, such as Turkish, and then
back into English, you can get an error. Many machine translation systems yield the
stereotypical (and incorrect) results of She is a nurse and He is a doctor.
Demeaning: Language models can demean people. For example, an open-ended content
generation system with inappropriate mitigation might produce offensive text, targeted
at a particular group of people.
Over- and underrepresentation: Language models can over- or under-represent groups
of people, or even erase them entirely. For example, toxicity detection systems that rate
text containing the word gay as toxic might lead to the under-representation, or even
erasure, of legitimate text written by or about the LGBTQ community.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
\nInappropriate or offensive content: Language models can produce other types of
inappropriate or offensive content. Examples include:
Hate speech.
Text that contains profane words or phrases.
Text that relates to illicit activities.
Text that relates to contested, controversial, or ideologically polarizing topics.
Misinformation.
Text that's manipulative.
Text that relates to sensitive or emotionally charged topics.
For example, suggested-reply systems that are restricted to positive replies
 can suggest
inappropriate or insensitive replies for messages about negative events.
False information: The service doesn't check facts or verify content provided by
customers or users. Depending on how you've developed your application, it might
promote false information unless you've built in an effective mitigation for this possibility.
Inaccurate summary: The feature uses an abstractive summarization method, in which the
model doesn't simply extract contexts from the input text. Instead, the model tries to
understand the input and paraphrase the key information in succinct natural sentences.
However, there can be information or accuracy loss.
Genre consideration: The training data used to train the summarization feature in Azure
AI services for language is mainly texts and transcripts between two participants. The
model might perform with lower accuracy for the input text in other types of genres, such
as documents or reports, which are less represented in the training data.
Language support: Most of the training data is in English, and in other commonly used
languages like German and Spanish. The trained models might not perform as well on
input in other languages, because these languages are less represented in the training
data. Microsoft is invested in expanding the language support of this feature.
The performance of the models varies based on the scenario and input data. The following
sections are designed to help you understand key concepts about performance.
You can use document summarization in a wide range of applications, each with different
focuses and performance metrics. Here, we broadly consider performance to mean the
application performs as you expect, including the absence of harmful outputs. There are
Best practices for improving system performance
Document summarization
\nseveral steps you can take to mitigate some of the concerns mentioned earlier in this
article, and to improve performance:
*Because the document summarization feature is trained on document-based texts, such
as news articles, scientific reports, and legal documents, when used with texts in different
genres that are less represented in the training data, such as conversations and
transcriptions, the system might product output with lower accuracy.
When used with texts that may contain errors or are less similar to well-formed
sentences, such as texts extracted from lists, tables, charts, or scanned in via OCR
(Optical Character Recognition), the document summarization feature may produce
output with lower accuracy.
Most of the training data is in commonly used languages such as English, German,
French, Chinese, Japanese, and Korean. The trained models may not perform as well
on input in other languages.
Documents must be "cracked," or converted, from their original format into plain and
unstructured text.
Although the service can handle a maximum of 25 documents per request, the
latency performance of the API increases with larger documents (it becomes slower).
This is especially true if the documents contain close to the maximum 125,000
characters. Learn more about system limits
The extractive summarization gives a score between 0 and 1 to each sentence and
returns the highest scored sentences per request. If you request a three-sentence
summary, the service returns the three highest scored sentences. If you request a
five-sentence summary from the same document, the service returns the next two
highest scored sentences in addition to the first three sentences.
The extractive summarization returns extracted sentences in their chronological order
by default. To change the order, specify sortBy. The accepted values for sortBy are
Offset (default). The value of Offset is the character positions of the extracted
sentences and the value of Rank is the rank scores of the extracted sentences.
Transparency note for Azure AI Language
Transparency note for named entity recognition
Transparency note for health
Transparency note for key phrase extraction
Transparency note for sentiment analysis
Guidance for integration and responsible use with language
Data privacy for language
Next steps
\n\nData, privacy, and security for Azure AI
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