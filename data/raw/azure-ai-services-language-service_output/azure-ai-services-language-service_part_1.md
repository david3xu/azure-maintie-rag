Tell us about your PDF experience.
O V E R V I E W
What is Azure
AI Language?
W H A T ' S  N E W
What's new?
C O N C E P T
Responsible
use of AI
Extract information
Use Natural Language
Understanding (NLU) to extract
information from unstructured
text. For example, identify key…
ｅExtract key phrases
ｅFind linked entities
ｅNamed Entity Recognition
(NER)
ｅPersonally Identifiable
Information (PII) detection
ｅCustom Named Entity
Recognition (custom NER)
ｅText analytics for health
Summarize text-based
content
Summarize lengthy documents
and conversation transcripts.
ｅDocument summarization
ｅConversation
summarization
Classify Text
Use Natural Language
Understanding (NLU) to detect
the language or classify the
sentiment of text you have.…
ｅAnalyze sentiment and
mine text for opinions
ｅDetect Language
ｅCustom text classification
Azure AI Language documentation
Learn how to integrate AI into your applications that can extract information, classify text,
understand conversational language, answer questions and more.
\nAnswer questions
Provide answers to questions
being asked in unstructured
texts using our prebuilt
capabilities, or customize you…
ｅQuestion answering
Understand
conversations
Create your own models to
classify conversational
utterances and extract detailed
information from them to fulf…
ｅConversational language
understanding
ｅOrchestration workflow
Translate text
Use cloud-based neural
machine translation to build
intelligent, multi-language
solutions for your applications.
ｅUse machine translation on
text.
Resources
Pricing
Learn
Support
Support and help
\nWhat is Azure AI Language?
06/21/2025
Azure AI Language is a cloud-based service that provides Natural Language Processing (NLP)
features for understanding and analyzing text. Use this service to help build intelligent
applications using the web-based Language Studio, REST APIs, and client libraries.
This Language service unifies the following previously available Azure AI services: Text
Analytics, QnA Maker, and LUIS. If you need to migrate from these services, see the migration
section.
The Language service also provides several new features as well, which can either be:
Preconfigured, which means the AI models that the feature uses aren't customizable. You
just send your data, and use the feature's output in your applications.
Customizable, which means you train an AI model using our tools to fit your data
specifically.
Language features are also utilized in agent templates
:
Intent routing agent
 detects user intent and provides exact answering. Perfect for
deterministically intent routing and exact question answering with human controls.
Exact question answering agent
 answers high-value predefined questions
deterministically to ensure consistent and accurate responses.
Azure AI Foundry
 enables you to use most of the following service features without needing
to write code.
Available features
 Tip
Unsure which feature to use? See Which Language service feature should I use to help
you decide.
Named Entity Recognition (NER)
\nNamed entity recognition identifies different entries in text and categorizes them into
predefined types.

Personal and health data information detection
\n![Image](images/page4_image1.png)
\nPII detection identifies entities in text and conversations (chat or transcripts) that are associated
with individuals.

Language detection
\n![Image](images/page5_image1.png)

![Image](images/page5_image2.png)
\nLanguage detection evaluates text and detects a wide range of languages and variant dialects.
Sentiment analysis and opinion mining preconfigured features that help you understand public
perception of your brand or topic. These features analyze text to identify positive or negative
sentiments and can link them to specific elements within the text.

Sentiment Analysis and opinion mining

Summarization
\n![Image](images/page6_image1.png)

![Image](images/page6_image2.png)
\n
\n![Image](images/page7_image1.png)

![Image](images/page7_image2.png)

![Image](images/page7_image3.png)
\nSummarization condenses information for text and conversations (chat and transcripts). Text
summarization generates a summary, supporting two approaches: Extractive summarization
creates a summary by selecting key sentences from the document and preserving their original
positions. In contrast, abstractive summarization generates a summary by producing new,
concise, and coherent sentences or phrases that aren't directly copied from the original
document. Conversation summarization recaps and segments long meetings into timestamped
chapters. Call center summarization summarizes customer issues and resolution.
Key phrase extraction is a preconfigured feature that evaluates and returns the main concepts
in unstructured text, and returns them as a list.
Key phrase extraction

Entity linking

\n![Image](images/page8_image1.png)

![Image](images/page8_image2.png)
\nEntity linking is a preconfigured feature that disambiguates the identity of entities (words or
phrases) found in unstructured text and returns links to Wikipedia.
Text analytics for health Extracts and labels relevant health information from unstructured text.
Custom text classification enables you to build custom AI models to classify unstructured text
documents into custom classes you define.
Text analytics for health

Custom text classification

Custom Named Entity Recognition (Custom NER)
\n![Image](images/page9_image1.png)

![Image](images/page9_image2.png)
\nCustom NER enables you to build custom AI models to extract custom entity categories (labels
for words or phrases), using unstructured text that you provide.

Conversational language understanding

\n![Image](images/page10_image1.png)

![Image](images/page10_image2.png)