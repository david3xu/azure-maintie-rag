In redaction scenarios, for example, false negatives could lead to personal information leakage.
For redaction scenarios, consider a process for human review to account for this type of error.
For sensitivity label scenarios, both false positives and false negatives could lead to
misclassification of documents. The audience may unnecessarily limited for documents labelled
as confidential where a false positive occurred. PII could be leaked where a false negative
occurred and a public label was applied.
You can adjust the threshold for confidence score your system uses to tune your system. If it is
more important to identify all potential instances of PII, you can use a lower threshold. This
means that you may get more false positives (non- PII data being recognized as PII entities),
but fewer false negatives (PII entities not recognized as PII). If it is more important for your
system to recognize only true PII data, you can use a higher threshold. Threshold values may
not have consistent behavior across individual categories of PII entities. Therefore, it is critical
that you test your system with real data it will process in production.
Make sure you understand all the entity categories for NER and PII that can be
recognized by the system. Depending on your scenario, your data may include other
information that could be considered personal but is not covered by the categories the
service currently supports.
Context is important for all entity categories to be correctly recognized by the system, as
it often is for humans to recognize an entity. For example, without context a ten-digit
number is just a number. However, given context like "You can reach me at my office
phone number 2345678901," both the system and a human can recognize the ten-digit
number as a phone number. Always include context when sending text to the system to
obtain the best possible performance.
Person names in particular require linguistic context. Send as much context as possible for
better person name detection.
For conversational data, consider sending more than a single turn in the conversation to
ensure higher likelihood that the required context is included with the actual entities.
In the following conversation, if you send a single row at a time, the passport number will
not have any context associated with it and the EU Passport Number PII category will not
be recognized.
Understanding performance for PII
System limitations and best practices for enhancing
performance
\nHi, how can I help you today?
I want to renew my passport
Sure, what is your current passport number?
Its 123456789, thanks.
However, if you send the whole conversation it will be recognized because the context is
included.
Sometimes multiple entity categories can be recognized for the same entity. If we take
the previous example:
Hi, how can I help you today?
I want to renew my passport
Sure, what is your current passport number?
Its 123456789, thanks.
Several different countries have the same format for passport numbers, so several
different specific entity categories may be recognized. In some cases, using the highest
confidence score may not be sufficient to choose the right entity class. If your scenario
depends on the specific entity category being recognized, you may need to disambiguate
the result elsewhere in your system either through a human review or additional
validation code. Thorough testing on real life data can help you identify if you're likely to
see multiple entity categories for recognized for your scenario.
Not all entity categories are supported in all languages for both NER and PII. Be sure to
check the entity type article for the entities in the language you want to detect.
Many international PII entities are supported. By default, the entity categories returned
are those that match the language code sent with the API call. If you expect entities from
locales other than the one specified, you will need to specify them with the
piiCategories  parameter. Learn more about how to specify what your response will
include in the API reference
. Learn more about the categories supported for each locale
in the named entity types documentation.
In PII redaction scenarios, if you are using the version of the API that includes the optional
parameter piiCategories , it is important that you consider all the PII categories that
could be present in your text. If you are redacting only specific entity categories or the
default entity categories for a specific locale, other PII entity categories that unexpectedly
appear in your text will be leaked. For example, if you have sent the EN-US locale and not
specified any optional PII categories and a German Driver's License Number is present in
your text, it will be leaked. To prevent this you would need to specify the German Driver's
License Number category in the piiCategories  parameter. In addition, if you have
\nspecified one or more categories using the piiCategories  parameter for the specified
locale, be aware that those are the only categories that would be redacted. For example, if
you have sent the EN-US locale and have specified U.S. Social Security Number (SSN) as
the PII category for redaction, then any other EN-US categories such as U.S. Driver's
License Number or U.S. Passport Number would be leaked if they appear in the input text.
Since the PII service returns PII categories that match the language code in the call,
consider verifying the language the input text is in if you're not sure what language or
locale it will be. You can use the Language Detection feature to do this.
The PII service only takes text as an input. If you are redacting information from
documents in other formats, make sure to carefully test your redaction code to ensure
identified entities are not accidentally leaked.
Transparency note for Azure AI Language
Transparency note for the health feature
Transparency note for Key Phrase Extraction
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Data Privacy and Security for Azure AI Language
Guidance for integration and responsible use with Azure AI Language
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
\nHow to use Named Entity Recognition
(NER)
Article • 02/21/2025
The NER feature can evaluate unstructured text, and extract named entities from text in
several predefined categories, for example: person, location, event, product, and
organization.
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
When you submit input text to be processed, you can specify which of the supported
languages they're written in. if you don't specify a language, key phrase extraction
defaults to English. The API may return offsets in the response to support different
multilingual and emoji encodings.
Development options
ﾉ
Expand table
Determine how to process the data (optional)
Input languages
Submitting data
\nAnalysis is performed upon receipt of the request. Using the NER feature synchronously
is stateless. No data is stored in your account, and results are returned immediately in
the response.
When using this feature asynchronously, the API results are available for 24 hours from
the time the request was ingested, and is indicated in the response. After this time
period, the results are purged and are no longer available for retrieval.
The API attempts to detect the defined entity categories for a given input text language.
When you get results from NER, you can stream the results to an application or save the
output to a file on the local system. The API response includes recognized entities,
including their categories and subcategories, and confidence scores.
The API attempts to detect the defined entity types and tags for a given input text
language. The entity types and tags replace the categories and subcategories structure
the older models use to define entities for more flexibility. You can also specify which
entities are detected and returned, use the optional includeList  and excludeList
parameters with the appropriate entity types. The following example would detect only
Location . You can specify one or more entity types to be returned. Given the types and
tags hierarchy introduced for this version, you have the flexibility to filter on different
granularity levels as so:
Input:
Bash
Getting NER results
Select which entities to be returned
７ Note
In this example, it returns only the "Location" entity type.
{
    "kind": "EntityRecognition",
    "parameters": 
    {
        "includeList" :
        [
            "Location"
        ]
\nThe above examples would return entities falling under the Location  entity type such as
the GPE , Structural , and Geological  tagged entities as outlined by entity types and
tags. We could also further filter the returned entities by filtering using one of the entity
tags for the Location  entity type such as filtering over GPE  tag only as outlined:
Bash
This method returns all Location  entities only falling under the GPE  tag and ignore any
other entity falling under the Location  type that is tagged with any other entity tag such
as Structural  or Geological  tagged Location  entities. We could also further drill down
on our results by using the excludeList  parameter. GPE  tagged entities could be tagged
with the following tags: City , State , CountryRegion , Continent . We could, for example,
exclude Continent  and CountryRegion  tags for our example:
    },
    "analysisInput":
    {
        "documents":
        [
            {
                "id":"1",
                "language": "en",
                "text": "We went to Contoso foodplace located at downtown 
Seattle last week for a dinner party, and we adore the spot! They provide 
marvelous food and they have a great menu. The chief cook happens to be the 
owner (I think his name is John Doe) and he is super nice, coming out of the 
kitchen and greeted us all. We enjoyed very much dining in the place! The 
pasta I ordered was tender and juicy, and the place was impeccably clean. 
You can even pre-order from their online menu at www.contosofoodplace.com, 
call 112-555-0176 or send email to order@contosofoodplace.com! The only 
complaint I have is the food didn't come fast enough. Overall I highly 
recommend it!"
            }
        ]
    }
}
    "parameters": 
    {
        "includeList" :
        [
            "GPE"
        ]
    }