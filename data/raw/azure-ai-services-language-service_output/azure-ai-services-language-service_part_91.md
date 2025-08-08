Make sure you understand all the entity categories that can be recognized by the system.
Depending on your scenario, your data may include other information that could be
considered personal but is not covered by the categories the service currently supports.
Context is important for all entity categories to be correctly recognized by the system, as
it often is for humans to recognize an entity. For example, without context a ten-digit
number is just a number, not a PII entity. However, given context like You can reach me at
my office number 2345678901, both the system and a human can recognize the ten-digit
number as a phone number. Always include context when sending text to the system to
obtain the best possible performance.
Person names in particular require linguistic context. Send as much context as possible for
better person name detection.
For conversational data, consider sending more than a single turn in the conversation to
ensure higher likelihood that the required context is included with the actual entities.
In the following conversation, if you send a single row at a time, the passport number will
not have any context associated with it and the EU Passport Number PII category will not
be recognized.
Hi, how can I help you today?
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
\nsee multiple entity categories for recognized for your scenario.
Although many international entities are supported, currently the service only supports
English text. Consider verifying the language the input text is in if you're not sure it will be
all in English.
The PII service only takes text as an input. If you are redacting information from
documents in other formats, make sure to carefully test your redaction code to ensure
identified entities are not accidentally leaked.
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for health feature
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
\nDetect and redact Personally Identifying
Information in text
05/19/2025
Azure AI Language is a cloud-based service that applies Natural Language Processing (NLP)
features to text-based data. The PII feature can evaluate unstructured text, extract, and redact
sensitive information (PII) and health information (PHI) in text across several predefined
categories.
To use PII detection, you submit text for analysis and handle the API output in your application.
Analysis is performed as-is, with no customization to the model used on your data. There are
two ways to use PII detection:
Development
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use personally identifying
information detection with text examples with your own data when you sign up.
For more information, see the Azure AI Foundry website
 or Azure AI Foundry
documentation.
REST API or Client
library (Azure SDK)
Integrate PII detection into your applications using the REST API, or the client
library available in various languages. For more information, see the PII detection
quickstart.
By default, this feature uses the latest available AI model on your text. You can also configure
your API requests to use a specific model version.
When you submit input text to be processed, you can specify which of the supported
languages they're written in. If you don't specify a language, extraction defaults to English. The
API may return offsets in the response to support different multilingual and emoji encodings.
Development options
ﾉ
Expand table
Specify the PII detection model
Input languages
\nIn version 2024-11-5-preview , you're able to define the redactionPolicy  parameter to reflect
the redaction policy to be used when redacting text. The policy field supports three policy
types:
DoNotRedact
MaskWithCharacter  (default)
MaskWithEntityType
The DoNotRedact  policy allows the user to return the response without the redactedText  field,
that is, "John Doe received a call from 424-878-9192".
The MaskWithRedactionCharacter  policy allows the redactedText  to be masked with a character
(such as "*"), preserving the length and offset of the original text, that is, "******** received a
call from ************". This is the existing behavior.
There's also an optional field called redactionCharacter  where you can input the character to
be used in redaction if you're using the MaskWithCharacter  policy
The MaskWithEntityType  policy allows you to mask the detected PII entity text with the
detected entity type, that is, "[PERSON_1] received a call from [PHONENUMBER_1]".
The API attempts to detect the defined entity categories for a given input text language. If you
want to specify which entities are detected and returned, use the optional piiCategories
parameter with the appropriate entity categories. This parameter can also let you detect
entities that aren't enabled by default for your input text language. The following example
would detect only Person . You can specify one or more entity types to be returned.
Input:
Redaction Policy (version 2024-11-5-preview only)
Select which entities to be returned
 Tip
If you don't include default  when specifying entity categories, The API only returns the
entity categories you specify.
７ Note
\nhttps://<your-language-resource-endpoint>/language/:analyze-text?api-version=2022-05-01
Bash
Output:
Bash
In this example, it returns only the person entity type:
{
    "kind": "PiiEntityRecognition",
    "parameters": 
    {
        "modelVersion": "latest",
        "piiCategories" :
        [
            "Person"
        ]
    },
    "analysisInput":
    {
        "documents":
        [
            {
                "id":"1",
                "language": "en",
                "text": "We went to Contoso foodplace located at downtown Seattle 
last week for a dinner party, and we adore the spot! They provide marvelous food 
and they have a great menu. The chief cook happens to be the owner (I think his 
name is John Doe) and he is super nice, coming out of the kitchen and greeted us 
all. We enjoyed very much dining in the place! The pasta I ordered was tender and 
juicy, and the place was impeccably clean. You can even pre-order from their 
online menu at www.contosofoodplace.com, call 112-555-0176 or send email to 
order@contosofoodplace.com! The only complaint I have is the food didn't come fast 
enough. Overall I highly recommend it!"
            }
        ]
    },
    "kind": "PiiEntityRecognition", 
    "parameters": { 
        "redactionPolicy": { 
            "policyKind": "MaskWithCharacter"  
             //MaskWithCharacter|MaskWithEntityType|DoNotRedact 
            "redactionCharacter": "*"  
}
{
    "kind": "PiiEntityRecognitionResults",
\nTo accommodate and adapt to a customer’s custom vocabulary used to identify entities (also
known as the “context”), the entitySynonyms  feature allows customers to define their own
synonyms for specific entity types. The goal of this feature is to help detect entities in contexts
that the model is not familiar with but are used in the customer’s inputs by ensuring that the
customer’s unique terms are recognized and correctly associated during the detection process.
The valueExclusionPolicy  option allows customers to adapt the PII service for scenarios where
customers prefer certain terms not to be detected and redacted even if those terms fall into a
PII category they are interested in detected. For example, a police department might want
personal identifiers redacted in most cases except for terms like “police officer”, “suspect”, and
“witness”.
Customers can now adapt the PII service’s detecting by specifying their own regex using a
regex recognition configuration file. See our container how-to guides for a tutorial on how to
install and run Personally Identifiable Information (PII) Detection containers.
    "results": {
        "documents": [
            {
                "redactedText": "We went to Contoso foodplace located at downtown 
Seattle last week for a dinner party, and we adore the spot! They provide 
marvelous food and they have a great menu. The chief cook happens to be the owner 
(I think his name is ********) and he is super nice, coming out of the kitchen and 
greeted us all. We enjoyed very much dining in the place! The pasta I ordered was 
tender and juicy, and the place was impeccably clean. You can even pre-order from 
their online menu at www.contosofoodplace.com, call 112-555-0176 or send email to 
order@contosofoodplace.com! The only complaint I have is the food didn't come fast 
enough. Overall I highly recommend it!",
                "id": "1",
                "entities": [
                    {
                        "text": "John Doe",
                        "category": "Person",
                        "offset": 226,
                        "length": 8,
                        "confidenceScore": 0.98
                    }
                ],
                "warnings": []
            }
        ],
        "errors": [],
        "modelVersion": "2021-01-15"
    }
}
Adapting PII to your domain