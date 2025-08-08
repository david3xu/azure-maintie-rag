Sentiment Analysis and Named Entity Recognition endpoints. The LoggingOptOut
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
\nHow to use language detection
06/04/2025
The Language Detection feature can evaluate text, and return a language identifier that
indicates the language a document was written in.
Language detection is useful for content stores that collect arbitrary text, where language is
unknown. You can parse the results of this analysis to determine which language is used in the
input document. The response also returns a score between 0 and 1 that reflects the
confidence of the model.
The Language Detection feature can detect a wide range of languages, variants, dialects, and
some regional or cultural languages.
To use language detection, you submit raw unstructured text for analysis and handle the API
output in your application. Analysis is performed as-is, with no additional customization to the
model used on your data. There are three ways to use language detection:
Development
option
Description
Azure AI Foundry
Azure AI Foundry is a web-based platform that lets you use entity linking with
text examples with your own data when you sign up. For more information, see
the Azure AI Foundry website
 or Azure AI Foundry documentation.
REST API or Client
library (Azure SDK)
Integrate language detection into your applications using the REST API, or the
client library available in a variety of languages. For more information, see the
language detection quickstart.
Docker container
Use the available Docker container to deploy this feature on-premises. These
docker containers enable you to bring the service closer to your data for
compliance, security, or other operational reasons.
Development options
ﾉ
Expand table
Determine how to process the data (optional)
Specify the language detection model
\nBy default, language detection will use the latest available AI model on your text. You can also
configure your API requests to use a specific model version.
When you submit documents to be evaluated, language detection will attempt to determine if
the text was written in any of the supported languages.
If you have content expressed in a less frequently used language, you can try the Language
Detection feature to see if it returns a code. The response for languages that can't be detected
is unknown .
Analysis is performed upon receipt of the request. Using the language detection feature
synchronously is stateless. No data is stored in your account, and results are returned
immediately in the response.
When using this feature asynchronously, the API results are available for 24 hours from the
time the request was ingested, and is indicated in the response. After this time period, the
results are purged and are no longer available for retrieval.
When you get results from language detection, you can stream the results to an application or
save the output to a file on the local system.
Language detection will return one predominant language for each document you submit,
along with it's ISO 639-1
 name, a human-readable name, a confidence score, script name,
and script code according to the ISO 15924 standard
. A positive score of 1 indicates the
highest possible confidence level of the analysis.
Input languages
Submitting data
 Tip
You can use a Docker containerfor language detection, so you can use the API on-
premises.
Getting language detection results
Ambiguous content
\nIn some cases it may be hard to disambiguate languages based on the input. You can use the
countryHint  parameter to specify an ISO 3166-1 alpha-2
 country/region code. By default the
API uses "US" as the default country hint. To remove this behavior, you can reset this parameter
by setting this value to empty string countryHint = ""  .
For example, "communication" is common to both English and French and if given with limited
context the response will be based on the "US" country/region hint. If the origin of the text is
known to be coming from France that can be given as a hint.
Input
JSON
With the second document, the language detection model has additional context to make a
better judgment because it contains the countryHint  property in the input above. This will
return the following output.
Output
JSON
７ Note
Ambiguous content can cause confidence scores to be lower. The countryHint  in the
response is only applicable if the confidence score is less than 0.8.
{
    "documents": [
        {
            "id": "1",
            "text": "communication"
        },
        {
            "id": "2",
            "text": "communication",
            "countryHint": "fr"
        }
    ]
}
{
    "documents":[
        {
            "detectedLanguage":{
                "confidenceScore":0.62,
                "iso6391Name":"en",
                "name":"English"
\nIf the analyzer can't parse the input, it returns (Unknown) . An example is if you submit a text
string that consists solely of numbers.
JSON
Mixed-language content within the same document returns the language with the largest
representation in the content, but with a lower positive rating. The rating reflects the marginal
strength of the assessment. In the following example, input is a blend of English, Spanish, and
            },
            "id":"1",
            "warnings":[
                
            ]
        },
        {
            "detectedLanguage":{
                "confidenceScore":1.0,
                "iso6391Name":"fr",
                "name":"French"
            },
            "id":"2",
            "warnings":[
                
            ]
        }
    ],
    "errors":[
        
    ],
    "modelVersion":"2022-10-01"
}
{
    "documents": [
        {
            "id": "1",
            "detectedLanguage": {
                "name": "(Unknown)",
                "iso6391Name": "(Unknown)",
                "confidenceScore": 0.0
            },
            "warnings": []
        }
    ],
    "errors": [],
    "modelVersion": "2023-12-01"
}
Mixed-language content
\nFrench. The analyzer counts characters in each segment to determine the predominant
language.
Input
JSON
Output
The resulting output consists of the predominant language, with a score of less than 1.0, which
indicates a weaker level of confidence.
JSON
{
    "documents": [
        {
            "id": "1",
            "text": "Hello, I would like to take a class at your University. ¿Se 
ofrecen clases en español? Es mi primera lengua y más fácil para escribir. Que 
diriez-vous des cours en français?"
        }
    ]
}
{
    "kind": "LanguageDetectionResults",
    "results": {
        "documents": [
            {
                "id": "1",
                "detectedLanguage": {
                    "name": "Spanish",
                    "iso6391Name": "es",
                    "confidenceScore": 0.97,
                    "script": "Latin",
                    "scriptCode": "Latn"
                },
                "warnings": []
            }
        ],
        "errors": [],
        "modelVersion": "2023-12-01"
    }
}
Script name and script code
\nLanguage detection offers the ability to detect more than one script per language according to
the ISO 15924 standard
. Specifically, Language Detection returns two script-related
properties:
script : The human-readable name of the identified script
scriptCode : The ISO 15924 code for the identified script
The output of the API includes the value of the scriptCode  property for documents that are at
least 12 characters or greater in length and matches the list of supported languages and
scripts. Script detection is designed to benefit users whose language can be transliterated or
written in more than one script, such as Kazakh or Hindi language.
Previously, language detection was designed to detect the language of documents in a wide
variety of languages, dialects, and regional variants, but was limited by "Romanization".
Romanization refers to conversion of text from one writing system to the Roman (Latin) script,
and is necessary to detect many Indo-European languages. However, there are other languages
which are written in multiple scripts, such as Kazakh, which can be written in Cyrillic, Perso-
Arabic, and Latin scripts. There are also other cases in which users may either choose or are
required to transliterate their language in more than one script, such as Hindi transliterated in
Latin script, due to the limited availability of keyboards which support its Devanagari script.
Consequently, language detection's expanded support for script detection behaves as follows:
Input
JSON
７ Note
Script detection is currently limited to select languages.
The script detection is only available for textual input which is greater than 12
characters in length.
{ 
    "kind": "LanguageDetection", 
    "parameters": { 
        "modelVersion": "latest" 
    }, 
    "analysisInput": { 
        "documents": [ 
            { 
                "id": "1", 
                "text": "आप कहाँ जा रहे हैं?" 
            }, 
            { 
\nOutput
The resulting output consists of the predominant language, along with a script name, script
code, and confidence score.
JSON
For information on the size and number of requests you can send per minute and second, see
the service limits article.
                "id": "2", 
                "text": "Туған жерім менің - Қазақстаным" 
            } 
        ] 
    } 
} 
{ 
    "kind": "LanguageDetectionResults", 
    "results": { 
        "documents": [ 
            { 
                "id": "1", 
                "detectedLanguage": { 
                    "name": "Hindi", 
                    "iso6391Name": "hi", 
                    "confidenceScore": 1.0, 
                    "script": "Devanagari", 
                    "scriptCode": "Deva" 
                }, 
                "warnings": [] 
            }, 
            { 
                "id": "2", 
                "detectedLanguage": { 
                    "name": "Kazakh", 
                    "iso6391Name": "kk", 
                    "confidenceScore": 1.0, 
                    "script": "Cyrillic",  
                    "scriptCode": "Cyrl" 
                }, 
                "warnings": [] 
            } 
        ], 
        "errors": [], 
        "modelVersion": "2023-12-01" 
    } 
}
Service and data limits
\nLanguage detection overview
See also
\nUse language detection Docker containers
on-premises
06/21/2025
Containers enable you to host the Language Detection API on your own infrastructure. If you
have security or data governance requirements that can't be fulfilled by calling Language
Detection remotely, then containers might be a good option.
If you don't have an Azure subscription, create a free account
.
Docker
 installed on a host computer. Docker must be configured to allow the
containers to connect with and send billing data to Azure.
On Windows, Docker must also be configured to support Linux containers.
You should have a basic understanding of Docker concepts
.
A Language resource 
with the free (F0) or standard (S) pricing tier
.
Three primary parameters for all Azure AI containers are required. The Microsoft Software
License Terms must be present with a value of accept. An Endpoint URI and API key are also
needed.
The {ENDPOINT_URI}  value is available on the Azure portal Overview page of the corresponding
Azure AI services resource. Go to the Overview page, hover over the endpoint, and a Copy to
clipboard ＝ icon appears. Copy and use the endpoint where needed.
Prerequisites
Gather required parameters
Endpoint URI
\n![Image](images/page420_image1.png)