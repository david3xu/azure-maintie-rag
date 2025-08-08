for customer calls, how they were resolved, extracting and redacting conversation PII, and
more.
The Speech service offers the following features that can be used for call center use cases:
Real-time speech to text: Recognize and transcribe audio in real-time from multiple
inputs. For example, with virtual agents or agent-assist, you can continuously recognize
audio input and control how to process results based on multiple events.
Batch speech to text: Transcribe large amounts of audio files asynchronously including
speaker diarization and is typically used in post-call analytics scenarios. Diarization is the
process of recognizing and separating speakers in mono channel audio data.
Text to speech: Text to speech enables your applications, tools, or devices to convert text
into human like synthesized speech.
Speaker identification: Helps you determine an unknown speaker’s identity within a group
of enrolled speakers and is typically used for call center customer verification scenarios or
fraud detection.
Language Identification: Identify languages spoken in audio and can be used in real-time
and post-call analysis for insights or to control the environment (such as output language
of a virtual agent).
You might want to further customize and fine-tune the experience for your product or
environment. Typical examples for Speech fine-tuning include:
Speech
customization
Description
Custom speech
A speech to text feature used to evaluate and improve the speech recognition
accuracy of use-case specific entities (such as alpha-numeric customer, case, and
contract IDs, license plates, and names). You can also train a custom model with your
own product names and industry terminology.
Custom voice
A text to speech feature that lets you create a one-of-a-kind, customized, synthetic
voice for your applications.
The Language service offers the following features that can be used for call center use cases:
Speech service
ﾉ
Expand table
Language service
\nPersonally Identifiable Information (PII) extraction and redaction: Identify, categorize, and
redact sensitive information in conversation transcription.
Conversation summarization: Summarize in abstract text what each conversation
participant said about the issues and resolutions. For example, a call center can group
product issues that have a high volume.
Sentiment analysis and opinion mining: Analyze transcriptions and associate positive,
neutral, or negative sentiment at the utterance and conversation-level.
You might want to further customize and fine-tune models to extract more information from
your data. Typical examples for Language customization include:
Language customization
Description
Custom NER (named entity
recognition)
Improve the detection and extraction of entities in transcriptions.
Custom text classification
Classify and label transcribed utterances with either single or
multiple classifications.
You can find an overview of all Language service features and customization options here.
Post-call transcription and analytics quickstart
Try out the Language Studio
Try out the Speech Studio
ﾉ
Expand table
Next steps
\nQuickstart: Post-call transcription and
analytics
Article • 03/10/2025
Language service documentation | Language Studio
 | Speech service documentation |
Speech Studio
In this C# quickstart, you perform sentiment analysis and conversation summarization of
call center transcriptions. The sample will automatically identify, categorize, and redact
sensitive information. The quickstart implements a cross-service scenario that uses
features of the Azure Cognitive Speech and Azure Cognitive Language services.
The following Azure AI services for Speech features are used in the quickstart:
Batch transcription: Submit a batch of audio files for transcription.
Speaker separation: Separate multiple speakers through diarization of mono 16khz
16 bit PCM wav files.
The Language service offers the following features that are used:
Personally Identifiable Information (PII) extraction and redaction: Identify,
categorize, and redact sensitive information in conversation transcription.
Conversation summarization: Summarize in abstract text what each conversation
participant said about the issues and resolutions. For example, a call center can
group product issues that have a high volume.
Sentiment analysis and opinion mining: Analyze transcriptions and associate
positive, neutral, or negative sentiment at the utterance and conversation-level.
 Tip
Try the Language Studio
 or Speech Studio
 for a demonstration on how to use
the Language and Speech services to analyze call center conversations.
To deploy a call center transcription solution to Azure with a no-code approach, try
the Ingestion Client.
Prerequisites
Azure subscription - Create one for free
＂
\nFollow these steps to build and run the post-call transcription analysis quickstart code
example.
1. Copy the scenarios/csharp/dotnetcore/call-center/
 sample files from GitHub. If
you have Git installed
, open a command prompt and run the git clone
command to download the Speech SDK samples repository.
.NET CLI
2. Open a command prompt and change to the project directory.
.NET CLI
3. Build the project with the .NET CLI.
.NET CLI
Create a multi-service resource
 in the Azure portal. This quickstart only requires
one Azure AI services multi-service resource. The sample code allows you to specify
separate Language and Speech resource keys.
＂
Get the resource key and region. After your Azure AI services resource is deployed,
select Go to resource to view and manage keys.
＂
） Important
This quickstart requires access to conversation summarization. To get access, you
must submit an online request
 and have it approved.
The --languageKey  and --languageEndpoint  values in this quickstart must
correspond to a resource that's in one of the regions supported by the
conversation summarization API
: eastus , northeurope , and uksouth .
Run post-call transcription analysis with C#
git clone https://github.com/Azure-Samples/cognitive-services-speech-
sdk.git
cd <your-local-path>/scenarios/csharp/dotnetcore/call-center/call-
center/
dotnet build
\n4. Run the application with your preferred command line arguments. See usage and
arguments for the available options.
Here's an example that transcribes from an example audio file at GitHub
:
.NET CLI
If you already have a transcription for input, here's an example that only requires a
Language resource:
.NET CLI
Replace YourResourceKey  with your Azure AI services resource key, replace
YourResourceRegion  with your Azure AI services resource region (such as eastus ),
and replace YourResourceEndpoint  with your Azure AI services endpoint. Make sure
that the paths specified by --input  and --output  are valid. Otherwise you must
change the paths.
The console output shows the full conversation and summary. Here's an example of the
overall summary, with redactions for brevity:
Output
dotnet run --languageKey YourResourceKey --languageEndpoint 
YourResourceEndpoint --speechKey YourResourceKey --speechRegion 
YourResourceRegion --input "https://github.com/Azure-Samples/cognitive-
services-speech-sdk/raw/master/scenarios/call-
center/sampledata/Call1_separated_16k_health_insurance.wav" --stereo  -
-output summary.json
dotnet run --languageKey YourResourceKey --languageEndpoint 
YourResourceEndpoint --jsonInput "YourTranscriptionFile.json" --stereo  
--output summary.json
） Important
Remember to remove the key from your code when you're done, and never
post it publicly. For production, use a secure way of storing and accessing
your credentials like Azure Key Vault. See the Azure AI services security article
for more information.
Check results
\nIf you specify the --output FILE  optional argument, a JSON version of the results are
written to the file. The file output is a combination of the JSON responses from the
batch transcription (Speech), sentiment (Language), and conversation summarization
(Language) APIs.
The transcription  property contains a JSON object with the results of sentiment
analysis merged with batch transcription. Here's an example, with redactions for brevity:
JSON
The conversationAnalyticsResults  property contains a JSON object with the results of
the conversation PII and conversation summarization analysis. Here's an example, with
redactions for brevity:
JSON
Conversation summary:
    issue: Customer wants to sign up for insurance.
    resolution: Customer was advised that customer would be contacted by the 
insurance company.
{
    "source": "https://github.com/Azure-Samples/cognitive-services-speech-
sdk/raw/master/scenarios/call-
center/sampledata/Call1_separated_16k_health_insurance.wav",
// Example results redacted for brevity
        "nBest": [
          {
            "confidence": 0.77464247,
            "lexical": "hello thank you for calling contoso who am i 
speaking with today",
            "itn": "hello thank you for calling contoso who am i speaking 
with today",
            "maskedITN": "hello thank you for calling contoso who am i 
speaking with today",
            "display": "Hello, thank you for calling Contoso. Who am I 
speaking with today?",
            "sentiment": {
              "positive": 0.78,
              "neutral": 0.21,
              "negative": 0.01
            }
          },
        ]
// Example results redacted for brevity
}   
\n{
  "conversationAnalyticsResults": {
    "conversationSummaryResults": {
      "conversations": [
        {
          "id": "conversation1",
          "summaries": [
            {
              "aspect": "issue",
              "text": "Customer wants to sign up for insurance"
            },
            {
              "aspect": "resolution",
              "text": "Customer was advised that customer would be contacted 
by the insurance company"
            }
          ],
          "warnings": []
        }
      ],
      "errors": [],
      "modelVersion": "2022-05-15-preview"
    },
    "conversationPiiResults": {
      "combinedRedactedContent": [
        {
          "channel": "0",
          "display": "Hello, thank you for calling Contoso. Who am I 
speaking with today? Hi, ****. Uh, are you calling because you need health 
insurance?", // Example results redacted for brevity
          "itn": "hello thank you for calling contoso who am i speaking with 
today hi **** uh are you calling because you need health insurance", // 
Example results redacted for brevity
          "lexical": "hello thank you for calling contoso who am i speaking 
with today hi **** uh are you calling because you need health insurance" // 
Example results redacted for brevity
        },
        {
          "channel": "1",
          "display": "Hi, my name is **********. I'm trying to enroll myself 
with Contoso. Yes. Yeah, I'm calling to sign up for insurance.", // Example 
results redacted for brevity
          "itn": "hi my name is ********** i'm trying to enroll myself with 
contoso yes yeah i'm calling to sign up for insurance", // Example results 
redacted for brevity
          "lexical": "hi my name is ********** i'm trying to enroll myself 
with contoso yes yeah i'm calling to sign up for insurance" // Example 
results redacted for brevity
        }
      ],
      "conversations": [
        {
          "id": "conversation1",
          "conversationItems": [
\n            {
              "id": "0",
              "redactedContent": {
                "itn": "hello thank you for calling contoso who am i 
speaking with today",
                "lexical": "hello thank you for calling contoso who am i 
speaking with today",
                "text": "Hello, thank you for calling Contoso. Who am I 
speaking with today?"
              },
              "entities": [],
              "channel": "0",
              "offset": "PT0.77S"
            },
            {
              "id": "1",
              "redactedContent": {
                "itn": "hi my name is ********** i'm trying to enroll myself 
with contoso",
                "lexical": "hi my name is ********** i'm trying to enroll 
myself with contoso",
                "text": "Hi, my name is **********. I'm trying to enroll 
myself with Contoso."
              },
              "entities": [
                {
                  "text": "Mary Rondo",
                  "category": "Name",
                  "offset": 15,
                  "length": 10,
                  "confidenceScore": 0.97
                }
              ],
              "channel": "1",
              "offset": "PT4.55S"
            },
            {
              "id": "2",
              "redactedContent": {
                "itn": "hi **** uh are you calling because you need health 
insurance",
                "lexical": "hi **** uh are you calling because you need 
health insurance",
                "text": "Hi, ****. Uh, are you calling because you need 
health insurance?"
              },
              "entities": [
                {
                  "text": "Mary",
                  "category": "Name",
                  "offset": 4,
                  "length": 4,
                  "confidenceScore": 0.93
                }
              ],
\nUsage: call-center -- [...]
Connection options include:
--speechKey KEY : Your Azure AI services
 or Speech
 resource key. Required for
audio transcriptions with the --input  from URL option.
--speechRegion REGION : Your Azure AI services
 or Speech
 resource region.
Required for audio transcriptions with the --input  from URL option. Examples:
eastus , northeurope
--languageKey KEY : Your Azure AI services
 or Language
 resource key.
Required.
              "channel": "0",
              "offset": "PT9.55S"
            },
            {
              "id": "3",
              "redactedContent": {
                "itn": "yes yeah i'm calling to sign up for insurance",
                "lexical": "yes yeah i'm calling to sign up for insurance",
                "text": "Yes. Yeah, I'm calling to sign up for insurance."
              },
              "entities": [],
              "channel": "1",
              "offset": "PT13.09S"
            },
// Example results redacted for brevity
          ],
          "warnings": []
        }
      ]
    }
  }
}
Usage and arguments
） Important
You can use a multi-service
 resource or separate Language
 and Speech
resources. In either case, the --languageKey  and --languageEndpoint  values must
correspond to a resource that's in one of the regions supported by the
conversation summarization API
: eastus , northeurope , and uksouth .
\n--languageEndpoint ENDPOINT : Your Azure AI services
 or Language
 resource
endpoint. Required. Example:
https://YourResourceName.cognitiveservices.azure.com
Input options include:
--input URL : Input audio from URL. You must set either the --input  or --
jsonInput  option.
--jsonInput FILE : Input an existing batch transcription JSON result from FILE. With
this option, you only need a Language resource to process a transcription that you
already have. With this option, you don't need an audio file or an AI Services
resource for Speech. Overrides --input . You must set either the --input  or --
jsonInput  option.
--stereo : Indicates that the audio via ```input URL` should be in stereo format. If
stereo isn't specified, then mono 16khz 16 bit PCM wav files are assumed.
Diarization of mono files is used to separate multiple speakers. Diarization of
stereo files isn't supported, since 2-channel stereo files should already have one
speaker per channel.
--certificate : The PEM certificate file. Required for C++.
Language options include:
--language LANGUAGE : The language to use for sentiment analysis and conversation
analysis. This value should be a two-letter ISO 639-1 code. The default value is en .
--locale LOCALE : The locale to use for batch transcription of audio. The default
value is en-US .
Output options include:
--help : Show the usage help and stop
--output FILE : Output the transcription, sentiment, conversation PII, and
conversation summaries in JSON format to a text file. For more information, see
output examples.
You can use the Azure portal or Azure Command Line Interface (CLI) to remove the
Azure AI services resource you created.
Clean up resources
Next steps