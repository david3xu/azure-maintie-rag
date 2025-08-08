Feedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Yes
No
\nLanguage support for document and
conversation summarization
06/21/2025
Use this article to learn which natural languages are supported by document and conversation
summarization.
Extractive text and document summarization support the following languages:
Language
Language code
Notes
Chinese-Simplified
zh-hans
zh  also accepted
English
en
French
fr
German
de
Hebrew
he
Italian
it
Japanese
ja
Korean
ko
Polish
pl
Portuguese
pt
pt-br  also accepted
Spanish
es
Abstractive text and document summarization support the following languages:
Language
Language code
Notes
Arabic
ar
Text and document summarization
ﾉ
Expand table
ﾉ
Expand table
\nLanguage
Language code
Notes
Chinese-Simplified
zh-hans
zh  also accepted
English
en
French
fr
German
de
Hebrew
he
Italian
it
Japanese
ja
Korean
ko
Polish
pl
Portuguese
pt
pt-br  also accepted
Spanish
es
Conversation summarization supports the following languages:
Language
Language code
Notes
Chinese-Simplified
zh-hans
zh  also accepted
Chinese-Traditional
zh-hant
English
en
French
fr
German
de
Hebrew
he
Italian
it
Japanese
ja
Korean
ko
Conversation summarization
ﾉ
Expand table
\nLanguage
Language code
Notes
Polish
pl
Portuguese
pt
pt-br  also accepted
Spanish
es
Dutch
nl
Swedish
sv
Danish
da
Finnish
fi
Russian
ru
Norwegian
no
Turkish
tr
Arabic
ar
Czech
cs
Hungarian
hu
Thai
th
Summarization overview
Next steps
\nHow to use conversation summarization
Article • 03/05/2025
Chapter title and narrative (general conversation) are designed to summarize a
conversation into chapter titles, and a summarization of the conversation's
contents. This summarization aspect works on conversations with any number of
parties.
Issues and resolutions (call center focused) are designed to summarize text chat
logs between customers and customer-service agents. This feature is capable of
providing both issues and resolutions present in these logs, which occur between
two parties.
Narrative is designed to summarize the narrative of a conversation.
Recap is designed to condense lengthy meetings or conversations into a concise
one-paragraph summary to provide a quick overview.
Follow-up tasks is designed to summarize action items and tasks that arise during
a meeting.
） Important
Our preview region, Sweden Central, showcases our latest and continually evolving
LLM fine tuning techniques based on GPT models. You are welcome to try them out
with a Language resource in the Sweden Central region.
Conversation summarization is only available using:
REST API
Python
C#
Conversation summarization aspects
\nThe AI models used by the API are provided by the service. You just have to send
content for analysis.
For easier navigation, here are links to the corresponding sections for each service:
Aspect
Section
Issue and Resolution
Issue and Resolution
Chapter Title
Chapter Title
Narrative
Narrative
Recap and Follow-up
Recap and follow-up
The conversation summarization API uses natural language processing techniques to
summarize conversations into shorter summaries per request. Conversation
summarization can summarize for issues and resolutions discussed in a two-party
conversation or summarize a long conversation into chapters and a short narrative for
each chapter.
There's another feature in Azure AI Language named text summarization that is more
suitable to summarize documents into concise summaries. When you're deciding
ﾉ
Expand table
Features
\nbetween text summarization and conversation summarization, consider the following
points:
Input format: Conversation summarization can operate on both chat text and
speech transcripts, which have speakers and their utterances. Text summarization
operates using simple text, or Word, PDF, or PowerPoint formats.
Purpose of summarization: for example, conversation issue and resolution
summarization  returns a reason and the resolution for a chat between a customer
and a customer service agent.
You submit documents to the API as strings of text. Analysis is performed upon receipt
of the request. Because the API is asynchronous, there might be a delay between
sending an API request and receiving the results. For information on the size and
number of requests you can send per minute and second, see the following data limits.
When you use this feature, the API results are available for 24 hours from the time the
request was ingested, and is indicated in the response. After this time period, the results
are purged and are no longer available for retrieval.
When you submit data to conversation summarization, we recommend sending one
chat log per request, for better latency.
You can use conversation issue and resolution summarization to get summaries as you
need. To see an example using text chats, see the quickstart article.
The conversation issue and resolution summarization  also enables you to get
summaries from speech transcripts by using the Speech service's speech to text feature.
The following example shows a short conversation that you might include in your API
requests.
JSON
Submitting data
Get summaries from text chats
Get summaries from speech transcriptions
"conversations":[
   {
      "id":"abcdefgh-1234-1234-1234-1234abcdefgh",
      "language":"en",
      "modality":"transcript",
\nConversation chapter title summarization lets you get chapter titles from input
conversations. A guided example scenario follows:
1. Copy the following command into a text editor. The BASH example uses the \  line
continuation character. If your console or terminal uses a different line
continuation character, use that character.
Bash
      "conversationItems":[
         {
            "modality":"transcript",
            "participantId":"speaker",
            "id":"12345678-abcd-efgh-1234-abcd123456",
            "content":{
               "text":"Hi.",
               "lexical":"hi",
               "itn":"hi",
               "maskedItn":"hi",
               "audioTimings":[
                  {
                     "word":"hi",
                     "offset":4500000,
                     "duration":2800000
                  }
               ]
            }
         }
      ]
   }
]
Get chapter titles
curl -i -X POST https://<your-language-resource-endpoint>/language/analyze-
conversations/jobs?api-version=2023-11-15-preview \
-H "Content-Type: application/json" \
-H "Ocp-Apim-Subscription-Key: <your-language-resource-key>" \
-d \
' 
{
  "displayName": "Conversation Task Example",
  "analysisInput": {
    "conversations": [
      {
        "conversationItems": [
          {
            "text": "Hello, you're chatting with Rene. How may I help you?",
            "id": "1",
            "role": "Agent",
            "participantId": "Agent_1"
\n          },
          {
            "text": "Hi, I tried to set up wifi connection for Smart Brew 
300 espresso machine, but it didn't work.",
            "id": "2",
            "role": "Customer",
            "participantId": "Customer_1"
          },
          {
            "text": "I'm sorry to hear that. Let's see what we can do to fix 
this issue. Could you please try the following steps for me? First, could 
you push the wifi connection button, hold for 3 seconds, then let me know if 
the power light is slowly blinking on and off every second?",
            "id": "3",
            "role": "Agent",
            "participantId": "Agent_1"
          },
          {
            "text": "Yes, I pushed the wifi connection button, and now the 
power light is slowly blinking.",
            "id": "4",
            "role": "Customer",
            "participantId": "Customer_1"
          },
          {
            "text": "Great. Thank you! Now, please check in your Contoso 
Coffee app. Does it prompt to ask you to connect with the machine? ",
            "id": "5",
            "role": "Agent",
            "participantId": "Agent_1"
          },
          {
            "text": "No. Nothing happened.",
            "id": "6",
            "role": "Customer",
            "participantId": "Customer_1"
          },
          {
            "text": "I'm very sorry to hear that. Let me see if there's 
another way to fix the issue. Please hold on for a minute.",
            "id": "7",
            "role": "Agent",
            "participantId": "Agent_1"
          }
        ],
        "modality": "text",
        "id": "conversation1",
        "language": "en"
      }
    ]
  },
  "tasks": [
    {
      "taskName": "Conversation Task 1",
      "kind": "ConversationalSummarizationTask",
\n2. Make the following changes in the command where needed:
Replace the value your-value-language-key  with your key.
Replace the first part of the request URL your-language-resource-endpoint
with your endpoint URL.
3. Open a command prompt window (for example: BASH).
4. Paste the command from the text editor into the command prompt window, then
run the command.
5. Get the operation-location  from the response header. The value looks similar to
the following URL:
HTTP
6. To get the results of the request, use the following cURL command. Be sure to
replace <my-job-id>  with the GUID value you received from the previous
operation-location  response header:
curl
Example chapter title summarization JSON response:
JSON
      "parameters": {
        "summaryAspects": [
          "chapterTitle"
        ]
      }
    }
  ]
}
'
https://<your-language-resource-endpoint>/language/analyze-
conversations/jobs/12345678-1234-1234-1234-12345678?api-version=2023-11-15-
preview
curl -X GET https://<your-language-resource-endpoint>/language/analyze-
conversations/jobs/<my-job-id>?api-version=2023-11-15-preview \
-H "Content-Type: application/json" \
-H "Ocp-Apim-Subscription-Key: <your-language-resource-key>"
{
    "jobId": "b01af3b7-1870-460a-9e36-09af28d360a1",