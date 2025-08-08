"lastUpdatedDateTime": "2023-11-15T18:24:26Z",
    "createdDateTime": "2023-11-15T18:24:23Z",
    "expirationDateTime": "2023-11-16T18:24:23Z",
    "status": "succeeded",
    "errors": [],
    "displayName": "Conversation Task Example",
    "tasks": {
        "completed": 1,
        "failed": 0,
        "inProgress": 0,
        "total": 1,
        "items": [
            {
                "kind": "conversationalSummarizationResults",
                "taskName": "Conversation Task 1",
                "lastUpdateDateTime": "2023-11-15T18:24:26.3433677Z",
                "status": "succeeded",
                "results": {
                    "conversations": [
                        {
                            "summaries": [
                                {
                                    "aspect": "chapterTitle",
                                    "text": "\"Discussing the Problem of 
Smart Blend 300 Espresso Machine's Wi-Fi Connectivity\"",
                                    "contexts": [
                                        {
                                            "conversationItemId": "1",
                                            "offset": 0,
                                            "length": 53
                                        },
                                        {
                                            "conversationItemId": "2",
                                            "offset": 0,
                                            "length": 94
                                        },
                                        {
                                            "conversationItemId": "3",
                                            "offset": 0,
                                            "length": 266
                                        },
                                        {
                                            "conversationItemId": "4",
                                            "offset": 0,
                                            "length": 85
                                        },
                                        {
                                            "conversationItemId": "5",
                                            "offset": 0,
                                            "length": 119
                                        },
                                        {
                                            "conversationItemId": "6",
                                            "offset": 0,
                                            "length": 21
\nFor long conversation, the model might segment it into multiple cohesive parts, and
summarize each segment. There's also a lengthy contexts  field for each summary,
which tells from which range of the input conversation we generated the summary.
Conversation summarization also lets you get narrative summaries from input
conversations. A guided example scenario is provided:
1. Copy the following command into a text editor. The BASH example uses the \  line
continuation character. If your console or terminal uses a different line
continuation character, use that character.
Bash
                                        },
                                        {
                                            "conversationItemId": "7",
                                            "offset": 0,
                                            "length": 109
                                        }
                                    ]
                                }
                            ],
                            "id": "conversation1",
                            "warnings": []
                        }
                    ],
                    "errors": [],
                    "modelVersion": "latest"
                }
            }
        ]
    }
}
Get narrative summarization
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
\n            "id": "1",
            "role": "Agent",
            "participantId": "Agent_1"
          },
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
\n2. Make the following changes in the command where needed:
Replace the value your-language-resource-key  with your key.
Replace the first part of the request URL your-language-resource-endpoint
with your endpoint URL.
3. Open a command prompt window (for example: BASH).
4. Paste the command from the text editor into the command prompt window, then
run the command.
5. Get the operation-location  from the response header. The value looks similar to
the following URL:
HTTP
6. To get the results of a request, use the following cURL command. Be sure to
replace <my-job-id>  with the GUID value you received from the previous
operation-location  response header:
curl
Example narrative summarization JSON response:
JSON
    {
      "taskName": "Conversation Task 1",
      "kind": "ConversationalSummarizationTask",
      "parameters": {
        "summaryAspects": [
          "narrative"
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
\n{
  "jobId": "d874a98c-bf31-4ac5-8b94-5c236f786754",
  "lastUpdatedDateTime": "2022-09-29T17:36:42Z",
  "createdDateTime": "2022-09-29T17:36:39Z",
  "expirationDateTime": "2022-09-30T17:36:39Z",
  "status": "succeeded",
  "errors": [],
  "displayName": "Conversation Task Example",
  "tasks": {
    "completed": 1,
    "failed": 0,
    "inProgress": 0,
    "total": 1,
    "items": [
      {
        "kind": "conversationalSummarizationResults",
        "taskName": "Conversation Task 1",
        "lastUpdateDateTime": "2022-09-29T17:36:42.895694Z",
        "status": "succeeded",
        "results": {
          "conversations": [
            {
              "summaries": [
                {
                  "aspect": "narrative",
                  "text": "Agent_1 helps customer to set up wifi connection 
for Smart Brew 300 espresso machine.",
                  "contexts": [
                    { "conversationItemId": "1", "offset": 0, "length": 53 
},
                    { "conversationItemId": "2", "offset": 0, "length": 94 
},
                    { "conversationItemId": "3", "offset": 0, "length": 266 
},
                    { "conversationItemId": "4", "offset": 0, "length": 85 
},
                    { "conversationItemId": "5", "offset": 0, "length": 119 
},
                    { "conversationItemId": "6", "offset": 0, "length": 21 
},
                    { "conversationItemId": "7", "offset": 0, "length": 109 
}
                  ]
                }
              ],
              "id": "conversation1",
              "warnings": []
            }
          ],
          "errors": [],
          "modelVersion": "latest"
        }
      }
    ]
\nFor long conversation, the model might segment it into multiple cohesive parts, and
summarize each segment. There's also a lengthy contexts  field for each summary,
which tells from which range of the input conversation we generated the summary.
Conversation summarization also lets you get recaps and follow-up tasks from input
conversations. A guided example scenario is provided:
1. Copy the following command into a text editor. The BASH example uses the \  line
continuation character. If your console or terminal uses a different line
continuation character, use that character.
Bash
  }
}
Get recap and follow-up task summarization
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
          },
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
\n2. Make the following changes in the command where needed:
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
      "parameters": {
        "summaryAspects": [
          "recap",
          "follow-up tasks"
        ]
      }
    }
  ]
}
'
\nReplace the value your-language-resource-key  with your key.
Replace the first part of the request URL your-language-resource-endpoint
with your endpoint URL.
3. Open a command prompt window (for example: BASH).
4. Paste the command from the text editor into the command prompt window, then
run the command.
5. Get the operation-location  from the response header. The value looks similar to
the following URL:
HTTP
6. To get the results of a request, use the following cURL command. Be sure to
replace <my-job-id>  with the GUID value you received from the previous
operation-location  response header:
curl
Example recap and follow-up summarization JSON response:
JSON
https://<your-language-resource-endpoint>/language/analyze-
conversations/jobs/12345678-1234-1234-1234-12345678?api-version=2023-11-15-
preview
curl -X GET https://<your-language-resource-endpoint>/language/analyze-
conversations/jobs/<my-job-id>?api-version=2023-11-15-preview \
-H "Content-Type: application/json" \
-H "Ocp-Apim-Subscription-Key: <your-language-resource-key>"
{
    "jobId": "e585d097-c19a-466e-8f99-a9646e55b1f5",
    "lastUpdatedDateTime": "2023-11-15T18:19:56Z",
    "createdDateTime": "2023-11-15T18:19:53Z",
    "expirationDateTime": "2023-11-16T18:19:53Z",
    "status": "succeeded",
    "errors": [],
    "displayName": "Conversation Task Example",
    "tasks": {
        "completed": 1,
        "failed": 0,
        "inProgress": 0,
        "total": 1,
        "items": [
            {
\n                "kind": "conversationalSummarizationResults",
                "taskName": "Conversation Task 1",
                "lastUpdateDateTime": "2023-11-15T18:19:56.1801785Z",
                "status": "succeeded",
                "results": {
                    "conversations": [
                        {
                            "summaries": [
                                {
                                    "aspect": "recap",
                                    "text": "The customer contacted the 
service agent, Rene, regarding an issue with setting up a wifi connection 
for their Smart Brew 300 espresso machine. The agent guided the customer 
through several steps, including pushing the wifi connection button and 
checking if the power light was blinking. However, the customer reported 
that no prompts were received in the Contoso Coffee app to connect with the 
machine. The agent then decided to look for another solution.",
                                    "contexts": [
                                        {
                                            "conversationItemId": "1",
                                            "offset": 0,
                                            "length": 53
                                        },
                                        {
                                            "conversationItemId": "2",
                                            "offset": 0,
                                            "length": 94
                                        },
                                        {
                                            "conversationItemId": "3",
                                            "offset": 0,
                                            "length": 266
                                        },
                                        {
                                            "conversationItemId": "4",
                                            "offset": 0,
                                            "length": 85
                                        },
                                        {
                                            "conversationItemId": "5",
                                            "offset": 0,
                                            "length": 119
                                        },
                                        {
                                            "conversationItemId": "6",
                                            "offset": 0,
                                            "length": 21
                                        },
                                        {
                                            "conversationItemId": "7",
                                            "offset": 0,
                                            "length": 109
                                        }
                                    ]
                                },
\nFor long conversation, the model might segment it into multiple cohesive parts, and
summarize each segment. There's also a lengthy contexts  field for each summary,
which tells from which range of the input conversation we generated the summary.
The following text is an example of content you might submit for conversation issue and
resolution summarization. It's only an example. The API can accept longer input text. For
more information, see data limits.
Agent: "Hello, how can I help you?"
Customer: "How can I upgrade my Contoso subscription? I've been trying the entire day."
Agent: "Press the upgrade button then sign in and follow the instructions."
                                {
                                    "aspect": "Follow-Up Tasks",
                                    "text": "@Agent_1 will ask the customer 
to push the wifi connection button, hold for 3 seconds, then check if the 
power light is slowly blinking on and off every second."
                                },
                                {
                                    "aspect": "Follow-Up Tasks",
                                    "text": "@Agent_1 will ask the customer 
to check in the Contoso Coffee app if it prompts to connect with the 
machine."
                                },
                                {
                                    "aspect": "Follow-Up Tasks",
                                    "text": "@Agent_1 will investigate 
another way to fix the issue."
                                }
                            ],
                            "id": "conversation1",
                            "warnings": []
                        }
                    ],
                    "errors": [],
                    "modelVersion": "latest"
                }
            }
        ]
    }
}
Getting conversation issue and resolution
summarization results