Feedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other
resources associated with it.
Azure portal
Azure CLI
Key phrase extraction overview

Clean up resources
Next steps
Yes
No
\n![Image](images/page461_image1.png)
\nLanguage support for Key Phrase
Extraction
06/21/2025
Use this article to find the natural languages supported by Key Phrase Extraction. Both the
cloud-based API and Docker containers support the same languages.
Total supported language codes: 94
Language
Language code
Notes
Afrikaans     
      af   
                    
Albanian    
      sq   
                    
Amharic    
      am   
                    
Arabic   
      ar   
                    
Armenian   
      hy   
                    
Assamese   
      as   
                    
Azerbaijani   
      az   
                    
Basque   
      eu   
                    
Belarusian
      be   
                    
Bengali    
      bn   
                    
Bosnian   
      bs   
                    
Breton   
      br   
                    
Bulgarian     
      bg   
                    
Burmese   
      my   
                    
Catalan    
      ca  
                    
Chinese-Simplified    
      zh-hans  
                    
Supported languages
ﾉ
Expand table
\nLanguage
Language code
Notes
Chinese-Traditional
      zh-hant  
                    
Croatian
hr
Czech   
      cs   
                    
Danish
da
 Dutch                 
      nl       
                    
 English               
      en       
                    
Esperanto   
      eo   
                    
Estonian              
      et       
                    
Filipino   
      fil   
                    
 Finnish               
      fi       
                    
 French                
      fr       
                    
Galician   
      gl   
                    
Georgian   
      ka   
                    
 German                
      de       
                    
Greek    
      el  
                    
Gujarati   
      gu   
                    
Hausa     
      ha   
                    
Hebrew   
      he   
                    
Hindi     
      hi   
                    
Hungarian    
      hu  
                    
 Indonesian            
      id       
                    
 Irish            
      ga       
                    
 Italian               
      it       
                    
 Japanese              
      ja       
                    
 Javanese            
      jv       
                    
 Kannada            
      kn       
                    
\nLanguage
Language code
Notes
 Kazakh            
      kk       
                    
 Khmer            
      km       
                    
 Korean                
      ko       
                    
 Kurdish (Kurmanji)   
      ku       
                    
 Kyrgyz            
      ky       
                    
 Lao            
      lo       
                    
 Latin            
      la       
                    
Latvian               
      lv       
                    
 Lithuanian            
      lt       
                    
 Macedonian            
      mk       
                    
 Malagasy            
      mg       
                    
 Malay            
      ms       
                    
 Malayalam            
      ml       
                    
 Marathi            
      mr       
                    
 Mongolian            
      mn       
                    
 Nepali            
      ne       
                    
 Norwegian (Bokmål)    
      no       
  nb  also accepted 
 Odia            
      or       
                    
 Oromo            
      om       
                    
 Pashto            
      ps       
                    
 Persian       
      fa       
                    
 Polish                
      pl       
                    
 Portuguese (Brazil)   
     pt-BR     
                    
 Portuguese (Portugal) 
     pt-PT     
  pt  also accepted 
 Punjabi            
      pa       
                    
Romanian              
      ro       
                    
\nLanguage
Language code
Notes
 Russian               
      ru       
                    
 Sanskrit            
      sa       
                    
 Scottish Gaelic       
      gd       
                    
 Serbian            
      sr       
                    
 Sindhi            
      sd       
                    
 Sinhala            
      si       
                    
Slovak                
      sk       
                    
Slovenian             
      sl       
                    
 Somali            
      so       
                    
 Spanish               
      es       
                    
 Sudanese            
      su       
                    
 Swahili            
      sw       
                    
 Swedish               
      sv       
                    
 Tamil            
      ta       
                    
 Telugu           
      te       
                    
 Thai            
      th       
                    
Turkish              
      tr       
                    
 Ukrainian           
      uk       
                    
 Urdu            
      ur       
                    
 Uyghur            
      ug       
                    
 Uzbek            
      uz       
                    
 Vietnamese            
      vi       
                    
 Welsh            
      cy       
                    
 Western Frisian       
      fy       
                    
 Xhosa            
      xh       
                    
 Yiddish            
      yi       
                    
\nHow to call the API for more information.
Quickstart: Use the key phrase extraction client library and REST API
Next steps
\nTransparency note for Key Phrase
Extraction
06/24/2025
An AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Creating a system that
is fit for its intended purpose requires an understanding of how the technology works, its
capabilities and limitations, and how to achieve the best performance. Microsoft's Transparency
Notes are intended to help you understand how our AI technology works, the choices system
owners can make that influence system performance and behavior, and the importance of
thinking about the whole system, including the technology, the people, and the environment.
You can use Transparency Notes when developing or deploying your own system, or share
them with the people who will use or be affected by your system.
Microsoft's Transparency notes are part of a broader effort at Microsoft to put our AI principles
into practice. To find out more, see Responsible AI principles from Microsoft.
Azure AI Language Key Phrase Extraction feature allows you to quickly identify the main
concepts in text. For example, in the text "The food was delicious and there were wonderful
staff", Key Phrase Extraction will return the main talking points: "food" and "wonderful staff".
Non-essential words are discarded single terms or phrases that appear to be the subject or
object of a sentence are returned.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a transparency note?
） Important
This article assumes that you're familiar with guidelines and best practices for Azure AI
Language. For more information, see Transparency note for Azure AI Language.
Introduction to key phrase extraction
\nNote that no confidence score is returned for this feature, unlike some other Azure AI
Language features.
Key Phrase Extraction is used in multiple scenarios across a variety of industries. Some
examples include:
Enhancing search. Key phrases can be used to create a search index that can enhance
search results. For example, customers can provide thousands of documents and then run
Key Phrase Extraction on top of it using the built-in Azure Search skill. The outcome of
this are key phrases from the input dataset, which can then be used to create an index.
This index can be updated by running the skill again whenever there is a new document
set available.
View aggregate trends in text data. For example, a word cloud can be generated with
key phrases to help visualize key concepts in text comments or feedback. For example, a
hotel could generate a word cloud based on key phrases identified in their comments and
might see that people are commenting most frequently about the location, cleanliness
and helpful staff.
Do not use
Do not use for automatic actions without human intervention for high risk scenarios. A
person should always review source data when another person's economic situation,
health or safety is affected.
Legal and regulatory considerations: Organizations need to evaluate potential specific legal
and regulatory obligations when using any AI services and solutions, which may not be
appropriate for use in every industry or scenario. Additionally, AI services or solutions are not
designed for and may not be used in ways prohibited in applicable terms of service and
relevant codes of conduct.
Depending on your scenario and input data, you could experience different levels of
performance. The following information is designed to help you understand key concepts
about performance as they apply to using the Azure AI Language key phrase extraction feature.
Example use cases
Considerations when choosing a use case
Characteristics and limitations
\nUnlike other Azure AI Language features' models, the key phrase extraction model is an
unsupervised model that is not trained on human labeled ground truth data. All of the noun
phrases in the text sent to the service are detected and then ranked based on frequency and
co-occurrence. Therefore, what is returned by the model may not agree with what a human
would choose as the most important phrases. In some cases the model may appear partially
correct, in that a noun is returned without the adjective that modifies it.
Longer text will perform better. Do not break your source text up into pieces like
sentences or paragraphs. Send the entire text, for example, a complete customer review
or paper abstract.
If your text includes some boilerplate or other text that has no topical relevance to the
actual content you're trying to analyze, the words in this text will affect your results. For
example, emails might have "Subject:", "Body:", "Sender:", etc. included in the text. We
recommend removing any known text that is not part of the actual content you are trying
to analyze before sending it to the service.
Transparency note for Azure AI Language
Transparency note for Named Entity Recognition and Personally Identifying Information
Transparency note for the health feature
Transparency note for Language Detection
Transparency note for Question answering
Transparency note for Summarization
Transparency note for Sentiment Analysis
Data Privacy and Security for Azure AI Language
Guidance for integration and responsible use with Azure AI Language
System limitations and best practices for enhancing
performance
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