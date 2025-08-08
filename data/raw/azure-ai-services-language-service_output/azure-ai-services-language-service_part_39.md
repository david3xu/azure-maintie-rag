Install the client library by right-clicking on the solution in the Solution Explorer and selecting
Manage NuGet Packages. In the package manager that opens select Browse and search for
Azure.AI.TextAnalytics . Select version 5.2.0 , and then Install. You can also use the Package
Manager Console.
Copy the following code into your program.cs file and run the code.
C#
Code example
using Azure;
using System;
using System.Globalization;
using Azure.AI.TextAnalytics;
namespace EntityLinkingExample
{
    class Program
    {
        // This example requires environment variables named "LANGUAGE_KEY" and 
"LANGUAGE_ENDPOINT"
        static string languageKey = 
Environment.GetEnvironmentVariable("LANGUAGE_KEY");
        static string languageEndpoint = 
Environment.GetEnvironmentVariable("LANGUAGE_ENDPOINT");
        private static readonly AzureKeyCredential credentials = new 
AzureKeyCredential(languageKey);
        private static readonly Uri endpoint = new Uri(languageEndpoint);
        
        // Example method for recognizing entities and providing a link to an 
online data source.
        static void EntityLinkingExample(TextAnalyticsClient client)
        {
            var response = client.RecognizeLinkedEntities(
                "Microsoft was founded by Bill Gates and Paul Allen on April 4, 
1975, " +
                "to develop and sell BASIC interpreters for the Altair 8800. " +
                "During his career at Microsoft, Gates held the positions of 
chairman, " +
                "chief executive officer, president and chief software architect, 
" +
                "while also being the largest individual shareholder until May 
2014.");
            Console.WriteLine("Linked Entities:");
            foreach (var entity in response.Value)
            {
                Console.WriteLine($"\tName: {entity.Name},\tID: 
{entity.DataSourceEntityId},\tURL: {entity.Url}\tData Source: 
{entity.DataSource}");
\nConsole
                Console.WriteLine("\tMatches:");
                foreach (var match in entity.Matches)
                {
                    Console.WriteLine($"\t\tText: {match.Text}");
                    Console.WriteLine($"\t\tScore: {match.ConfidenceScore:F2}\n");
                }
            }
        }
        static void Main(string[] args)
        {
            var client = new TextAnalyticsClient(endpoint, credentials);
            EntityLinkingExample(client);
            Console.Write("Press any key to exit.");
            Console.ReadKey();
        }
    }
}
Output
Linked Entities:
    Name: Microsoft,        ID: Microsoft,  URL: 
https://en.wikipedia.org/wiki/Microsoft    Data Source: Wikipedia
    Matches:
            Text: Microsoft
            Score: 0.55
            Text: Microsoft
            Score: 0.55
    Name: Bill Gates,       ID: Bill Gates, URL: 
https://en.wikipedia.org/wiki/Bill_Gates   Data Source: Wikipedia
    Matches:
            Text: Bill Gates
            Score: 0.63
            Text: Gates
            Score: 0.63
    Name: Paul Allen,       ID: Paul Allen, URL: 
https://en.wikipedia.org/wiki/Paul_Allen   Data Source: Wikipedia
    Matches:
            Text: Paul Allen
            Score: 0.60
    Name: April 4,  ID: April 4,    URL: https://en.wikipedia.org/wiki/April_4      
\nIf you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other resources
associated with it.
Azure portal
Azure CLI
Entity linking language support
How to call the entity linking API
Reference documentation
Additional samples
Data Source: Wikipedia
    Matches:
            Text: April 4
            Score: 0.32
    Name: BASIC,    ID: BASIC,      URL: https://en.wikipedia.org/wiki/BASIC        
Data Source: Wikipedia
    Matches:
            Text: BASIC
            Score: 0.33
    Name: Altair 8800,      ID: Altair 8800,        URL: 
https://en.wikipedia.org/wiki/Altair_8800  Data Source: Wikipedia
    Matches:
            Text: Altair 8800
            Score: 0.88
Clean up resources
Next steps
\nEntity linking language support
06/21/2025
Language
Language code
Notes
English
en
Spanish
es
Entity linking overview
ﾉ
Expand table
Next steps
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
\nHow to use entity linking
06/21/2025
The entity linking feature can be used to identify and disambiguate the identity of an entity
found in text (for example, determining whether an occurrence of the word "Mars" refers to the
planet, or to the Roman god of war). It will return the entities in the text with links to
Wikipedia
 as a knowledge base.
To use entity linking, you submit raw unstructured text for analysis and handle the API output
in your application. Analysis is performed as-is, with no additional customization to the model
used on your data. There are two ways to use entity linking:
Development option
Description
Language studio
Language Studio is a web-based platform that lets you try entity linking with text
examples without an Azure account, and your own data when you sign up. For
more information, see the Language Studio website
.
REST API or Client
library (Azure SDK)
Integrate entity linking into your applications using the REST API, or the client
library available in a variety of languages. For more information, see the entity
linking quickstart.
By default, entity linking will use the latest available AI model on your text. You can also
configure your API requests to use a specific model version.
When you submit documents to be processed by entity linking, you can specify which of the
supported languages they're written in. if you don't specify a language, entity linking will
default to English. Due to multilingual and emoji support, the response may contain text
offsets.
Development options
ﾉ
Expand table
Determine how to process the data (optional)
Specify the entity linking model
Input languages
\nEntity linking produces a higher-quality result when you give it smaller amounts of text to work
on. This is opposite from some features, like key phrase extraction which performs better on
larger blocks of text. To get the best results from both operations, consider restructuring the
inputs accordingly.
To send an API request, you will need a Language resource endpoint and key.
Analysis is performed upon receipt of the request. Using entity linking synchronously is
stateless. No data is stored in your account, and results are returned immediately in the
response.
When using this feature asynchronously, the API results are available for 24 hours from the
time the request was ingested, and is indicated in the response. After this time period, the
results are purged and are no longer available for retrieval.
You can stream the results to an application, or save the output to a file on the local system.
For information on the size and number of requests you can send per minute and second, see
the service limits article.
Entity linking overview
Submitting data
７ Note
You can find the key and endpoint for your Language resource on the Azure portal. They
will be located on the resource's Key and endpoint page, under resource management.
Getting entity linking results
Service and data limits
See also