4. Once the project is loaded, select Training jobs on the left. Press on Start a training job,
provide the model name v1 and press Train. All other settings such as Standard Training
and the evaluation settings can be left as is.
5. Once training is complete, click to Deploying a model on the left. Select Add Deployment
and create a new deployment with the name Testing, and assign model v1 to the
deployment.


\n![Image](images/page361_image1.png)

![Image](images/page361_image2.jpeg)
\nNow that your CLU project is deployed and ready, update the settings that will connect to the
deployment.
In the Core Bot sample, update your appsettings.json
 with the appropriate values.
The CluProjectName is FlightBooking.
The CluDeploymentName is Testing
The CluAPIKey can be either of the keys in the Keys and Endpoint section for your
Language resource in the Azure portal
. You can also copy your key from the Project
Settings tab in CLU.
The CluAPIHostName is the endpoint found in the Keys and Endpoint section for your
Language resource in the Azure portal. Note the format should be
<Language_Resource_Name>.cognitiveservices.azure.com  without https:// .
JSON

Update the settings file
{
  "MicrosoftAppId": "",
  "MicrosoftAppPassword": "",
  "CluProjectName": "",
  "CluDeploymentName": "",
  "CluAPIKey": "",
\n![Image](images/page362_image1.png)
\nIn the Core Bot sample, you can check out the FlightBookingRecognizer.cs file. Here is where
the CLU API call to the deployed endpoint is made to retrieve the CLU prediction for intents
and entities.
C#
Under the Dialogs folder, find the MainDialog which uses the following to make a CLU
prediction.
C#
The logic that determines what to do with the CLU result follows it.
C#
  "CluAPIHostName": ""
}
Identify integration points
        public FlightBookingRecognizer(IConfiguration configuration)
        {
            var cluIsConfigured = 
!string.IsNullOrEmpty(configuration["CluProjectName"]) && 
!string.IsNullOrEmpty(configuration["CluDeploymentName"]) && 
!string.IsNullOrEmpty(configuration["CluAPIKey"]) && 
!string.IsNullOrEmpty(configuration["CluAPIHostName"]);
            if (cluIsConfigured)
            {
                var cluApplication = new CluApplication(
                    configuration["CluProjectName"],
                    configuration["CluDeploymentName"],
                    configuration["CluAPIKey"],
                    "https://" + configuration["CluAPIHostName"]);
                // Set the recognizer options depending on which endpoint version 
you want to use.
                var recognizerOptions = new CluOptions(cluApplication)
                {
                    Language = "en"
                };
                _recognizer = new CluRecognizer(recognizerOptions);
            }
            var cluResult = await _cluRecognizer.RecognizeAsync<FlightBooking>
(stepContext.Context, cancellationToken);
\nRun the sample locally on your machine OR run the bot from a terminal or from Visual Studio:
From a terminal, navigate to the cognitive-service-language-samples/CoreBotWithCLU  folder.
 switch (cluResult.TopIntent().intent)
            {
                case FlightBooking.Intent.BookFlight:
                    // Initialize BookingDetails with any entities we may have 
found in the response.
                    var bookingDetails = new BookingDetails()
                    {
                        Destination = cluResult.Entities.toCity,
                        Origin = cluResult.Entities.fromCity,
                        TravelDate = cluResult.Entities.flightDate,
                    };
                    // Run the BookingDialog giving it whatever details we have 
from the CLU call, it will fill out the remainder.
                    return await 
stepContext.BeginDialogAsync(nameof(BookingDialog), bookingDetails, 
cancellationToken);
                case FlightBooking.Intent.GetWeather:
                    // We haven't implemented the GetWeatherDialog so we just 
display a TODO message.
                    var getWeatherMessageText = "TODO: get weather flow here";
                    var getWeatherMessage = 
MessageFactory.Text(getWeatherMessageText, getWeatherMessageText, 
InputHints.IgnoringInput);
                    await stepContext.Context.SendActivityAsync(getWeatherMessage, 
cancellationToken);
                    break;
                default:
                    // Catch all for unhandled intents
                    var didntUnderstandMessageText = $"Sorry, I didn't get that. 
Please try asking in a different way (intent was {cluResult.TopIntent().intent})";
                    var didntUnderstandMessage = 
MessageFactory.Text(didntUnderstandMessageText, didntUnderstandMessageText, 
InputHints.IgnoringInput);
                    await 
stepContext.Context.SendActivityAsync(didntUnderstandMessage, cancellationToken);
                    break;
            }
Run the bot locally
Run the bot from a terminal
\nThen run the following command
Bash
1. Launch Visual Studio
2. From the top navigation menu, select File, Open, then Project/Solution
3. Navigate to the cognitive-service-language-samples/CoreBotWithCLU  folder
4. Select the CoreBotCLU.csproj  file
5. Press F5  to run the project
Bot Framework Emulator
 is a desktop application that allows bot developers to test and
debug their bots on localhost or running remotely through a tunnel.
Install the latest Bot Framework Emulator
.
1. Launch Bot Framework Emulator
2. Select File, then Open Bot
3. Enter a Bot URL of http://localhost:3978/api/messages  and press Connect and wait for it
to load
4. You can now query for different examples such as "Travel from Cairo to Paris" and observe
the results
If the top intent returned from CLU resolves to "Book flight". Your bot will ask additional
questions until it has enough information stored to create a travel booking. At that point it will
return this booking information back to your user.
Learn more about the Bot Framework SDK.
# run the bot
dotnet run
Run the bot from Visual Studio
Testing the bot using Bot Framework Emulator
Connect to the bot using Bot Framework Emulator
Next steps
\nSupported prebuilt entity components
06/04/2025
Conversational Language Understanding allows you to add prebuilt components to entities.
Prebuilt components automatically predict common types from utterances. See the entity
components article for information on components.
The following prebuilt components are available in Conversational Language Understanding.
Type
Description
Supported languages
Quantity.Age
Age of a person or thing. For example: "30 years
old", "9 months old"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Quantity.Number
A cardinal number in numeric or text form. For
example: "Thirty", "23", "14.5", "Two and a half"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Quantity.Percentage
A percentage using the symbol % or the word
"percent". For example: "10%", "5.6 percent"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Quantity.Ordinal
An ordinal number in numeric or text form. For
example: "first", "second", "last", "10th"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Quantity.Dimension
Special dimensions such as length, distance,
volume, area, and speed. For example: "two miles",
"650 square kilometers", "35 km/h"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Quantity.Temperature
A temperature in Celsius or Fahrenheit. For
example: "32F", "34 degrees celsius", "2 deg C"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Quantity.Currency
Monetary amounts including currency. For example
"1000.00 US dollars", "£20.00", "$67.5B"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Quantity.NumberRange
A numeric interval. For example: "between 25 and
35"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Reference
ﾉ
Expand table
\nType
Description
Supported languages
Datetime
Dates and times. For example: "June 23, 1976", "7
AM", "6:49 PM", "Tomorrow at 7 PM", "Next Week"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Person.Name
The name of an individual. For example: "Joe",
"Ann"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Email
Email Addresses. For example:
"user@contoso.com", "user_name@contoso.com",
"user.name@contoso.com"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Phone Number
US Phone Numbers. For example: "123-456-7890",
"+1 123 456 7890", "(123)456-7890"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
URL
Website URLs and Links.
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
General.Organization
Companies and corporations. For example:
"Microsoft"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
Geography.Location
The name of a location. For example: "Tokyo"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
IP Address
An IP address. For example: "192.168.0.4"
English, Chinese, French,
Italian, German, Brazilian
Portuguese, Spanish
In multilingual conversation projects, you can enable any of the prebuilt components. The
component is only predicted if the language of the query is supported by the prebuilt entity.
The language is either specified in the request or defaults to the primary language of the
application if not provided.
Entity components
Prebuilt components in multilingual projects
Next steps
\nConversational language understanding
limits
06/04/2025
Use this article to learn about the data and service limits when using conversational language
understanding.
Your Language resource must be one of the following pricing tiers
:
Tier
Description
Limit
F0
Free tier
You are only allowed one F0 Language resource per subscription.
S
Paid tier
You can have up to 100 Language resources in the S tier per region.
See pricing
 for more information.
You can have up to 500 projects per resource.
Project names have to be unique within the same resource across all custom features.
See Language service regional availability.
Item
Request type
Maximum limit
Authoring API
POST
10 per minute
Authoring API
GET
100 per minute
Prediction API
GET/POST
1,000 per minute
Language resource limits
ﾉ
Expand table
Regional availability
API limits
ﾉ
Expand table
\nPricing tier
Item
Limit
F
Training time
1 hour per month
S
Training time
Unlimited, Standard
F
Prediction Calls
5,000 request per month
S
Prediction Calls
Unlimited, Standard
The following limits are observed for the conversational language understanding.
Item
Lower Limit
Upper Limit
Number of utterances per project
1
50,000
Utterance length in characters (authoring)
1
500
Utterance length in characters (prediction)
1
1000
Number of intents per project
1
500
Number of entities per project
0
350
Number of list synonyms per entity
0
20,000
Number of list synonyms per project
0
2,000,000
Number of prebuilt components per entity
0
7
Number of regular expressions per project
0
20
Number of trained models per project
0
10
Number of deployments per project
0
10
Quota limits
ﾉ
Expand table
Data limits
ﾉ
Expand table
Naming limits
\nItem
Limits
Project name
You can only use letters (a-z, A-Z) , and numbers (0-9)  , symbols _ . - , with no
spaces. Maximum allowed length is 50 characters.
Model name
You can only use letters (a-z, A-Z) , numbers (0-9)  and symbols _ . - . Maximum
allowed length is 50 characters.
Deployment
name
You can only use letters (a-z, A-Z) , numbers (0-9)  and symbols _ . - . Maximum
allowed length is 50 characters.
Intent name
You can only use letters (a-z, A-Z) , numbers (0-9)  and all symbols except ":", $ & % *
( ) + ~ # / ? . Maximum allowed length is 50 characters.
Entity name
You can only use letters (a-z, A-Z) , numbers (0-9)  and all symbols except ":", $ & % *
( ) + ~ # / ? . Maximum allowed length is 50 characters.
Conversational language understanding overview
ﾉ
Expand table
Next steps