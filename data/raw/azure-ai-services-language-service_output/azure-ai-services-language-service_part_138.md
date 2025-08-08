The following list presents all the recognized relations by the Text Analytics for health API.
ABBREVIATION
AMOUNT_OF_SUBSTANCE_USE
BODY_SITE_OF_CONDITION
BODY_SITE_OF_TREATMENT
COURSE_OF_CONDITION
COURSE_OF_EXAMINATION
COURSE_OF_MEDICATION
COURSE_OF_TREATMENT
DIRECTION_OF_BODY_STRUCTURE
DIRECTION_OF_CONDITION
DIRECTION_OF_EXAMINATION
DIRECTION_OF_TREATMENT
DOSAGE_OF_MEDICATION
EXAMINATION_FINDS_CONDITION
EXPRESSION_OF_GENE
EXPRESSION_OF_VARIANT
FORM_OF_MEDICATION
FREQUENCY_OF_CONDITION
            {
                "ref": "#/results/documents/0/entities/2",
                "role": "Route"
            }
        ]
    }
]
Recognized relations
\nFREQUENCY_OF_MEDICATION
FREQUENCY_OF_SUBSTANCE_USE
FREQUENCY_OF_TREATMENT
MUTATION_TYPE_OF_GENE
MUTATION_TYPE_OF_VARIANT
QUALIFIER_OF_CONDITION
RELATION_OF_EXAMINATION
ROUTE_OF_MEDICATION
SCALE_OF_CONDITION
TIME_OF_CONDITION
TIME_OF_EVENT
TIME_OF_EXAMINATION
TIME_OF_MEDICATION
TIME_OF_TREATMENT
UNIT_OF_CONDITION
UNIT_OF_EXAMINATION
VALUE_OF_CONDITION
VALUE_OF_EXAMINATION
VARIANT_OF_GENE
How to call the Text Analytics for health
Next steps
\nAssertion detection
06/21/2025
The meaning of medical content is highly affected by modifiers, such as negative or conditional
assertions, which can have critical implications if misrepresented. Text Analytics for health
supports four categories of assertion detection for entities in the text:
Certainty
Conditional
Association
Temporal
Text Analytics for health returns assertion modifiers, which are informative attributes assigned
to medical concepts that provide a deeper understanding of the concepts’ context within the
text. These modifiers are divided into four categories, each focusing on a different aspect and
containing a set of mutually exclusive values. Only one value per category is assigned to each
entity. The most common value for each category is the Default value. The service’s output
response contains only assertion modifiers that are different from the default value. In other
words, if no assertion is returned, the implied assertion is the default value.
CERTAINTY – provides information regarding the presence (present vs. absent) of the concept
and how certain the text is regarding its presence (definite vs. possible).
Positive [Default]: the concept exists or has happened.
Negative: the concept does not exist now or never happened.
Positive_Possible: the concept likely exists but there is some uncertainty.
Negative_Possible: the concept’s existence is unlikely but there is some uncertainty.
Neutral_Possible: the concept may or may not exist without a tendency to either side.
An example of assertion detection is shown below where a negated entity is returned with a
negative value for the certainty category:
JSON
Assertion output
{
    "offset": 381,
    "length": 3,
    "text": "SOB",
    "category": "SymptomOrSign",
    "confidenceScore": 0.98,
    "assertion": {
        "certainty": "negative"
\nCONDITIONALITY – provides information regarding whether the existence of a concept
depends on certain conditions.
None [Default]: the concept is a fact and not hypothetical and does not depend on
certain conditions.
Hypothetical: the concept may develop or occur in the future.
Conditional: the concept exists or occurs only under certain conditions.
ASSOCIATION – describes whether the concept is associated with the subject of the text or
someone else.
Subject [Default]: the concept is associated with the subject of the text, usually the
patient.
Other: the concept is associated with someone who is not the subject of the text.
TEMPORAL - provides additional temporal information for a concept detailing whether it is an
occurrence related to the past, present, or future.
Current [Default]: the concept is related to conditions/events that belong to the current
encounter. For example, medical symptoms that have brought the patient to seek medical
attention (e.g., “started having headaches 5 days prior to their arrival to the ER”). This
includes newly made diagnoses, symptoms experienced during or leading to this
encounter, treatments and examinations done within the encounter.
Past: the concept is related to conditions, examinations, treatments, medication events
that are mentioned as something that existed or happened prior to the current
encounter, as might be indicated by hints like s/p, recently, ago, previously, in childhood,
at age X. For example, diagnoses that were given in the past, treatments that were done,
past examinations and their results, past admissions, etc. Medical background is
considered as PAST.
Future: the concept is related to conditions/events that are planned/scheduled/suspected
to happen in the future, e.g., will be obtained, will undergo, is scheduled in two weeks
    },
    "name": "Dyspnea",
    "links": [
        {
            "dataSource": "UMLS",
            "id": "C0013404"
        },
        {
            "dataSource": "AOD",
            "id": "0000005442"
        },
    ...
}
\nfrom now.
How to call the Text Analytics for health
Next steps
\nUtilizing Fast Healthcare Interoperability
Resources (FHIR) structuring in Text
Analytics for Health
Article • 04/29/2025
When you process unstructured data using Text Analytics for health, you can request that the
output response includes a Fast Healthcare Interoperability Resources (FHIR) resource bundle.
The FHIR resource bundle output is enabled by passing the FHIR version as part of the options
in each request. How you pass the FHIR version differs depending on whether you're using the
SDK or the REST API.
When you use the REST API as part of building the request payload, you include a Tasks object.
Each of the Tasks can have parameters. One of the options for parameters is fhirVersion . By
including the fhirVersion  parameter in the Task object parameters, you're requesting the
output to include a FHIR resource bundle in addition to the normal Text Analytics for health
output. The following example shows the inclusion of fhirVersion  in the request parameters.
JSON
Use the REST API
{
      "analysis input": {
            "documents:"[
                {
                text:"54 year old patient had pain in the left elbow with no 
relief from 100 mg Ibuprofen",
                "language":"en",
                "id":"1"
                }
            ]
        },
    "tasks"[
       {
       "taskId":"analyze 1",
       "kind":"Healthcare",
       "parameters":
            {
            "fhirVersion":"4.0.1"
            }
        }
    ]
}
\nOnce the request has completed processing by Text Analytics for health and you pull the
response from the REST API, you'll find the FHIR resource bundle in the output. You can locate
the FHIR resource bundle inside each document processed using the property name
fhirBundle . The following partial sample is output highlighting the fhirBundle .
JSON
You can also use the SDK to make the request for Text Analytics for health to include the FHIR
resource bundle in the output. To accomplish this request with the SDK, you would create an
instance of AnalyzeHealthcareEntitiesOptions  and populate the FhirVersion  property with the
{
  "jobID":"50d11b05-7a03-a611-6f1e95ebde07",
  "lastUpdatedDateTime":"2024-06-05T17:29:51Z",
  "createdDateTime:"2024-06-05T17:29:40Z",
  "expirationDateTime":"2024-06-05T17:29:40Z",
  "status":"succeeded",
  "errors":[],
  "tasks":{
    "completed": 1,
    "failed": 0,
    "inProgress": 0,
    "total": 1,
    "items": [
        {
          "kind":"HealthcareLROResults",
          "lastUpdatedDateTime":"2024-06-05T17:29:51.5839858Z",
          "status":"succeeded",
          "results": {
              "documents": [
                  {
                    "id": "1",
                    "entities": [...
                    ],
                    "relations": [...
                    ].
                    "warnings":[],
                    "fhirBundle": {
                        "resourceType": "Bundle",
                        "id": "b4d907ed-0334-4186-9e21-8ed4d79e709f",
                        "meta": {
                            "profile": [
                                
"http://hl7.org/fhir/4.0.1/StructureDefinition/Bundle"
                                  ]
                                },  
Use the REST SDK
\nFHIR version. This options object is then passed to each StartAnalyzeHealthcareEntitiesAsync
method call to configure the request to include a FHIR resource bundle in the output.
How to call the Text Analytics for health
Next steps
\nWhat is summarization?
Article • 03/05/2025
Summarization is a feature offered by Azure AI Language, a combination of generative
Large Language models and task-optimized encoder models that offer summarization
solutions with higher quality, cost efficiency, and lower latency. Use this article to learn
more about this feature, and how to use it in your applications.
Out of the box, the service provides summarization solutions for three types of genre,
plain texts, conversations, and native documents. Text summarization only accepts plain
text blocks. Conversation summarization accepts conversational input, including various
speech audio signals. Native document summarization accepts documents in their
native formats, such as Word, PDF, or plain text. For more information, see Supported
document formats.
This documentation contains the following article types:
） Important
Our preview region, Sweden Central, showcases our latest and continually evolving
LLM fine tuning techniques based on GPT models. You are welcome to try them out
with a Language resource in the Sweden Central region.
Conversation summarization is only available using:
REST API
Python
C#
 Tip
Try out Summarization in Azure AI Foundry portal
. There you can utilize a
currently existing Language Studio resource or create a new Azure AI Foundry
resource in order to use this service.
Capabilities
Text summarization
\nQuickstarts are getting-started instructions to guide you through making
requests to the service.
How-to guides contain instructions for using the service in more specific or
customized ways.
To use this feature, you submit data for analysis and handle the API output in your
application. Analysis is performed as-is, with no added customization to the model
used on your data.
1. Create an Azure AI Language resource, which grants you access to the
features offered by Azure AI Language. It generates a password (called a key)
and an endpoint URL that you use to authenticate API requests.
2. Create a request using either the REST API or the client library for C#, Java,
JavaScript, and Python. You can also send asynchronous calls with a batch
request to combine API requests for multiple features into a single call.
3. Send the request containing your text data. Your key and endpoint are used
for authentication.
4. Stream or store the response locally.
Text summarization uses natural language processing techniques to generate a
summary for plain texts, which can be from a document, conversation, or any texts.
There are two approaches of summarization this API provides:
Extractive summarization: Produces a summary by extracting salient
sentences within the source text, together the positioning information of these
sentences.
Multiple extracted sentences: These sentences collectively convey the main
idea of the input text. They're original sentences extracted from the input
text content.
Rank score: The rank score indicates how relevant a sentence is to the main
topic. Text summarization ranks extracted sentences, and you can
determine whether they're returned in the order they appear, or according
to their rank. For example, if you request a three-sentence summary
extractive summarization returns the three highest scored sentences.
Positional information: The start position and length of extracted sentences.
Typical workflow
Key features for text summarization