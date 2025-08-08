An AI system includes the technology, the people who will use it, the people who will be
affected by it, and the environment in which it is deployed. Read the transparency note for Text
Analytics for health to learn about responsible AI use and deployment in your systems. You can
also refer to the following articles for more information:
Transparency note for Azure AI Language
Integration and responsible use
Data, privacy, and security
\nQuickstart: Using Text Analytics for
health client library and REST API
Article • 02/16/2025
This article contains Text Analytics for health quickstarts that help with using the
supported client libraries, C#, Java, NodeJS, and Python as well as with using the REST
API.
Create a Project in Foundry in the Azure AI Foundry Portal
Using the left side pane, select Playgrounds. Then select the Try the Language
Playground button.
 Tip
You can use Azure AI Foundry to try summarization without needing to write code.
Prerequisites
Navigate to the Azure AI Foundry Playground
\nThe Language Playground consists of four sections:
Top banner: You can select any of the currently available Language services here.
Right pane: This pane is where you can find the Configuration options for the
service, such as the API and model version, along with features specific to the
service.
Center pane: This pane is where you enter your text for processing. After the
operation is run, some results are shown here.
Right pane: This pane is where Details of the run operation are shown.
Here you can select the Text Analytics for Health capability by choosing the top banner
tile, Extract health information.

Use Text Analytics for Health in the Azure AI
Foundry Playground
Use Extract health information
\n![Image](images/page1293_image1.png)
\nExtract health information is designed to identify and extract health information in text.
In Configuration there are the following options:
Option
Description
Select API version
Select which version of the API to use.
Select model version
Select which version of the model to use.
Select text language
Select which language the language is input in.
Return output in FHIR
structure
Returns the output in the Fast Healthcare Interoperability Resources
(FHIR) structure.
After your operation is completed, the type of entity is displayed beneath each entity in
the center pane and the Details section contains the following fields for each entity:
Field
Description
Entity
The detected entity.
Category
The type of entity that was detected.
Confidence
How confident the model is in the correctness of identification of entity's type.
ﾉ
Expand table
ﾉ
Expand table

Clean up resources
\n![Image](images/page1294_image1.png)
\nFeedback
Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
If you want to clean up and remove an Azure AI services subscription, you can delete the
resource or resource group. Deleting the resource group also deletes any other
resources associated with it.
Azure portal
Azure CLI
How to call the hosted API
How to use the service with Docker containers
Next steps
Yes
No
\nLanguage support for Text Analytics for
health
06/21/2025
Use this article to learn which natural languages are supported by Text Analytics for health and
its Docker container.
The hosted API service supports the English, Spanish, French, German, Italian, and Portuguese
languages.
When structuring the API request, the relevant language tags must be added for these
languages:
JSON
Hosted API Service
English – “en”
Spanish – “es”
French  - “fr”
German – “de”
Italian – “it”
Portuguese – “pt”
json
{
    "analysisInput": {
        "documents": [
            {
                "text": "El médico prescrió 200 mg de ibuprofeno.",
                "language": "es",
                "id": "1"
            }
        ]
    },
    "tasks": [
        {
            "taskName": "analyze 1",
            "kind": "Healthcare",
            "parameters":
            {
            "modelVersion": "2022-08-15-preview"
            }
\nThe docker container supports the English, Spanish, French, German, Italian, Portuguese and
Hebrew languages. Full details for deploying the service in a container can be found here.
In order to download the new container images from the Microsoft public container registry,
use the following docker pull
 command.
For English, Spanish, Italian, French, German and Portuguese:
For Hebrew:
When structuring the API request, the relevant language tags must be added for these
languages:
The following json is an example of a JSON file attached to the Language request's POST body,
for a Spanish document:
JSON
        }
    ]
}
Docker container
docker pull mcr.microsoft.com/azure-cognitive-
services/textanalytics/healthcare:latin
docker pull mcr.microsoft.com/azure-cognitive-
services/textanalytics/healthcare:semitic
English – “en”
Spanish – “es”
French  - “fr”
German – “de”
Italian – “it”
Portuguese – “pt”
Hebrew – “he”
\nText Analytics for health overview
json
{
    "analysisInput": {
        "documents": [
            {
                "text": "El médico prescrió 200 mg de ibuprofeno.",
                "language": "es",
                "id": "1"
            }
        ]
    },
    "tasks": [
        {
            "taskName": "analyze 1",
            "kind": "Healthcare",
        }
    ]
}
See also
\nTransparency note for Text Analytics for
health
06/24/2025
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a transparency note?
） Important
Text Analytics for health is a capability provided “AS IS” and “WITH ALL FAULTS.” Text
Analytics for health is not intended or made available for use as a medical device, clinical
support, diagnostic tool, or other technology intended to be used in the diagnosis, cure,
mitigation, treatment, or prevention of disease or other conditions, and no license or right
is granted by Microsoft to use this capability for such purposes. This capability is not
designed or intended to be implemented or deployed as a substitute for professional
medical advice or healthcare opinion, diagnosis, treatment, or the clinical judgment of a
healthcare professional, and should not be used as such. The customer is solely
responsible for any use of Text Analytics for health. The customer must separately license
any and all source vocabularies it intends to use under the terms set for that UMLS
Metathesaurus License Agreement Appendix or any future equivalent link. The customer is
responsible for ensuring compliance with those license terms, including any geographic or
other applicable restrictions.
Text Analytics for health now allows extraction of Social Determinants of Health (SDOH)
and ethnicity mentions in text. This capability may not cover all potential SDOH and does
not derive inferences based on SDOH or ethnicity (for example, substance use information
is surfaced, but substance abuse is not inferred). All decisions leveraging outputs of the
Text Analytics for health that impact individuals or resource allocation (including, but not
limited to, those related to billing, human resources, or treatment managing care) should
be made with human oversight and not be based solely on the findings of the model. The
purpose of the SDOH and ethnicity extraction capability is to help providers improve
health outcomes and it should not be used to stigmatize or draw negative inferences
\nAn AI system includes not only the technology, but also the people who will use it, the people
who will be affected by it, and the environment in which it is deployed. Creating a system that
is fit for its intended purpose requires an understanding of how the technology works, what its
capabilities and limitations are, and how to achieve the best performance. Microsoft’s
Transparency Notes are intended to help you understand how our AI technology works, the
choices system owners can make that influence system performance and behavior, and the
importance of thinking about the whole system, including the technology, the people, and the
environment. You can use Transparency Notes when developing or deploying your own system,
or share them with the people who will use or be affected by your system.
Microsoft's Transparency notes are part of a broader effort at Microsoft to put our AI principles
into practice. To find out more, see Responsible AI principles
 from Microsoft.
The Text Analytics for health feature of Azure AI Language uses natural language processing
techniques to find and label valuable health information such as diagnoses, symptoms,
medications, and treatments in unstructured text. The service can be used for diverse types of
unstructured medical documents, including discharge summaries, clinical notes, clinical trial
protocols, medical publications, and more. Text Analytics for health performs Named Entity
Recognition (NER), extracts relations between identified entities, surfaces assertions such as
negation and conditionality, and links detected entities to common vocabularies.
Text Analytics for health can receive unstructured text in English as part of its general
availability offering. Additional languages are currently supported in a preview offering. For
more information, see Language support.
You can read an overview of the API and its capabilities. Also, see supported entities and
relations.
Additionally, customization is now offered for Text Analytics for health under the new preview
feature, custom Text Analytics for health. Custom Text Analytics for health allows customers to
use their own data train a custom NER model, designed for healthcare, to extract their domain
specific categories, extending the existing Text Analytics for health entity map. Customers can
also define lexicon or specific vocabulary for the newly defined custom entities as well as
existing Text Analytics for health entities such as Medication Name. Therefore, custom Text
about the users or consumers of SDOH data, or patient populations beyond the stated
purpose of helping providers improving health outcomes.
The basics of Text Analytics for health
Introduction