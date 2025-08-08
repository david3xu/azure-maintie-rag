Language support for custom named entity
recognition
06/30/2025
Use this article to learn about the languages currently supported by custom named entity
recognition feature.
With custom NER, you can train a model in one language and use to extract entities from
documents in another language. This feature is powerful because it helps save time and effort.
Instead of building separate projects for every language, you can handle multi-lingual dataset
in one project. Your dataset doesn't have to be entirely in the same language but you should
enable the multi-lingual option for your project while creating or later in project settings. If you
notice your model performing poorly in certain languages during the evaluation process,
consider adding more data in these languages to your training set.
You can train your project entirely with English documents, and query it in: French, German,
Mandarin, Japanese, Korean, and others. Custom named entity recognition makes it easy for
you to scale your projects to multiple languages by using multilingual technology to train your
models.
Whenever you identify that a particular language is not performing as well as other languages,
you can add more documents for that language in your project. In the data labeling page in
Language Studio, you can select the language of the document you're adding. When you
introduce more documents for that language to the model, it is introduced to more of the
syntax of that language, and learns to predict it better.
You aren't expected to add the same number of documents for every language. You should
build the majority of your project in one language, and only add a few documents in languages
you observe aren't performing well. If you create a project that is primarily in English, and start
testing it in French, German, and Spanish, you might observe that German doesn't perform as
well as the other two languages. In that case, consider adding 5% of your original English
documents in German, train a new model and test in German again. You should see better
results for German queries. The more labeled documents you add, the more likely the results
are going to get better.
When you add data in another language, you shouldn't expect it to negatively affect other
languages.
Multi-lingual option
\nCustom NER supports .txt  files in the following languages:
Language
Language code
Afrikaans
af
Amharic
am
Arabic
ar
Assamese
as
Azerbaijani
az
Belarusian
be
Bulgarian
bg
Bengali
bn
Breton
br
Bosnian
bs
Catalan
ca
Czech
cs
Welsh
cy
Danish
da
German
de
Greek
el
English (US)
en-us
Esperanto
eo
Spanish
es
Estonian
et
Basque
eu
Persian
fa
Language support
ﾉ
Expand table
\nLanguage
Language code
Finnish
fi
French
fr
Western Frisian
fy
Irish
ga
Scottish Gaelic
gd
Galician
gl
Gujarati
gu
Hausa
ha
Hebrew
he
Hindi
hi
Croatian
hr
Hungarian
hu
Armenian
hy
Indonesian
id
Italian
it
Japanese
ja
Javanese
jv
Georgian
ka
Kazakh
kk
Khmer
km
Kannada
kn
Korean
ko
Kurdish (Kurmanji)
ku
Kyrgyz
ky
Latin
la
Lao
lo
\nLanguage
Language code
Lithuanian
lt
Latvian
lv
Malagasy
mg
Macedonian
mk
Malayalam
ml
Mongolian
mn
Marathi
mr
Malay
ms
Burmese
my
Nepali
ne
Dutch
nl
Norwegian (Bokmal)
nb
Odia
or
Punjabi
pa
Polish
pl
Pashto
ps
Portuguese (Brazil)
pt-br
Portuguese (Portugal)
pt-pt
Romanian
ro
Russian
ru
Sanskrit
sa
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
\nLanguage
Language code
Albanian
sq
Serbian
sr
Sundanese
su
Swedish
sv
Swahili
sw
Tamil
ta
Telugu
te
Thai
th
Filipino
tl
Turkish
tr
Uyghur
ug
Ukrainian
uk
Urdu
ur
Uzbek
uz
Vietnamese
vi
Xhosa
xh
Yiddish
yi
Chinese (Simplified)
zh-hans
Zulu
zu
Custom NER overview
Service limits
Next steps
\nFrequently asked questions for Custom
Named Entity Recognition
06/30/2025
Find answers to commonly asked questions about concepts, and scenarios related to custom
NER in Azure AI Language.
See the quickstart to quickly create your first project, or view how to create projects for more
detailed information.
See the service limits article for more information.
Generally, diverse and representative tagged data leads to better results, given that the
tagging is done precisely, consistently and completely. There is no set number of tagged
instances that will make every model perform well. Performance highly dependent on your
schema, and the ambiguity of your schema. Ambiguous entity types need more tags.
Performance also depends on the quality of your tagging. The recommended number of
tagged instances per entity is 50.
The training process can take a long time. As a rough estimate, the expected training time for
files with a combined length of 12,800,000 chars is 6 hours.
How do I get started with the service?
What are the service limits?
How many tagged files are needed?
Training is taking a long time, is this expected?
How do I build my custom model
programmatically?
７ Note
Currently you can only build a model using the REST API or Language Studio.
\nYou can use the REST APIs
 to build your custom models. Follow this quickstart to get started
with creating a project and creating a model through APIs for examples of how to call the
Authoring API.
When you're ready to start using your model to make predictions, you can use the REST API, or
the client library.
You can train multiple models on the same dataset within the same project. After you have
trained your model successfully, you can view its performance. You can deploy and test your
model within Language studio
. You can add or remove labels from your data and train a new
model and test it as well. View service limits to learn about maximum number of trained
models with the same project. When you train a model, you can determine how your dataset is
split into training and testing sets. You can also have your data split randomly into training and
testing set where there is no guarantee that the reflected model evaluation is about the same
test set, and the results are not comparable. It's recommended that you develop your own test
set and use it to evaluate both models so you can measure improvement.
Model evaluation may not always be comprehensive. This depends on:
If the test set is too small so the good/bad scores are not representative of model's actual
performance. Also if a specific entity type is missing or under-represented in your test set
it will affect model performance.
Data diversity if your data only covers few scenarios/examples of the text you expect in
production, your model will not be exposed to all possible scenarios and might perform
poorly on the scenarios it hasn't been trained on.
Data representation if the dataset used to train the model is not representative of the
data that would be introduced to the model in production, model performance will be
affected greatly.
See the data selection and schema design article for more information.
View the model confusion matrix. If you notice that a certain entity type is frequently not
predicted correctly, consider adding more tagged instances for this class. If you notice
What is the recommended CI/CD process?
Does a low or high model score guarantee bad or
good performance in production?
How do I improve model performance?
\nthat two entity types are frequently predicted as each other, this means the schema is
ambiguous, and you should consider merging them both into one entity type for better
performance.
Review test set predictions. If one of the entity types has a lot more tagged instances than
the others, your model may be biased towards this type. Add more data to the other
entity types or remove examples from the dominating type.
Learn more about data selection and schema design.
Review your test set to see predicted and tagged entities side-by-side so you can get a
better idea of your model performance, and decide if any changes in the schema or the
tags are necessary.
When you train your model, you can determine if you want your data to be split randomly
into train and test sets. If you do, so there is no guarantee that the reflected model
evaluation is on the same test set, so results are not comparable.
If you're retraining the same model, your test set will be the same, but you might notice a
slight change in predictions made by the model. This is because the trained model is not
robust enough and this is a factor of how representative and distinct your data is and the
quality of your tagged data.
First, you need to enable the multilingual option when creating your project or you can enable
it later from the project settings page. After you train and deploy your model, you can start
querying it in multiple languages. You may get varied results for different languages. To
improve the accuracy of any language, add more tagged instances to your project in that
language to introduce the trained model to more syntax of that language.
You need to deploy your model before you can test it.
Why do I get different results when I retrain my
model?
How do I get predictions in different languages?
I trained my model, but I can't test it
How do I use my trained model for predictions?
\nAfter deploying your model, you call the prediction API, using either the REST API or client
libraries.
Custom NER is a data processor for General Data Protection Regulation (GDPR) purposes. In
compliance with GDPR policies, Custom NER users have full control to view, export, or delete
any user content either through the Language Studio
 or programmatically by using REST
APIs
.
Your data is only stored in your Azure Storage account. Custom NER only has access to read
from it during training.
To clone your project you need to use the export API to export the project assets, and then
import them into a new project. See the REST API
 reference for both operations.
Custom NER overview
Quickstart
Data privacy and security
How to clone my project?
Next steps
\nHow to create custom NER project
06/30/2025
Use this article to learn how to set up the requirements for starting with custom NER and
create a project.
Before you start using custom NER, you will need:
An Azure subscription - Create one for free
.
Before you start using custom NER, you will need an Azure AI Language resource. It is
recommended to create your Language resource and connect a storage account to it in the
Azure portal. Creating a resource in the Azure portal lets you create an Azure storage account
at the same time, with all of the required permissions pre-configured. You can also read further
in the article to learn how to use a pre-existing resource, and configure it to work with custom
named entity recognition.
You also will need an Azure storage account where you will upload your .txt  documents that
will be used to train a model to extract entities.
You can create a resource in the following ways:
The Azure portal
Language Studio
Prerequisites
Create a Language resource
７ Note
You need to have an owner role assigned on the resource group to create a
Language resource.
If you will connect a pre-existing storage account, you should have an owner role
assigned to it.
Create Language resource and connect storage
account