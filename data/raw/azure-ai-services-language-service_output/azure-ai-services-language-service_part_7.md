Language support for custom text
classification
06/30/2025
Use this article to learn about the languages currently supported by custom text classification
feature.
With custom text classification, you can train a model in one language and use to classify
documents in another language. This feature is useful because it helps save time and effort.
Instead of building separate projects for every language, you can handle multi-lingual dataset
in one project. Your dataset doesn't have to be entirely in the same language but you should
enable the multi-lingual option for your project while creating or later in project settings. If you
notice your model performing poorly in certain languages during the evaluation process,
consider adding more data in these languages to your training set.
You can train your project entirely with English documents, and query it in: French, German,
Mandarin, Japanese, Korean, and others. Custom text classification makes it easy for you to
scale your projects to multiple languages by using multilingual technology to train your
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
\nCustom text classification supports .txt  files in the following languages:
Language
Language Code
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
Languages supported by custom text classification
ﾉ
Expand table
\nLanguage
Language Code
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
Language Code
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
Language Code
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
Custom text classification overview
Service limits
Next steps
\nFrequently asked questions
06/30/2025
Find answers to commonly asked questions about concepts, and scenarios related to custom
text classification in Azure AI Language.
See the quickstart to quickly create your first project, or view how to create projects for more
details.
See the service limits article.
See the language support article.
Generally, diverse and representative tagged data leads to better results, given that the
tagging is done precisely, consistently and completely. There's no set number of tagged classes
that will make every model perform well. Performance is highly dependent on your schema and
the ambiguity of your schema. Ambiguous classes need more tags. Performance also depends
on the quality of your tagging. The recommended number of tagged instances per class is 50.
The training process can take some time. As a rough estimate, the expected training time for
files with a combined length of 12,800,000 chars is 6 hours.
You can use the REST APIs
 to build your custom models. Follow this quickstart to get started
with creating a project and creating a model through APIs for examples of how to call the
How do I get started with the service?
What are the service limits?
Which languages are supported in this feature?
How many tagged files are needed?
Training is taking a long time, is this expected?
How do I build my custom model
programmatically?
\nAuthoring API.
When you're ready to start using your model to make predictions, you can use the REST API, or
the client library.
You can train multiple models on the same dataset within the same project. After you have
trained your model successfully, you can view its evaluation. You can deploy and test your
model within Language studio
. You can add or remove tags from your data and train a new
model and test it as well. View service limitsto learn about maximum number of trained models
with the same project. When you tag your data, you can determine how your dataset is split
into training and testing sets.
Model evaluation may not always be comprehensive, depending on:
If the test set is too small, the good/bad scores are not representative of model's actual
performance. Also if a specific class is missing or under-represented in your test set it will
affect model performance.
Data diversity if your data only covers few scenarios/examples of the text you expect in
production, your model will not be exposed to all possible scenarios and might perform
poorly on the scenarios it hasn't been trained on.
Data representation if the dataset used to train the model is not representative of the
data that would be introduced to the model in production, model performance will be
affected greatly.
See the data selection and schema design article for more information.
View the model confusion matrix, if you notice that a certain class is frequently classified
incorrectly, consider adding more tagged instances for this class. If you notice that two
classes are frequently classified as each other, this means the schema is ambiguous,
consider merging them both into one class for better performance.
Examine Data distribution If one of the classes has many more tagged instances than the
others, your model may be biased towards this class. Add more data to the other classes
or remove most of the examples from the dominating class.
What is the recommended CI/CD process?
Does a low or high model score guarantee bad or
good performance in production?
How do I improve model performance?
\nReview the data selection and schema design article for more information.
Review your test set to see predicted and tagged classes side-by-side so you can get a
better idea of your model performance, and decide if any changes in the schema or the
tags are necessary.
When you tag your data you can determine how your dataset is split into training and
testing sets. You can also have your data split randomly into training and testing sets, so
there is no guarantee that the reflected model evaluation is on the same test set, so
results are not comparable.
If you are retraining the same model, your test set will be the same, but you might notice
a slight change in predictions made by the model. This is because the trained model is
not robust enough, which is a factor of how representative and distinct your data is, and
the quality of your tagged data.
First, you need to enable the multilingual option when creating your project or you can enable
it later from the project settings page. After you train and deploy your model, you can start
querying it in multiple languages. You may get varied results for different languages. To
improve the accuracy of any language, add more tagged instances to your project in that
language to introduce the trained model to more syntax of that language. See language
support for more information.
You need to deploy your model before you can test it.
After deploying your model, you call the prediction API, using either the REST API or client
libraries.
When I retrain my model I get different results,
why is this?
How do I get predictions in different languages?
I trained my model, but I can't test it
How do I use my trained model to make
predictions?
\nCustom text classification is a data processor for General Data Protection Regulation (GDPR)
purposes. In compliance with GDPR policies, custom text classification users have full control to
view, export, or delete any user content either through the Language Studio
 or
programmatically by using REST APIs
.
Your data is only stored in your Azure Storage account. Custom text classification only has
access to read from it during training.
To clone your project you need to use the export API to export the project assets and then
import them into a new project. See REST APIs
 reference for both operations.
Custom text classification overview
Quickstart
Data privacy and security
How to clone my project?
Next steps
\nUse cases for custom text classification
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
Microsoft's Transparency Notes are part of a broader effort at Microsoft to put our AI principles
into practice. To find out more, see Microsoft AI Principles
.
Custom text classification is a cloud-based API service that applies machine-learning
intelligence to enable you to build custom models for text classification tasks.
Custom text classification supports two types of projects:
Single label classification: You assign only one label for each file in your dataset. For
example, if a file is a movie script, it could only be classified as "Action," "Thriller," or
"Romance."
Multiple label classification: You assign multiple labels for each file in your dataset. For
example, if a file is a movie script, it could be classified as "Action" or "Action" and
"Thriller."
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a Transparency Note?
Introduction to custom text classification
The basics of custom text classification