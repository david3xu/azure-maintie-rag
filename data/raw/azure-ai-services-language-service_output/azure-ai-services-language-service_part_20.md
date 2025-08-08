Generally after training a model you would review its evaluation details. In this quickstart, you
will just deploy your model, and make it available for you to try in Language studio, or you can
call the prediction API
.
To deploy your model from within the Language Studio
:
1. Select Deploying a model from the left side menu.
2. Select Add deployment to start the Add deployment wizard.
3. Select Create a new deployment name to create a new deployment and assign a trained
model from the dropdown below. You can otherwise select Overwrite an existing
deployment name to effectively replace the model that's used by an existing deployment.
The machine learning used to train models is regularly updated. To train on a
previous configuration version, select Select here to change from the Start a
training job page and choose a previous version.
Deploy your model

７ Note
Overwriting an existing deployment doesn't require changes to your Prediction
API
 call but the results you get will be based on the newly assigned model.
\n![Image](images/page191_image1.png)
\n4. Select a trained model from the Model dropdown.
5. Select Deploy to start the deployment job.
6. After deployment is successful, an expiration date will appear next to it. Deployment
expiration is when your deployed model will be unavailable to be used for prediction,
which typically happens twelve months after a training configuration expires.
To test your deployed models from within the Language Studio
:
1. Select Testing deployments from the left side menu.
2. For multilingual projects, from the Select text language dropdown, select the language of
the utterance you're testing.
3. From the Deployment name dropdown, select the deployment name corresponding to
the model that you want to test. You can only test models that are assigned to
deployments.
4. In the text box, enter an utterance to test. For example, if you created an application for
email-related utterances you could enter Delete this email.
5. Towards the top of the page, select Run the test.

Test deployed model
\n![Image](images/page192_image1.png)
\n6. After you run the test, you should see the response of the model in the result. You can
view the results in entities cards view or view it in JSON format.
When you don't need your project anymore, you can delete your project using Language
Studio. Select Projects from the left pane, select the project you want to delete, and then select
Delete from the top menu.
Learn about entity components
Clean up resources
Next steps
\nLanguage support for Conversational
Language Understanding (CLU)
05/19/2025
Use this article to learn about the languages currently supported by CLU feature.
With conversational language understanding, you can train a model in one language and use
to predict intents and entities from utterances in another language. This feature is powerful
because it helps save time and effort. Instead of building separate projects for every language,
you can handle multi-lingual dataset in one project. Your dataset doesn't have to be entirely in
the same language but you should enable the multi-lingual option for your project while
creating or later in project settings. If you notice your model performing poorly in certain
languages during the evaluation process, consider adding more data in these languages to
your training set.
You can train your project entirely with English utterances, and query it in: French, German,
Mandarin, Japanese, Korean, and others. Conversational language understanding makes it easy
for you to scale your projects to multiple languages by using multilingual technology to train
your models.
Whenever you identify that a particular language is not performing as well as other languages,
you can add utterances for that language in your project. In the tag utterances page in
Language Studio, you can select the language of the utterance you're adding. When you
introduce examples for that language to the model, it is introduced to more of the syntax of
that language, and learns to predict it better.
You aren't expected to add the same number of utterances for every language. You should
build the majority of your project in one language, and only add a few utterances in languages
you observe aren't performing well. If you create a project that is primarily in English, and start
testing it in French, German, and Spanish, you might observe that German doesn't perform as
well as the other two languages. In that case, consider adding 5% of your original English
Multi-lingual option
 Tip
We recommend using English for the LLM-powered features, like Quick Deploy and
Conversation-level understanding, but your project will continue to function for all
languages.
\nexamples in German, train a new model and test in German again. You should see better results
for German queries. The more utterances you add, the more likely the results are going to get
better.
When you add data in another language, you shouldn't expect it to negatively affect other
languages.
Projects with multiple languages enabled will allow you to specify synonyms per language for
every list key. Depending on the language you query your project with, you will only get
matches for the list component with synonyms of that language. When you query your project,
you can specify the language in the request body:
JSON
If you do not provide a language, it will fall back to the default language of your project.
Prebuilt components are similar, where you should expect to get predictions for prebuilt
components that are available in specific languages. The request's language again determines
which components are attempting to be predicted.
Conversational language understanding supports utterances in the following languages:
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
List and prebuilt components in multiple languages
"query": "{query}"
"language": "{language code}"
Languages supported by conversational language
understanding
ﾉ
Expand table
\nLanguage
Language code
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
English (UK)
en-gb
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
\nLanguage
Language code
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
\nLanguage
Language code
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
\nLanguage
Language code
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
Chinese (Traditional)
zh-hant
Zulu
zu
Conversational language understanding overview
Service limits
Next steps
\nFrequently asked questions for
conversational language understanding
06/04/2025
Use this article to quickly get the answers to common questions about conversational language
understanding
See the quickstart to quickly create your first project, or the how-to article for more details.
Yes, using orchestration workflow. See the orchestration workflow documentation for more
information.
Conversational language understanding is the next generation of LUIS.
For conversation projects, long training times are expected. Based on the number of examples
you have your training times may vary from 5 minutes to 1 hour or more.
See the entity components article.
See the language support article.
How do I create a project?
Can I use more than one conversational language
understanding project together?
What is the difference between LUIS and
conversational language understanding?
Training is taking a long time, is this expected?
How do I use entity components?
Which languages are supported in this feature?