Language
Language code
Notes
Russian
ru
Sanskrit (new)
sa
Scottish Gaelic (new)
gd
Serbian
sr
Sindhi (new)
sd
Sinhala (new)
si
Slovak
sk
Slovenian
sl
Somali
so
Spanish
es
Sundanese (new)
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
Western Frisian (new)
fy
Xhosa (new)
xh
Yiddish (new)
yi
\nTotal supported language codes: 94
Language
Language code
Notes
Afrikaans (new)
af
Albanian (new)
sq
Amharic (new)
am
Arabic
ar
Armenian (new)
hy
Assamese (new)
as
Azerbaijani (new)
az
Basque (new)
eu
Belarusian (new)
be
Bengali
bn
Bosnian (new)
bs
Breton (new)
br
Bulgarian (new)
bg
Burmese (new)
my
Catalan (new)
ca
Chinese (Simplified)
zh-hans
zh  also accepted
Chinese (Traditional) (new)
zh-hant
Croatian (new)
hr
Czech (new)
cs
Danish
da
Dutch
nl
English
en
Opinion Mining language support
ﾉ
Expand table
\nLanguage
Language code
Notes
Esperanto (new)
eo
Estonian (new)
et
Filipino (new)
fil
Finnish
fi
French
fr
Galician (new)
gl
Georgian (new)
ka
German
de
Greek
el
Gujarati (new)
gu
Hausa (new)
ha
Hebrew (new)
he
Hindi
hi
Hungarian
hu
Indonesian
id
Irish (new)
ga
Italian
it
Japanese
ja
Javanese (new)
jv
Kannada (new)
kn
Kazakh (new)
kk
Khmer (new)
km
Korean
ko
Kurdish (Kurmanji)
ku
Kyrgyz (new)
ky
Lao (new)
lo
\nLanguage
Language code
Notes
Latin (new)
la
Latvian (new)
lv
Lithuanian (new)
lt
Macedonian (new)
mk
Malagasy (new)
mg
Malay (new)
ms
Malayalam (new)
ml
Marathi
mr
Mongolian (new)
mn
Nepali (new)
ne
Norwegian
no
Odia (new)
or
Oromo (new)
om
Pashto (new)
ps
Persian (new)
fa
Polish
pl
Portuguese (Portugal)
pt-PT
pt  also accepted
Portuguese (Brazil)
pt-BR
Punjabi (new)
pa
Romanian (new)
ro
Russian
ru
Sanskrit (new)
sa
Scottish Gaelic (new)
gd
Serbian (new)
sr
Sindhi (new)
sd
Sinhala (new)
si
\nLanguage
Language code
Notes
Slovak (new)
sk
Slovenian (new)
sl
Somali (new)
so
Spanish
es
Sundanese (new)
su
Swahili (new)
sw
Swedish
sv
Tamil
ta
Telugu
te
Thai (new)
th
Turkish
tr
Ukrainian (new)
uk
Urdu (new)
ur
Uyghur (new)
ug
Uzbek (new)
uz
Vietnamese (new)
vi
Welsh (new)
cy
Western Frisian (new)
fy
Xhosa (new)
xh
Yiddish (new)
yi
With Custom sentiment analysis, you can train a model in one language and use to classify
documents in another language. This feature is useful because it helps save time and effort.
Instead of building separate projects for every language, you can handle multi-lingual dataset
in one project. Your dataset doesn't have to be entirely in the same language but you should
Multi-lingual option (Custom sentiment analysis
only)
\nenable the multi-lingual option for your project while creating or later in project settings. If you
notice your model performing poorly in certain languages during the evaluation process,
consider adding more data in these languages to your training set.
You can train your project entirely with English documents, and query it in: French, German,
Mandarin, Japanese, Korean, and others. Custom sentiment analysis makes it easy for you to
scale your projects to multiple languages by using multilingual technology to train your
models.
Whenever you identify that a particular language is not performing as well as other languages,
you can add more documents for that language in your project.
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
how to call the API for more information.
Quickstart: Use the Sentiment Analysis client library and REST API
Next steps
\nTransparency note for Sentiment Analysis
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
into practice. To find out more, see Microsoft AI Principles
.
The Sentiment Analysis feature of Azure AI Language evaluates text and returns sentiment
scores and labels for each sentence. This is useful for detecting positive, neutral and negative
sentiment in social media, customer reviews, discussion forums and other product and service
scenarios.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
What is a transparency note?
） Important
This article assumes that you're familiar with guidelines and best practices for Azure AI
Language. For more information, see Transparency note for Azure AI Language.
The basics of Sentiment Analysis
Introduction
\nSentiment analysis provides sentiment labels (such as "negative", "neutral" and "positive")
based on the highest confidence score found by the service at a sentence and document-level.
This feature also returns confidence scores between 0 and 1 for each document and sentence
for positive, neutral and negative sentiment. Scores closer to 1 indicate a higher confidence in
the label's classification, while lower scores indicate lower confidence. By default, the overall
sentiment label is the greatest of the three confidence scores, however, you can define a
threshold for any or all of the individual sentiment confidence scores depending on what works
best for your scenario. For each document or each sentence, the predicted scores associated
with the labels (positive, negative and neutral) add up to 1. Read more details about sentiment
labels and scores.
In addition, the optional opinion mining feature returns aspects (such as the attributes of
products or services) and their associated opinion words. For each aspect an overall sentiment
label is returned along with confidence scores for positive and negative sentiment. For
example, the sentence "The restaurant had great food and our waiter was friendly" has two
aspects, "food" and "waiter," and their corresponding opinion words are "great" and "friendly."
The two aspects therefore receive sentiment classification positive , with confidence scores
between 0 and 1.0. Read more details about opinion mining.
See the JSON response for this example.
Sentiment Analysis can be used in multiple scenarios across a variety of industries. Some
examples include:
Monitor for positive and negative feedback trends in aggregate. After introducing a
new product, a retailer can use the sentiment service to monitor multiple social media
outlets for mentions of the product and its associated sentiment. The trending sentiment
can be used in product meetings to make business decisions about the new product.
Run sentiment analysis on raw text results of surveys to gain insights for analysis and
follow-up with participants (customers, employees, consumers, etc.). A store with a
policy to follow up on customers' negative reviews within 24 hours and positive reviews
within a week can use the sentiment service to categorize reviews for easy and timely
follow up.
Help customer service staff improve customer engagement through insights captured
from real-time analysis of interactions. Extract insights from transcribed customer
Capabilities
System behavior
Use cases
\nservices calls to better understand customer-agent interactions and trends to improve
customer engagements.
Avoid automatic actions without human intervention for high impact scenarios. For
example, employee bonuses should not be automatically based on sentiment scores from
their customer service interaction text. Source data should always be reviewed when a
person's economic situation, health or safety is affected.
Carefully consider scenarios outside of the product and service review domain. Since
the model is trained on product and service reviews, the system may not accurately
recognize sentiment focused language in other domains. Always make sure to test the
system on operational test datasets to ensure you get the performance you need. Your
operational test dataset should reflect the real data your system will see in production
with all the characteristics and variation you will have when your product is deployed.
Synthetic data and tests that don't reflect your end-to-end scenario likely won't be
sufficient.
Carefully consider scenarios that take automatic action to filter or remove content. You
can add a human review cycle and/or re-rank content (rather than filtering it completely)
if your goal is to ensure content meets your community standards.
Legal and regulatory considerations: Organizations need to evaluate potential specific
legal and regulatory obligations when using any AI services and solutions, which may not
be appropriate for use in every industry or scenario. Additionally, AI services or solutions
are not designed for and may not be used in ways prohibited in applicable terms of
service and relevant codes of conduct.
Depending on your scenario and input data, you could experience different levels of
performance. The following information is designed to help you understand system limitations
and key concepts about performance as they apply to Sentiment Analysis.
Key limitations to consider:
The machine learning model that is used to predict sentiment was trained on product and
service reviews. That means the service will perform most accurately for similar scenarios
and less accurately for scenarios outside the scope of the product and service reviews. For
example, personnel reviews may use different language to describe sentiment and thus,
you might not get the results or performance you would expect. A word like "strong" in
the phrase "Shafali was a strong leader" may not obtain a positive sentiment because the
word strong may not have a clear positive sentiment in product and service reviews.
Considerations when choosing a use case
Limitations
\nSince the model is trained on product and service reviews, dialects and language that are
less represented in the dataset may have lower accuracy.
The model has no understanding of the relative importance of various sentences that are
sent together. Since the overall sentiment is a simple aggregate score of the sentences,
the overall sentiment score may not agree with a human's interpretation which would
take into account the fact that some sentences may have more importance in
determining the overall sentiment.
The model may not recognize sarcasm. Context, like tone of voice, facial expression, the
author of the text, the audience for the text, or prior conversation are often important to
understanding the sentiment. With sarcasm, additional context is often needed to
recognize if a text input is positive or negative. Given that the service only sees the text
input, classifying sarcastic sentiment may be less accurate. For example, that was
awesome, could be either positive or negative depending on the context, tone of voice,
facial expression, author and the audience.
The confidence score magnitude does not reflect the intensity of the sentiment. It is
based on the confidence of the model for a particular sentiment (positive, neutral,
negative). Therefore, if your system depends on the intensity of the sentiment, consider
using a human reviewer or post processing logic on the individual opinion scores or the
original text to help rank the intensity of the sentiment.
While we’ve made efforts to reduce the bias exhibited by our models, the limitations that
come with language models, including the potential for it to produce inaccurate,
unreliable, and biased output, apply to the Azure AI Language Sentiment Analysis model.
We expect the model to have some false negatives and positives for now, but we are
eager to collect user feedback to aid our ongoing work to improve this service.
Because sentiment is somewhat subjective, it is not possible to provide a universally applicable
estimate of performance for the model. Ultimately, performance depends on a number of
factors such as the subject domain, the characteristics of the text processed, the use case for
the system, and how people interpret the system's output.
You may find confidence scores for positive, negative, and neutral sentiments differ according
to your scenario. Instead of using the overall sentence level sentiment for the full document or
sentence, you can define a threshold for any or all of the individual sentiment confidence
scores that works best for your scenario. For example, if it is more important to identify all
potential instances of negative sentiment, you can use a lower threshold on the negative
sentiment instead of looking at the overall sentiment label. This means that you may get more
Best practices for improving system performance