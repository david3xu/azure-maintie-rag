The Text Analytics client library provides a TextAnalyticsClient
 to do analysis on
batches of documents.
It provides both synchronous and asynchronous operations to
access a specific use of text analysis, such as language detection or key phrase
extraction.
A document is a single unit to be analyzed by the predictive models in the Language
service.
The input for each operation is passed as a list of documents.
Each document can be passed as a string in the list, e.g.
Python
or, if you wish to pass in a per-item document id or language/country_hint, they can
be passed as a list of
DetectLanguageInput
 or
TextDocumentInput

or a dict-like
representation of the object:
Python
See service limitations
 for the input, including document length limits, maximum
batch size, and supported text encoding.
text_analytics_client = TextAnalyticsClient(endpoint, credential=credential)
Key concepts
TextAnalyticsClient
Input
documents = ["I hated the movie. It was so slow!", "The movie made it into 
my top ten favorites. What a great movie!"]
documents = [
    {"id": "1", "language": "en", "text": "I hated the movie. It was so 
slow!"},
    {"id": "2", "language": "en", "text": "The movie made it into my top ten 
favorites. What a great movie!"},
]
Return Value
\nThe return value for a single document can be a result or error object.
A heterogeneous
list containing a collection of result and error objects is returned from each operation.
These results/errors are index-matched with the order of the provided documents.
A result, such as AnalyzeSentimentResult
,
is the result of a text analysis operation and
contains a prediction or predictions about a document input.
The error object, DocumentError
, indicates that the service had trouble processing the
document and contains
the reason it was unsuccessful.
You can filter for a result or error object in the list by using the is_error attribute. For a
result object this is always False and for a
DocumentError
 this is True.
For example, to filter out all DocumentErrors you might use list comprehension:
Python
You can also use the kind attribute to filter between result types:
Python
Long-running operations are operations which consist of an initial request sent to the
service to start an operation,
followed by polling the service at intervals to determine
whether the operation has completed or failed, and if it has
succeeded, to get the result.
Methods that support healthcare analysis, custom text analysis, or multiple analyses are
modeled as long-running operations.
The client exposes a begin_<method-name> method
Document Error Handling
response = text_analytics_client.analyze_sentiment(documents)
successful_responses = [doc for doc in response if not doc.is_error]
poller = text_analytics_client.begin_analyze_actions(documents, actions)
response = poller.result()
for result in response:
    if result.kind == "SentimentAnalysis":
        print(f"Sentiment is {result.sentiment}")
    elif result.kind == "KeyPhraseExtraction":
        print(f"Key phrases: {result.key_phrases}")
    elif result.is_error is True:
        print(f"Document error: {result.code}, {result.message}")
Long-Running Operations
\nthat returns a poller object. Callers should wait
for the operation to complete by calling
result() on the poller object returned from the begin_<method-name> method.
Sample
code snippets are provided to illustrate using long-running operations below.
The following section provides several code snippets covering some of the most
common Language service tasks, including:
Analyze Sentiment
Recognize Entities
Recognize Linked Entities
Recognize PII Entities
Extract Key Phrases
Detect Language
Healthcare Entities Analysis
Multiple Analysis
Custom Entity Recognition
Custom Single Label Classification
Custom Multi Label Classification
Extractive Summarization
Abstractive Summarization
Dynamic Classification
analyze_sentiment
 looks at its input text and determines whether its sentiment is
positive, negative, neutral or mixed. It's response includes per-sentence sentiment
analysis and confidence scores.
Python
Examples
Analyze Sentiment
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, 
credential=AzureKeyCredential(key))
documents = [
    """I had the best day of my life. I decided to go sky-diving and it made 
\nThe returned response is a heterogeneous list of result and error objects:
list[AnalyzeSentimentResult
, DocumentError
]
Please refer to the service documentation for a conceptual discussion of sentiment
analysis. To see how to conduct more granular analysis into the opinions related to
individual aspects (such as attributes of a product or service) in a text, see here
.
recognize_entities
 recognizes and categories entities in its input text as people, places,
organizations, date/time, quantities, percentages, currencies, and more.
Python
me appreciate my whole life so much more.
    I developed a deep-connection with my instructor as well, and I feel as 
if I've made a life-long friend in her.""",
    """This was a waste of my time. All of the views on this drop are 
extremely boring, all I saw was grass. 0/10 would
    not recommend to any divers, even first timers.""",
    """This was pretty good! The sights were ok, and I had fun with my 
instructors! Can't complain too much about my experience""",
    """I only have one word for my experience: WOW!!! I can't believe I have 
had such a wonderful skydiving company right
    in my backyard this whole time! I will definitely be a repeat customer, 
and I want to take my grandmother skydiving too,
    I know she'll love it!"""
]
result = text_analytics_client.analyze_sentiment(documents, 
show_opinion_mining=True)
docs = [doc for doc in result if not doc.is_error]
print("Let's visualize the sentiment of each of these documents")
for idx, doc in enumerate(docs):
    print(f"Document text: {documents[idx]}")
    print(f"Overall sentiment: {doc.sentiment}")
Recognize Entities
import os
import typing
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, 
credential=AzureKeyCredential(key))
reviews = [
\nThe returned response is a heterogeneous list of result and error objects:
list[RecognizeEntitiesResult
, DocumentError
]
Please refer to the service documentation for a conceptual discussion of named entity
recognition
and supported types
.
recognize_linked_entities
 recognizes and disambiguates the identity of each entity
found in its input text (for example,
determining whether an occurrence of the word
Mars refers to the planet, or to the
Roman god of war). Recognized entities are
associated with URLs to a well-known knowledge base, like Wikipedia.
Python
    """I work for Foo Company, and we hired Contoso for our annual founding 
ceremony. The food
    was amazing and we all can't say enough good words about the quality and 
the level of service.""",
    """We at the Foo Company re-hired Contoso after all of our past 
successes with the company.
    Though the food was still great, I feel there has been a quality drop 
since their last time
    catering for us. Is anyone else running into the same problem?""",
    """Bar Company is over the moon about the service we received from 
Contoso, the best sliders ever!!!!"""
]
result = text_analytics_client.recognize_entities(reviews)
result = [review for review in result if not review.is_error]
organization_to_reviews: typing.Dict[str, typing.List[str]] = {}
for idx, review in enumerate(result):
    for entity in review.entities:
        print(f"Entity '{entity.text}' has category '{entity.category}'")
        if entity.category == 'Organization':
            organization_to_reviews.setdefault(entity.text, [])
            organization_to_reviews[entity.text].append(reviews[idx])
for organization, reviews in organization_to_reviews.items():
    print(
        "\n\nOrganization '{}' has left us the following review(s): 
{}".format(
            organization, "\n\n".join(reviews)
        )
    )
Recognize Linked Entities
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
\nThe returned response is a heterogeneous list of result and error objects:
list[RecognizeLinkedEntitiesResult
, DocumentError
]
Please refer to the service documentation for a conceptual discussion of entity linking
and supported types
.
recognize_pii_entities
 recognizes and categorizes Personally Identifiable Information
(PII) entities in its input text, such as
Social Security Numbers, bank account information,
credit card numbers, and more.
Python
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, 
credential=AzureKeyCredential(key))
documents = [
    """
    Microsoft was founded by Bill Gates with some friends he met at Harvard. 
One of his friends,
    Steve Ballmer, eventually became CEO after Bill Gates as well. Steve 
Ballmer eventually stepped
    down as CEO of Microsoft, and was succeeded by Satya Nadella.
    Microsoft originally moved its headquarters to Bellevue, Washington in 
January 1979, but is now
    headquartered in Redmond.
    """
]
result = text_analytics_client.recognize_linked_entities(documents)
docs = [doc for doc in result if not doc.is_error]
print(
    "Let's map each entity to it's Wikipedia article. I also want to see how 
many times each "
    "entity is mentioned in a document\n\n"
)
entity_to_url = {}
for doc in docs:
    for entity in doc.entities:
        print("Entity '{}' has been mentioned '{}' time(s)".format(
            entity.name, len(entity.matches)
        ))
        if entity.data_source == "Wikipedia":
            entity_to_url[entity.name] = entity.url
Recognize PII Entities
\nThe returned response is a heterogeneous list of result and error objects:
list[RecognizePiiEntitiesResult
, DocumentError
]
Please refer to the service documentation for supported PII entity types
.
Note: The Recognize PII Entities service is available in API version v3.1 and newer.
extract_key_phrases
 determines the main talking points in its input text. For example,
for the input text "The food was delicious and there were wonderful staff", the API
returns: "food" and "wonderful staff".
Python
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)
documents = [
    """Parker Doe has repaid all of their loans as of 2020-04-25.
    Their SSN is 859-98-0987. To contact them, use their phone number
    555-555-5555. They are originally from Brazil and have Brazilian CPF 
number 998.214.865-68"""
]
result = text_analytics_client.recognize_pii_entities(documents)
docs = [doc for doc in result if not doc.is_error]
print(
    "Let's compare the original document with the documents after redaction. 
"
    "I also want to comb through all of the entities that got redacted"
)
for idx, doc in enumerate(docs):
    print(f"Document text: {documents[idx]}")
    print(f"Redacted document text: {doc.redacted_text}")
    for entity in doc.entities:
        print("...Entity '{}' with category '{}' got redacted".format(
            entity.text, entity.category
        ))
Extract Key Phrases
\nThe returned response is a heterogeneous list of result and error objects:
list[ExtractKeyPhrasesResult
, DocumentError
]
Please refer to the service documentation for a conceptual discussion of key phrase
extraction.
detect_language
 determines the language of its input text, including the confidence
score of the predicted language.
Python
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, 
credential=AzureKeyCredential(key))
articles = [
    """
    Washington, D.C. Autumn in DC is a uniquely beautiful season. The leaves 
fall from the trees
    in a city chock-full of forests, leaving yellow leaves on the ground and 
a clearer view of the
    blue sky above...
    """,
    """
    Redmond, WA. In the past few days, Microsoft has decided to further 
postpone the start date of
    its United States workers, due to the pandemic that rages with no end in 
sight...
    """,
    """
    Redmond, WA. Employees at Microsoft can be excited about the new coffee 
shop that will open on campus
    once workers no longer have to work remotely...
    """
]
result = text_analytics_client.extract_key_phrases(articles)
for idx, doc in enumerate(result):
    if not doc.is_error:
        print("Key phrases in article #{}: {}".format(
            idx + 1,
            ", ".join(doc.key_phrases)
        ))
Detect Language
\nThe returned response is a heterogeneous list of result and error objects:
list[DetectLanguageResult
, DocumentError
]
Please refer to the service documentation for a conceptual discussion of language
detection
and language and regional support.
Long-running operation begin_analyze_healthcare_entities
 extracts entities
recognized within the healthcare domain, and identifies relationships between entities
within the input document and links to known sources of information in various well
known databases, such as UMLS, CHV, MSH, etc.
Python
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, 
credential=AzureKeyCredential(key))
documents = [
    """
    The concierge Paulette was extremely helpful. Sadly when we arrived the 
elevator was broken, but with Paulette's help we barely noticed this 
inconvenience.
    She arranged for our baggage to be brought up to our room with no extra 
charge and gave us a free meal to refurbish all of the calories we lost from
    walking up the stairs :). Can't say enough good things about my 
experience!
    """,
    """
    最近由于工作压力太大，我们决定去富酒店度假。那儿的温泉实在太舒服了，我跟我丈夫
都完全恢复了工作前的青春精神！加油！
    """
]
result = text_analytics_client.detect_language(documents)
reviewed_docs = [doc for doc in result if not doc.is_error]
print("Let's see what language each review is in!")
for idx, doc in enumerate(reviewed_docs):
    print("Review #{} is in '{}', which has ISO639-1 name '{}'\n".format(
        idx, doc.primary_language.name, doc.primary_language.iso6391_name
    ))
Healthcare Entities Analysis
\nimport os
import typing
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, 
HealthcareEntityRelation
endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
key = os.environ["AZURE_LANGUAGE_KEY"]
text_analytics_client = TextAnalyticsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
)
documents = [
    """
    Patient needs to take 100 mg of ibuprofen, and 3 mg of potassium. Also 
needs to take
    10 mg of Zocor.
    """,
    """
    Patient needs to take 50 mg of ibuprofen, and 2 mg of Coumadin.
    """
]
poller = text_analytics_client.begin_analyze_healthcare_entities(documents)
result = poller.result()
docs = [doc for doc in result if not doc.is_error]
print("Let's first visualize the outputted healthcare result:")
for doc in docs:
    for entity in doc.entities:
        print(f"Entity: {entity.text}")
        print(f"...Normalized Text: {entity.normalized_text}")
        print(f"...Category: {entity.category}")
        print(f"...Subcategory: {entity.subcategory}")
        print(f"...Offset: {entity.offset}")
        print(f"...Confidence score: {entity.confidence_score}")
        if entity.data_sources is not None:
            print("...Data Sources:")
            for data_source in entity.data_sources:
                print(f"......Entity ID: {data_source.entity_id}")
                print(f"......Name: {data_source.name}")
        if entity.assertion is not None:
            print("...Assertion:")
            print(f"......Conditionality: 
{entity.assertion.conditionality}")
            print(f"......Certainty: {entity.assertion.certainty}")
            print(f"......Association: {entity.assertion.association}")
    for relation in doc.entity_relations:
        print(f"Relation of type: {relation.relation_type} has the following 
roles")
        for role in relation.roles: