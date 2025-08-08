Custom text classification is offered as part of the custom features within Azure AI Language.
This feature enables its users to build custom AI models to classify text into custom categories
predefined by the user. By creating a custom text classification project, developers can
iteratively tag data and train, evaluate, and improve model performance before they make it
available for consumption. The quality of the tagged data greatly affects model performance.
To simplify building and customizing your model, the service offers a custom web portal that
can be accessed through the Language Studio
. You can easily get started with the service by
following the steps in this quickstart.
The following terms are commonly used within custom text classification:
Term
Definition
Project
A project is a work area for building your custom AI models based on your data. Your project
can only be accessed by you and others who have contributor access to the Azure resource
being used. Within a project, you can tag data, build models, evaluate and improve them where
necessary, and eventually deploy a model to be ready for consumption. You can build multiple
models within your project on the same dataset.
Model
A model is an object that's trained to do a certain task. For this system, the models classify text.
Models are trained by learning from tagged data.
Class
A class is a user-defined category that indicates the overall classification of the text. Developers
tag their data with their assigned classes before they pass it to the model for training.
Custom text classification can be used in multiple scenarios across a variety of industries. Some
examples are:
Automatic emails or ticket triaging: Support centers of all types receive a high volume of
emails or tickets containing unstructured, freeform text and attachments. Timely review,
acknowledgment, and routing to subject matter experts within internal teams is critical.
Email triage at this scale requires people to review and route to the right departments,
which takes time and resources. Custom text classification can be used to analyze
incoming text, and triage and categorize the content to be automatically routed to the
relevant departments for further action.
Custom text classification terminology
ﾉ
Expand table
Example use cases for custom text classification
\nKnowledge mining to enhance and enrich semantic search: Search is foundational to
any app that surfaces text content to users. Common scenarios include catalog or
document searches, retail product searches, or knowledge mining for data science. Many
enterprises across various industries are seeking to build a rich search experience over
private, heterogeneous content, which includes both structured and unstructured
documents. As a part of their pipeline, developers can use custom text classification to
categorize their text into classes that are relevant to their industry. The predicted classes
can be used to enrich the indexing of the file for a more customized search experience.
Avoid using custom text classification for decisions that might have serious adverse
impacts. Include human review of decisions that have the potential for serious impacts on
individuals. For example, identifying whether to accept or reject an insurance claim based
on a user's description of an incident.
Avoid creating classes that are ambiguous and not representative. When you design
your schema, avoid classes that are so similar to each other that there might be difficulty
differentiating them from each other. For example, if you're classifying movie scripts,
avoid creating a class for romance, comedy, and rom-com. Instead, consider using a
multiple-label classification model with romance and comedy classes. Then, for rom-com
movies, assign both classes.
Legal and regulatory considerations: Organizations need to evaluate potential specific
legal and regulatory obligations when using any AI services and solutions, which may not
be appropriate for use in every industry or scenario. Additionally, AI services or solutions
are not designed for and may not be used in ways prohibited in applicable terms of
service and relevant codes of conduct.
Introduction to custom text classification
Characteristics and limitations for using custom text classification
Data privacy and security
Guidance for integration and responsible use
Microsoft AI principles
Considerations when you choose a use case
Next steps
\nGuidance for integration and responsible
use with custom text classification
06/24/2025
Microsoft works to help customers responsibly develop and deploy solutions by using custom
text classification. Our principled approach upholds personal agency and dignity by
considering the AI system's:
Fairness, reliability, and safety.
Privacy and security.
Inclusiveness.
Transparency.
Human accountability.
These considerations reflect our commitment to developing responsible AI.
When you get ready to integrate and responsibly use AI-powered products or features, the
following activities help to set you up for success:
Understand what it can do. Fully assess the potential of any AI system to understand its
capabilities and limitations. Understand how it will perform in your particular scenario and
context by thoroughly testing it with real-life conditions and data.
Respect an individual's right to privacy. Only collect data and information from
individuals for lawful and justifiable purposes. Only use data and information that you
have consent to use for this purpose.
Obtain legal review. Obtain appropriate legal advice to review your solution, particularly
if you'll use it in sensitive or high-risk applications. Understand what restrictions you
might need to work within and your responsibility to resolve any issues that might come
up in the future.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
General guidelines for integration and responsible
use principles
\nHave a human in the loop. Keep a human in the loop, and include human oversight as a
consistent pattern area to explore. Ensure constant human oversight of the AI-powered
product or feature. Maintain the role of humans in decision making. Make sure you can
have real-time human intervention in the solution to prevent harm and manage situations
when the AI system doesn’t perform as expected.
Maintain security. Ensure your solution is secure and has adequate controls to preserve
the integrity of your content and prevent unauthorized access.
Build trust with affected stakeholders. Communicate the expected benefits and potential
risks to affected stakeholders. Help people understand why the data is needed and how
the use of the data will lead to their benefit. Describe data handling in an understandable
way.
Create a customer feedback loop. Provide a feedback channel that allows users and
individuals to report issues with the service after it's deployed. After you've deployed an
AI-powered product or feature, it requires ongoing monitoring and improvement. Be
ready to implement any feedback and suggestions for improvement. Establish channels
to collect questions and concerns from affected stakeholders. People who might be
directly or indirectly affected by the system include employees, visitors, and the general
public. For example, consider using:
Feedback features built into app experiences.
An easy-to-remember email address for feedback.
Anonymous feedback boxes placed in semi-private spaces.
Knowledgeable representatives in the lobby.
Always plan to have the user confirm an action before being processed. Plan to have
your user confirm an action before being processed by your client application to avoid
incorrect responses that might come from the custom text classification models. For
example, suppose your custom text classification model is integrated in an insurance
claim approval system to classify nonurgent and urgent cases. Have someone on your
side confirm the model's prediction before processing it.
Always plan to have a correction path for the user. After a certain action is taken by the
client application, show a confirmation message to the user of the action that was
processed. Plan that the response of the custom text classification model might not be
accurate and that your user might end up in an error state. In that case, always have a
fallback plan or a correction path that the user can use to exit from that state.
Introduction to custom text classification
Next steps
\nCustom text classification Transparency Note
Microsoft AI principles
\nCharacteristics and limitations for using
custom text classification
06/24/2025
Performance of custom text classification models will vary based on the scenario and input
data. The following sections are designed to help you understand key concepts about
performance and evaluation of custom text classification models.
Reviewing model evaluation is an important step in the custom text classification model's
development life cycle. It helps you determine how well your model is performing and to
gauge the expected performance when the model is used in production.
In the process of building a model, training and testing sets are either defined during tagging
or chosen at random during training. Either way, the training and testing sets are essential for
training and evaluating custom text classification models. The training set is used to train the
custom machine learning model. The test set is used as a blind set to evaluate model
performance.
The model evaluation process is triggered after training is completed successfully. The
evaluation process takes place by using the trained model to predict user-defined classes for
files in the test set and compare the predictions with the provided data tags (ground truth).
The results are returned to you to review the model's performance.
The first step in calculating the model's evaluation is categorizing the predicted labels in one of
the following categories: true positives, false positives, or false negatives. The following table
further explains these terms.
Term
Correct/Incorrect
Definition
Example
True
positive
Correct
The model predicts a
class, and it's the same
For a comedy  movie script, the class comedy  is
predicted.
） Important
Non-English translations are provided for convenience only. Please consult the EN-US
version of this document for the binding version.
Performance evaluation metrics
ﾉ
Expand table
\nTerm
Correct/Incorrect
Definition
Example
as the text has been
tagged.
False
positive
Incorrect
The model predicts the
wrong class for a
specific text.
For a comedy  movie script, the class drama  is
predicted.
False
negative
Incorrect
The system doesn't
return a result when a
human judge would
return a correct result.
For a drama  movie script, the class comedy  is
predicted. In multiclassification only, for a
romance  and comedy  movie script, the class
comedy  is predicted, but the class romance  is
not predicted.
For single-label classification, it is not possible to have a false negative, because single-label
classification models will always predict one class for each file. For a multi-label classification it
is counted as both a false negative and false positive, false negative for the tagged class and
false positive for the predicted class.
The preceding categories are then used to calculate precision, recall and an F1 score. These
metrics are provided as part of the service's model evaluation. Here are the metric definitions
and how they're calculated:
Precision: The measure of the model's ability to predict actual positive classes. It's the ratio
between the predicted true positives and the actually tagged positives. Recall returns how
many predicted classes are correct.
Recall: The measure of the model's ability to predict actual positive classes. It's the ratio
between the predicted true positives and the actually tagged positives. Recall returns how
many predicted classes are correct.
F1 score: A function of precision and recall. An F1 score is needed when you are seeking a
balance between precision and recall.
Model evaluation scores might not always be comprehensive, especially if a specific class is
missing or underrepresented in the training data. This can occur if an insufficient number of
tagged files were provided in the training phase. This situation would affect the quantity and
quality of the testing split, which may affect the quality of the evaluation.
７ Note
For single classification, because the count of false positives and false negatives is always
equal, it follows that precision, recall, and the F1 score are always equal to each other.
\nAny custom text classification model is expected to experience both false negative and false
positive errors. You need to consider how each type of error affects the overall system and
carefully think through scenarios where true events won't be recognized and incorrect events
will be recognized. Depending on your scenario, precision or recall could be a more suitable
metric for evaluating your model's performance. For example, if your scenario is about ticket
triaging, predicting the wrong class would cause it to be forwarded to the wrong team, which
costs time and effort. In this case, your system should be more sensitive to false positives and
precision would then be a more relevant metric for evaluation.
If your scenario is about categorizing email as important or spam, failing to predict that a
certain email is important would cause you to miss it. But if spam email was mistakenly marked
important, you would simply disregard it. In this case, the system should be more sensitive to
false negatives and recall would then be a more relevant evaluation metric.
If you want to optimize for general purpose scenarios or when precision and recall are equally
important, the F1 score would be the most relevant metric. Evaluation scores are dependent on
your scenario and acceptance criteria. There's no absolute metric that will work for all
scenarios.
Understand service limitations: There are some limits enforced on the user, such as the
number of files and classes contained in your data or entity length. Learn more about
system limitations.
Plan your schema: Identify the categories that you want to classify your data into. You
need to plan your schema to avoid ambiguity and to take the complexity of classes into
consideration. Learn more about recommended practices.
Select training data: The quality of training data is an important factor in model quality.
Using diverse and real-life data similar to the data you expect during production will
make the model more robust and better able to handle real-life scenarios. Make sure to
include all layouts and formats of text that will be used in production. If the model isn't
exposed to a certain scenario or class during training, it won't be able to recognize it in
production. Learn more about recommended practices.
Tag data accurately: The quality of your tagged data is a key factor in model
performance, and it's considered the ground truth from which the model learns. Tag
precisely and consistently. When you tag a specific file, make sure you assign it to the
most relevant class. Make sure similar files in your data are always tagged with the same
class. Make sure all classes are well represented and that you have a balanced data
System limitations and best practices for
enhancing system performance
\ndistribution across all entities. Examine data distribution to make sure all your classes are
adequately represented. If a certain class is tagged less frequently than the others, this
class may be underrepresented and may not be recognized properly by the model during
production. In this case, consider adding more files from the underrepresented class to
your training data and then train a new model.
Review evaluation and improve model: After the model is successfully trained, check the
model evaluation and confusion matrix. This review helps you understand where your
model went wrong and learn about classes that aren't performing well. It's also
considered a best practice to review the test set and view the predicted and tagged
classes side by side. It gives you a better idea of the model's performance and helps you
decide if any changes in the schema or the tags are necessary. You can also review the
confusion matrix to identify classes that are often mistakenly predicted to see if anything
can be done to improve model performance.
The following guidelines will help you to understand and improve performance in custom text
classification.
After you've tagged data and trained your model, you'll need to deploy it to be consumed in a
production environment. Deploying a model means making it available for use via the runtime
API
 to predict classes for a submitted text. The API returns a JSON object that contains the
predicted class or classes and the confidence score. The confidence score is a decimal number
between zero (0) and one (1). It serves as an indicator of how confident the system is with its
prediction. A higher value indicates higher confidence in the accuracy of that result. The
returned score is directly affected by the data you tagged when you built the custom model. If
the user's input is similar to the data used in training, higher scores and more accurate
predictions can be expected. If a certain class is consistently predicted with a low confidence
score, you might want to examine the tagged data and add more instances for this class, and
then retrain the model.
The confidence score threshold can be adjusted based on your scenario. You can automate
decisions in your scenario based on the confidence score the system returns. You can also set a
certain threshold so that predicted classes with confidence scores higher or lower than this
General guidelines to understand and improve
performance
Understand confidence scores
Set confidence score thresholds
\nthreshold are treated differently. For example, if a prediction is returned with a confidence
score below the threshold, the file can be flagged for additional review.
Different scenarios call for different approaches. If the actions based on the predicted class will
have high-impacts, you might decide to set a higher threshold to ensure accuracy of
classification. In this case, you would expect fewer false positives but more false negatives
resulting in higher precision. If no high-impact decision based on the predicted class will be
made in your scenario, you might accept a lower threshold because you would want to predict
all possible classes that might apply to the submitted text (in multi-label classification cases). In
this case, you would expect more false positives but fewer false negatives. The result is a higher
recall.
It's very important to evaluate your system with the set thresholds by using real data that the
system will process in production to determine the effects on precision and recall.
Retraining the same model without any changes in tagged data will result in the same model
output, and as a result, the same evaluation scores. If you add or remove tags, the model
performance changes accordingly. Provided that no new files were added during tagging, the
evaluation scores can be compared with the previous version of the model because both have
the same files in the training and testing sets.
Adding new files or training a different model with random set splits leads to different files in
training and testing sets. Although changes in evaluation scores might occur, they can't be
directly compared to other models because performance is calculated on different splits for
test sets.
After you've trained your model, you can review model evaluation details to identify areas for
improvement. The model-level metrics provide information on the overall model performance.
By observing the class-level performance metrics, you can identify if there are any issues within
a specific class.
If you notice that a specific class has low performance, it means the model is having trouble
predicting it. This issue could be due to an ambiguous schema, which means the class can't be
differentiated from other classes. It could also be caused by a data imbalance, which means
this class is underrepresented. In this instance you will need to add more tagged examples for
the model to better predict this class.
Different training sessions and changes in evaluation
Review incorrect predictions to improve performance