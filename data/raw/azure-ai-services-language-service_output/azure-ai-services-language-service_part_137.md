Was this page helpful?
Provide product feedback 
| Get help at Microsoft Q&A
Yes
No
\nSupported Text Analytics for health entity
categories
06/21/2025
Text Analytics for health processes and extracts insights from unstructured medical data. The
service detects and surfaces medical concepts, assigns assertions to concepts, infers semantic
relations between concepts and links them to common medical ontologies.
Text Analytics for health detects medical concepts that fall under the following categories.
BODY_STRUCTURE - Body systems, anatomic locations or regions, and body sites. For
example, arm, knee, abdomen, nose, liver, head, respiratory system, lymphocytes.
AGE - All age terms and phrases, including ones for patients, family members, and others. For
example, 40-year-old, 51 yo, 3 months old, adult, infant, elderly, young, minor, middle-aged.
ETHNICITY - Phrases that indicate the ethnicity of the subject. For example, African American
or Asian.
GENDER - Terms that disclose the gender of the subject. For example, male, female, woman,
gentleman, lady.
Anatomy
Entities

Demographics
Entities


\n![Image](images/page1362_image1.png)

![Image](images/page1362_image2.png)

![Image](images/page1362_image3.png)
\nEXAMINATION_NAME – Diagnostic procedures and tests, including vital signs and body
measurements. For example, MRI, ECG, HIV test, hemoglobin, platelets count, scale systems
such as Bristol stool scale.
ALLERGEN – an antigen triggering an allergic reaction. For example, cats, peanuts.
COURSE - Description of a change in another entity over time, such as condition progression
(for example: improvement, worsening, resolution, remission), a course of treatment or
medication (for example: increase in medication dosage).
DATE - Full date relating to a medical condition, examination, treatment, medication, or
administrative event.
Examinations
Entities

External Influence
Entities

General attributes
Entities

\n![Image](images/page1363_image1.png)

![Image](images/page1363_image2.png)

![Image](images/page1363_image3.png)
\nDIRECTION – Directional terms that may relate to a body structure, medical condition,
examination, or treatment, such as: left, lateral, upper, posterior.
FREQUENCY - Describes how often a medical condition, examination, treatment, or medication
occurred, occurs, or should occur.
TIME - Temporal terms relating to the beginning and/or length (duration) of a medical
condition, examination, treatment, medication, or administrative event.
MEASUREMENT_UNIT – The unit of measurement related to an examination or a medical
condition measurement.
MEASUREMENT_VALUE – The value related to an examination or a medical condition
measurement.
RELATIONAL_OPERATOR - Phrases that express the quantitative relation between an entity
and some additional information.




\n![Image](images/page1364_image1.png)

![Image](images/page1364_image2.png)

![Image](images/page1364_image3.png)

![Image](images/page1364_image4.png)
\nVARIANT - All mentions of gene variations and mutations. For example, c.524C>T ,
(MTRR):r.1462_1557del96
GENE_OR_PROTEIN – All mentions of names and symbols of human genes as well as
chromosomes and parts of chromosomes and proteins. For example, MTRR, F2.
MUTATION_TYPE - Description of the mutation, including its type, effect, and location. For
example, trisomy, germline mutation, loss of function.
EXPRESSION - Gene expression level. For example, positive for-, negative for-, overexpressed,
detected in high/low levels, elevated.
ADMINISTRATIVE_EVENT – Events that relate to the healthcare system but of an
administrative/semi-administrative nature. For example, registration, admission, trial, study
entry, transfer, discharge, hospitalization, hospital stay.

Genomics
Entities


Healthcare
Entities
\n![Image](images/page1365_image1.png)

![Image](images/page1365_image2.png)

![Image](images/page1365_image3.png)
\nCARE_ENVIRONMENT – An environment or location where patients are given care. For
example, emergency room, physician’s office, cardio unit, hospice, hospital.
HEALTHCARE_PROFESSION – A healthcare practitioner licensed or non-licensed. For example,
dentist, pathologist, neurologist, radiologist, pharmacist, nutritionist, physical therapist,
chiropractor.
DIAGNOSIS – Disease, syndrome, poisoning. For example, breast cancer, Alzheimer’s, HTN,
CHF, spinal cord injury.
SYMPTOM_OR_SIGN – Subjective or objective evidence of disease or other diagnoses. For
example, chest pain, headache, dizziness, rash, SOB, abdomen was soft, good bowel sounds,
well nourished.
CONDITION_QUALIFIER - Qualitative terms that are used to describe a medical condition. All
the following subcategories are considered qualifiers:
Time-related expressions: those are terms that describe the time dimension qualitatively,
such as sudden, acute, chronic, longstanding.


Medical condition
Entities

\n![Image](images/page1366_image1.png)

![Image](images/page1366_image2.png)

![Image](images/page1366_image3.png)
\nQuality expressions: Those are terms that describe the “nature” of the medical condition,
such as burning, sharp.
Severity expressions: severe, mild, a bit, uncontrolled.
Extensivity expressions: local, focal, diffuse.
CONDITION_SCALE – Qualitative terms that characterize the condition by a scale, which is a
finite ordered list of values.
MEDICATION_CLASS – A set of medications that have a similar mechanism of action, a related
mode of action, a similar chemical structure, and/or are used to treat the same disease. For
example, ACE inhibitor, opioid, antibiotics, pain relievers.
MEDICATION_NAME – Medication mentions, including copyrighted brand names, and non-
brand names. For example, Ibuprofen.
DOSAGE - Amount of medication ordered. For example, Infuse Sodium Chloride solution 1000
mL.
MEDICATION_FORM - The form of the medication. For example, solution, pill, capsule, tablet,
patch, gel, paste, foam, spray, drops, cream, syrup.


Medication
Entities

\n![Image](images/page1367_image1.png)

![Image](images/page1367_image2.png)

![Image](images/page1367_image3.png)
\nMEDICATION_ROUTE - The administration method of medication. For example, oral, topical,
inhaled.
FAMILY_RELATION – Mentions of family relatives of the subject. For example, father, daughter,
siblings, parents.
EMPLOYMENT – Mentions of employment status including specific profession, such as
unemployed, retired, firefighter, student.
LIVING_STATUS – Mentions of the housing situation, including homeless, living with parents,
living alone, living with others.


Social
Entities



\n![Image](images/page1368_image1.png)

![Image](images/page1368_image2.png)

![Image](images/page1368_image3.png)

![Image](images/page1368_image4.png)

![Image](images/page1368_image5.png)
\nSUBSTANCE_USE – Mentions of use of legal or illegal drugs, tobacco or alcohol. For example,
smoking, drinking, or heroin use.
SUBSTANCE_USE_AMOUNT – Mentions of specific amounts of substance use. For example, a
pack (of cigarettes) or a few glasses (of wine).
TREATMENT_NAME – Therapeutic procedures. For example, knee replacement surgery, bone
marrow transplant, TAVI, diet.
How to call the Text Analytics for health


Treatment
Entities

Next steps
\n![Image](images/page1369_image1.png)

![Image](images/page1369_image2.png)

![Image](images/page1369_image3.png)
\nRelation extraction
06/21/2025
Text Analytics for health features relation extraction, which is used to identify meaningful
connections between concepts, or entities, mentioned in the text. For example, a "time of
condition" relation is found by associating a condition name with a time. Another example is a
"dosage of medication" relation, which is found by relating an extracted medication to its
extracted dosage. The following example shows how relations are expressed in the JSON
output.
Relation extraction output contains URI references and assigned roles of the entities of the
relation type. For example, in the following JSON:
JSON
７ Note
Relations referring to CONDITION may refer to either the DIAGNOSIS entity type or
the SYMPTOM_OR_SIGN entity type.
Relations referring to MEDICATION may refer to either the MEDICATION_NAME
entity type or the MEDICATION_CLASS entity type.
Relations referring to TIME may refer to either the TIME entity type or the DATE
entity type.
"relations": [
    {
        "relationType": "DosageOfMedication",
        "entities": [
            {
                "ref": "#/results/documents/0/entities/0",
                "role": "Dosage"
            },
            {
                "ref": "#/results/documents/0/entities/1",
                "role": "Medication"
            }
        ]
    },
    {
        "relationType": "RouteOfMedication",
        "entities": [
            {
                "ref": "#/results/documents/0/entities/1",
                "role": "Medication"
            },