# Data Flow: Concrete Input/Output at Each Step

## Professional Architecture: Clear Data Transformations
*Based on actual codebase - shows exactly what goes in and out*

---

## **🔄 Data Flow Architecture**

```
Raw JSON Files → Structured Objects → Pattern Lists → Domain Config → Query Intelligence
```

---

## **Step 1: Raw Data Input** 📁

### **INPUT: 3 Raw JSON Files**

**silver_release.json** (2,500+ entries)
```json
{
  "text": "inspect bearing on bend pulley",
  "tokens": ["inspect", "bearing", "on", "bend", "pulley"],
  "entities": [
    {"start": 0, "end": 1, "type": "Activity"},
    {"start": 1, "end": 2, "type": "PhysicalObject"},
    {"start": 3, "end": 5, "type": "PhysicalObject"}
  ],
  "relations": [
    {"head": 0, "tail": 1, "type": "hasPatient"}
  ]
}
```

**gold_release.json** (High confidence annotations)
```json
{
  "text": "replace faulty motor bearing seal",
  "tokens": ["replace", "faulty", "motor", "bearing", "seal"],
  "entities": [
    {"start": 0, "end": 1, "type": "Activity"},
    {"start": 2, "end": 3, "type": "PhysicalObject"},
    {"start": 3, "end": 4, "type": "PhysicalObject"}
  ]
}
```

**scheme.json** (Type hierarchy)
```json
{
  "entity": [
    {
      "name": "PhysicalObject",
      "fullname": "PhysicalObject",
      "children": [
        {"name": "DrivingObject", "fullname": "PhysicalObject/DrivingObject"}
      ]
    }
  ]
}
```

**WHY NEEDED**: Raw annotations provide maintenance scenarios, entity labels, and type hierarchy for processing.

---

## **Step 2: Data Transformation** 🔄

### **INPUT**: Raw JSON files from Step 1
### **PROCESS**: `data_transformer.py::extract_maintenance_knowledge()`

```python
# What actually happens:
raw_data = self.load_raw_data()  # Loads all 3 JSON files
gold_stats = self._process_dataset(raw_data["gold"], confidence_base=0.9)
silver_stats = self._process_dataset(raw_data["silver"], confidence_base=0.7)
```

### **OUTPUT**: 2 Structured Files

**maintenance_entities.json**
```json
{
  "doc_0_entity_1_2": {
    "entity_id": "doc_0_entity_1_2",
    "text": "bearing",
    "entity_type": "PhysicalObject",
    "confidence": 0.9,
    "context": "inspect bearing on bend pulley",
    "metadata": {
      "start": 1,
      "end": 2,
      "document_id": "doc_0"
    }
  },
  "doc_0_entity_3_5": {
    "entity_id": "doc_0_entity_3_5",
    "text": "bend pulley",
    "entity_type": "PhysicalObject",
    "confidence": 0.9
  }
}
```

**maintenance_documents.json**
```json
[
  {
    "document_id": "doc_0",
    "text": "inspect bearing on bend pulley",
    "entities": ["doc_0_entity_1_2", "doc_0_entity_3_5"],
    "relations": [
      {
        "source_entity": "doc_0_entity_0_1",
        "target_entity": "doc_0_entity_1_2",
        "relation_type": "hasPatient"
      }
    ]
  }
]
```

**WHY NEEDED**: Converts raw annotations into structured objects that can be programmatically processed for pattern extraction.

---

## **Step 3: Pattern Extraction** 🔍

### **INPUT**: Structured files from Step 2
### **PROCESS**: `simple_extraction.py::update_domain_config()`

```python
# What actually happens:
entities = self._load_entities()  # From maintenance_entities.json
documents = self._load_documents()  # From maintenance_documents.json

equipment_terms = self.extract_equipment_terms(entities)
abbreviations = self.extract_common_abbreviations(documents)
```

### **EXTRACTION LOGIC**:

**Equipment Terms** (from entities)
```python
for entity in entities:
    if entity.entity_type == EntityType.PHYSICAL_OBJECT:
        equipment_terms.append(entity.text.lower())
```

**Abbreviations** (from document text)
```python
abbrev_pattern = r'(\b[A-Z]{2,5}\b)\s*\([^)]*([^)]+)\)'
matches = re.findall(abbrev_pattern, doc.text)
```

### **OUTPUT**: Pattern Lists

```json
{
  "extracted_equipment": ["bearing", "pulley", "motor", "pump", "valve"],
  "extracted_abbreviations": {"pm": "preventive maintenance"},
  "extracted_failure_terms": ["failure", "leak", "vibration"],
  "extracted_procedure_terms": ["inspect", "replace", "repair"]
}
```

**WHY NEEDED**: Discovers actual maintenance patterns from real data rather than using assumptions.

---

## **Step 4: Domain Knowledge Assembly** 📊

### **INPUT**: Pattern lists from Step 3
### **PROCESS**: `extract_knowledge.py::update_domain_config()`

```python
# What actually happens:
config = self._load_existing_config()
config["maintie_equipment"] = equipment_terms[:50]
config["extracted_abbreviations"] = abbreviations
```

### **OUTPUT**: domain_knowledge.json

```json
{
  "maintie_equipment": [
    "bearing", "pulley", "motor", "pump", "valve", "seal"
  ],
  "extracted_abbreviations": {
    "pm": "preventive maintenance",
    "cm": "corrective maintenance"
  },
  "extracted_failure_terms": [
    "failure", "leak", "vibration", "wear", "noise"
  ],
  "extracted_procedure_terms": [
    "inspect", "maintenance", "repair", "replacement"
  ],
  "equipment_patterns": {
    "pump": "pump",
    "motor": "motor",
    "bearing": "bearing"
  },
  "extraction_metadata": {
    "source": "raw_data_processing",
    "entities_processed": 2847,
    "documents_processed": 2500
  }
}
```

**WHY NEEDED**: Creates a single configuration file with all discovered maintenance knowledge for query processing.

---

## **Step 5: Query Intelligence** 🔧

### **INPUT**: domain_knowledge.json from Step 4
### **PROCESS**: `query_analyzer.py::analyze_query()`

```python
# What actually happens:
self.domain_knowledge = self._load_domain_knowledge()
self.equipment_patterns = self._build_equipment_patterns()
entities = self._extract_entities(query)
```

### **USAGE EXAMPLE**:

**User Query**: "pump bearing failure troubleshooting"

**Processing**:
1. Load extracted equipment: `["bearing", "pump"]`
2. Load extracted failure terms: `["failure"]`
3. Match patterns: `pump=equipment, bearing=equipment, failure=failure_mode`
4. Classify: `troubleshooting` (high urgency)

**OUTPUT**: QueryAnalysis object
```python
QueryAnalysis(
    original_query="pump bearing failure troubleshooting",
    query_type=QueryType.TROUBLESHOOTING,
    entities=["pump", "bearing", "failure"],
    urgency="high",
    confidence=0.9
)
```

**WHY NEEDED**: Converts natural language maintenance queries into structured requests for the RAG system.

---

## **⚠️ Current Implementation Gaps**

### **Gap 1: Hardcoded Fallbacks**
**Location**: `query_analyzer.py::_get_default_domain_knowledge()`
**Problem**: Falls back to hardcoded values when domain_knowledge.json missing
**Impact**: Breaks data traceability

### **Gap 2: scheme.json Under-utilization**
**Current**: Only uses basic type mapping
**Available**: Rich hierarchy paths, metadata, examples
**Missing**: Hierarchy-based pattern discovery

### **Gap 3: Mixed Architecture**
**Current**: Base structure + extracted patterns
**Target**: Pure extracted patterns only

---

## **Professional Architecture Benefits**

### **Simple Lifecycle**:
```
Raw Data → Transform → Extract → Assemble → Use
```

### **Clear Separation**:
- **Data Layer**: JSON files
- **Processing Layer**: Python objects
- **Pattern Layer**: Lists and dictionaries
- **Config Layer**: Single domain knowledge file
- **Application Layer**: Query analysis

### **Traceability**:
Every pattern can be traced back to specific raw data entries

### **Maintainability**:
Each step has clear input/output contracts

### **Scalability**:
New raw data automatically updates all downstream processing

**Result**: Professional maintenance intelligence system with clear data flow and full traceability from raw annotations to query understanding.