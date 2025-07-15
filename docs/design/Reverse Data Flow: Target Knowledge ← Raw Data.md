# Reverse Data Flow: Target Knowledge ← Raw Data

## What Domain Knowledge Do We Actually Need?
*Based on actual `query_analyzer.py` usage*

---

## **🎯 Target Domain Knowledge Types**

### **From `query_analyzer.py` - What's Actually Used:**

```python
# These are the actual domain knowledge patterns used in query processing:
self.equipment_patterns = self._build_equipment_patterns()      # Equipment detection
self.failure_patterns = self._build_failure_patterns()          # Failure mode detection
self.procedure_patterns = self._build_procedure_patterns()      # Procedure detection
self.component_patterns = self.domain_knowledge.get("component_patterns", {})
self.maintie_equipment = self.domain_knowledge.get("maintie_equipment", [])
self.extracted_abbreviations = self.domain_knowledge.get("extracted_abbreviations", {})

# Query classification patterns:
self.troubleshooting_keywords = self.domain_knowledge.get("query_classification", {}).get("troubleshooting", [])
self.procedural_keywords = self.domain_knowledge.get("query_classification", {}).get("procedural", [])
```

---

## **📋 Required Domain Knowledge Structure**

### **For Query Processing (SQL-like lookup):**

```json
{
  "equipment_patterns": {
    "pump": "pump",
    "motor": "motor",
    "bearing": "bearing"
  },
  "failure_patterns": {
    "failure": "failure",
    "leak": "leak",
    "vibration": "vibration"
  },
  "procedure_patterns": {
    "maintenance": "maintenance",
    "repair": "repair",
    "inspect": "inspect"
  },
  "component_patterns": {
    "bearing": "bearing",
    "seal": "seal",
    "valve": "valve"
  },
  "extracted_abbreviations": {
    "pm": "preventive maintenance",
    "cm": "corrective maintenance"
  },
  "query_classification": {
    "troubleshooting": ["failure", "problem", "broken"],
    "procedural": ["how to", "procedure", "steps"],
    "preventive": ["maintenance", "inspection", "schedule"]
  }
}
```

**Usage in Query Processing:**
```python
# When user asks: "pump bearing failure troubleshooting"
equipment = self._extract_equipment_entities(query)  # Uses equipment_patterns
failures = self._extract_failure_entities(query)     # Uses failure_patterns
query_type = self._classify_query_type(query)        # Uses query_classification
```

---

## **⬅️ Raw Data Mapping**

### **What Raw Data Supports Each Knowledge Type:**

#### **1. Equipment Patterns ← PhysicalObject Entities**

**Raw Data (silver_release.json):**
```json
{
  "text": "inspect bearing on motor",
  "entities": [
    {"start": 1, "end": 2, "type": "PhysicalObject"},  // "bearing"
    {"start": 3, "end": 4, "type": "PhysicalObject"}   // "motor"
  ]
}
```

**Extraction Logic:**
```python
def extract_equipment_terms(self, entities):
    equipment_terms = []
    for entity in entities:
        if entity.entity_type == EntityType.PHYSICAL_OBJECT:
            equipment_terms.append(entity.text.lower())  # "bearing", "motor"
    return equipment_terms
```

**Result:** `equipment_patterns: {"bearing": "bearing", "motor": "motor"}`

#### **2. Procedure Patterns ← Activity Entities**

**Raw Data:**
```json
{
  "text": "inspect motor and repair seal",
  "entities": [
    {"start": 0, "end": 1, "type": "Activity"},  // "inspect"
    {"start": 3, "end": 4, "type": "Activity"}   // "repair"
  ]
}
```

**Extraction Logic:**
```python
def extract_procedure_terms(self, documents):
    procedure_terms = []
    procedure_keywords = ['maintenance', 'inspection', 'repair']
    for doc in documents:
        for keyword in procedure_keywords:
            if keyword in doc.text.lower():
                procedure_terms.append(keyword)
    return procedure_terms
```

**Result:** `procedure_patterns: {"inspect": "inspect", "repair": "repair"}`

#### **3. Failure Patterns ← Text Analysis**

**Raw Data:**
```json
{
  "text": "motor bearing failure causing vibration",
  "entities": [
    {"start": 2, "end": 3, "type": "Process"}  // "failure"
  ]
}
```

**Extraction Logic:**
```python
def extract_failure_terms(self, documents):
    failure_terms = []
    failure_keywords = ['failure', 'leak', 'vibration', 'noise']
    for doc in documents:
        for keyword in failure_keywords:
            if keyword in doc.text.lower():
                failure_terms.append(keyword)
    return failure_terms
```

**Result:** `failure_patterns: {"failure": "failure", "vibration": "vibration"}`

#### **4. Abbreviations ← Text Pattern Matching**

**Raw Data:**
```json
{
  "text": "schedule PM (preventive maintenance) for pumps"
}
```

**Extraction Logic:**
```python
def extract_common_abbreviations(self, documents):
    abbreviations = {}
    abbrev_pattern = r'(\b[A-Z]{2,5}\b)\s*\([^)]*([^)]+)\)'
    for doc in documents:
        matches = re.findall(abbrev_pattern, doc.text)
        for abbrev, expansion in matches:
            abbreviations[abbrev.lower()] = expansion.lower().strip()
    return abbreviations
```

**Result:** `extracted_abbreviations: {"pm": "preventive maintenance"}`

#### **5. Query Classification ← Co-occurrence Analysis**

**Raw Data Analysis:**
```json
// Documents containing "failure" + "problem" patterns
{"text": "pump failure problem troubleshooting"}
{"text": "motor failure issue diagnosis"}

// Documents containing "how to" + "procedure" patterns
{"text": "how to maintain bearing procedure"}
{"text": "procedure for valve maintenance"}
```

**Extraction Logic:**
```python
def _discover_query_types(self, documents):
    query_patterns = defaultdict(list)
    for doc in documents:
        text_lower = doc.text.lower()
        if any(word in text_lower for word in ['problem', 'issue', 'failure']):
            query_patterns['troubleshooting'].extend(re.findall(r'\b\w+\b', text_lower))
        if any(word in text_lower for word in ['how', 'procedure', 'steps']):
            query_patterns['procedural'].extend(re.findall(r'\b\w+\b', text_lower))
    return query_patterns
```

**Result:**
```json
{
  "troubleshooting": ["failure", "problem", "issue", "broken"],
  "procedural": ["how", "procedure", "steps", "process"]
}
```

---

## **🔄 Complete Reverse Mapping**

### **Target Knowledge ← Raw Data Sources:**

| Target Knowledge Type | Raw Data Source | Extraction Method |
|----------------------|-----------------|-------------------|
| **equipment_patterns** | PhysicalObject entities | Entity type filtering |
| **failure_patterns** | Text + Process entities | Keyword pattern matching |
| **procedure_patterns** | Activity entities + text | Activity detection + keywords |
| **component_patterns** | PhysicalObject sub-types | Entity hierarchy analysis |
| **extracted_abbreviations** | Text patterns | Regex: "ABC (full form)" |
| **query_classification** | Text co-occurrence | Statistical pattern discovery |

### **Data Completeness Check:**

**From Raw Data (3 JSON files):**
- ✅ **PhysicalObject entities**: 2,500+ equipment terms
- ✅ **Activity entities**: 800+ procedure terms
- ✅ **Process entities**: 400+ failure modes
- ✅ **Text patterns**: 2,500+ documents for abbreviations
- ✅ **Hierarchy info**: scheme.json provides type relationships

**Missing/Weak Areas:**
- ⚠️ **Safety classifications**: Limited safety-related annotations
- ⚠️ **Tool mappings**: No tool-equipment relationships in raw data
- ⚠️ **Urgency patterns**: No explicit urgency indicators

---

## **💡 Architecture Insight**

### **What We're Building:**
A **lookup system** where:
- User query → Pattern matching → Entity extraction → Response generation
- Similar to SQL: `SELECT response WHERE equipment LIKE '%pump%' AND failure LIKE '%leak%'`

### **Why This Raw Data Works:**
- **Rich annotations**: Every maintenance scenario has entity/relation labels
- **Diverse coverage**: 2,500+ real maintenance requests
- **Structured hierarchy**: scheme.json provides semantic relationships
- **Pattern density**: Enough examples to discover reliable patterns

### **Simple Architecture:**
```
Raw JSON → Extract Patterns → Build Lookup Tables → Process Queries
```

**Result:** The 3 raw JSON files contain sufficient data to build all required domain knowledge patterns for maintenance query processing.