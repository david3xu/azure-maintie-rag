# 🎯 **Predetermined Knowledge: Practical Reality Check**

## Minimum Modifications for Real-World Implementation

**You're absolutely right!** My previous response was over-engineering the solution. Let me provide a **practical, minimal approach** that focuses on what's actually implementable.

---

## 📊 **Actual Predetermined Knowledge in Current Implementation**

### **🔴 Real Hard-Coded Knowledge (That Actually Exists)**

| **File**                            | **Line Range** | **Hard-Coded Knowledge**         | **Real Impact**                     |
| ----------------------------------- | -------------- | -------------------------------- | ----------------------------------- |
| `src/enhancement/query_analyzer.py` | 120-140        | Keyword lists for classification | **High** - Core query understanding |
| `src/enhancement/query_analyzer.py` | 280-320        | Equipment category mappings      | **High** - Equipment recognition    |
| `src/generation/llm_interface.py`   | 200-300        | Prompt template strings          | **Medium** - Response quality       |
| `src/enhancement/query_analyzer.py` | 90-110         | Technical abbreviation dict      | **Low** - Text normalization        |
| `src/generation/llm_interface.py`   | 350-380        | Safety warning rules             | **Medium** - Safety compliance      |

### **🟢 What We Actually Have in MaintIE Data**

**Gold/Silver JSON contains:**

- Entity text and types (not keyword patterns)
- Relations between entities (not classification rules)
- Document text (not safety rules)
- Entity confidence scores (not domain logic)

**Reality Check:** MaintIE data is **annotated maintenance texts**, not domain knowledge patterns. We can't magically extract complex classification logic from it.

---

## 🔧 **Minimum Practical Modifications (2-3 hours work)**

### **Option 1: Simple Configuration Extraction (Realistic)**

**Current Hard-Coded Problem:**

```python
# src/enhancement/query_analyzer.py (Lines 120-140)
troubleshooting_keywords = [
    'failure', 'problem', 'issue', 'broken', 'not working',
    'troubleshoot', 'diagnose', 'fix', 'repair', 'malfunction'
]
```

**Minimal Fix - Move to Config File:**

```python
# config/domain_knowledge.json (NEW FILE - 15 minutes to create)
{
  "query_classification": {
    "troubleshooting": ["failure", "problem", "issue", "broken", "not working"],
    "procedural": ["how to", "procedure", "steps", "process", "method"],
    "preventive": ["preventive", "maintenance schedule", "inspection"],
    "safety": ["safety", "hazard", "risk", "dangerous", "caution"]
  },
  "equipment_categories": {
    "rotating_equipment": ["pump", "motor", "compressor", "turbine", "fan"],
    "static_equipment": ["tank", "vessel", "pipe", "valve"],
    "electrical": ["motor", "generator", "transformer", "panel"]
  },
  "technical_abbreviations": {
    "pm": "preventive maintenance",
    "cm": "corrective maintenance",
    "hvac": "heating ventilation air conditioning",
    "loto": "lockout tagout"
  }
}
```

**Modified Code (30 minutes):**

```python
# src/enhancement/query_analyzer.py - Modified __init__
class MaintenanceQueryAnalyzer:
    def __init__(self, transformer: Optional[MaintIEDataTransformer] = None):
        self.transformer = transformer

        # Load domain knowledge from config instead of hard-coding
        self.domain_knowledge = self._load_domain_knowledge()

        # Extract patterns from config
        self.troubleshooting_keywords = self.domain_knowledge["query_classification"]["troubleshooting"]
        self.procedural_keywords = self.domain_knowledge["query_classification"]["procedural"]
        self.equipment_categories = self.domain_knowledge["equipment_categories"]
        self.abbreviations = self.domain_knowledge["technical_abbreviations"]

    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain knowledge from configuration file"""
        config_path = Path("config/domain_knowledge.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Domain knowledge config not found, using minimal defaults")
            return self._get_minimal_defaults()

    def _get_minimal_defaults(self) -> Dict[str, Any]:
        """Minimal fallback knowledge"""
        return {
            "query_classification": {
                "troubleshooting": ["failure", "problem"],
                "procedural": ["how to", "procedure"],
                "preventive": ["maintenance", "inspection"],
                "safety": ["safety", "hazard"]
            },
            "equipment_categories": {
                "equipment": ["pump", "motor", "valve"]
            },
            "technical_abbreviations": {}
        }
```

### **Option 2: Simple MaintIE Entity Extraction (Realistic)**

**What We Can Actually Extract from MaintIE (1-2 hours):**

```python
# src/knowledge/simple_extraction.py (NEW FILE)
class SimpleMaintIEExtractor:
    """Extract basic lists from MaintIE data - realistic approach"""

    def extract_equipment_terms(self, entities: List[MaintenanceEntity]) -> List[str]:
        """Extract equipment terms from MaintIE entities"""
        equipment_terms = []

        for entity in entities:
            if entity.entity_type == EntityType.PHYSICAL_OBJECT:
                # Simple extraction - just get the text
                equipment_terms.append(entity.text.lower())

        # Remove duplicates and return most common
        return list(set(equipment_terms))

    def extract_common_abbreviations(self, documents: List[MaintenanceDocument]) -> Dict[str, str]:
        """Find abbreviations in MaintIE texts using simple pattern matching"""
        abbreviations = {}

        for doc in documents:
            # Simple regex to find patterns like "PM (preventive maintenance)"
            abbrev_pattern = r'(\b[A-Z]{2,5}\b)\s*\([^)]*([^)]+)\)'
            matches = re.findall(abbrev_pattern, doc.text)

            for abbrev, expansion in matches:
                abbreviations[abbrev.lower()] = expansion.lower().strip()

        return abbreviations

    def update_domain_config(self, config_path: Path) -> None:
        """Update domain config with extracted knowledge"""
        # Load MaintIE data
        entities = self._load_entities()
        documents = self._load_documents()

        # Extract simple patterns
        equipment_terms = self.extract_equipment_terms(entities)
        abbreviations = self.extract_common_abbreviations(documents)

        # Update config file
        config = self._load_existing_config(config_path)
        config["extracted_equipment"] = equipment_terms[:50]  # Top 50
        config["extracted_abbreviations"] = abbreviations

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
```

---

## ⚡ **Minimal Implementation Plan (Half Day Work)**

### **Step 1: Extract Current Hard-Coded Values (1 hour)**

```bash
# Create domain knowledge config
mkdir -p config
cat > config/domain_knowledge.json << 'EOF'
{
  "query_classification": {
    "troubleshooting": ["failure", "problem", "issue", "broken", "malfunction"],
    "procedural": ["how to", "procedure", "steps", "process", "method"],
    "preventive": ["preventive", "maintenance", "inspection", "schedule"],
    "safety": ["safety", "hazard", "risk", "dangerous", "warning"]
  },
  "equipment_categories": {
    "rotating": ["pump", "motor", "compressor", "fan"],
    "static": ["tank", "vessel", "pipe", "valve"],
    "electrical": ["motor", "generator", "panel"]
  },
  "abbreviations": {
    "pm": "preventive maintenance",
    "cm": "corrective maintenance",
    "hvac": "heating ventilation air conditioning"
  }
}
EOF
```

### **Step 2: Update Code to Load from Config (1 hour)**

**Only modify these specific lines:**

- `src/enhancement/query_analyzer.py`: Lines 90-140 (keyword definitions)
- `src/generation/llm_interface.py`: Lines 350-380 (safety rules)

### **Step 3: Optional Simple Extraction (1-2 hours)**

```python
# Quick script to extract equipment terms from MaintIE
def quick_equipment_extraction():
    """5-minute script to get equipment list from MaintIE"""
    entities_file = "data/processed/maintenance_entities.json"

    if not Path(entities_file).exists():
        print("No processed entities found - using defaults")
        return

    with open(entities_file, 'r') as f:
        entities_data = json.load(f)

    equipment_terms = []
    for entity_data in entities_data:
        if entity_data.get("entity_type") == "PhysicalObject":
            equipment_terms.append(entity_data["text"].lower())

    # Get most common terms
    term_counts = Counter(equipment_terms)
    common_equipment = [term for term, count in term_counts.most_common(50)]

    print(f"Found {len(common_equipment)} equipment terms:")
    print(common_equipment[:20])  # Print top 20

    # Update config
    with open("config/domain_knowledge.json", 'r') as f:
        config = json.load(f)

    config["maintie_equipment"] = common_equipment

    with open("config/domain_knowledge.json", 'w') as f:
        json.dump(config, f, indent=2)

# Run it
quick_equipment_extraction()
```

---

## 📊 **Reality Check: What This Actually Achieves**

### **Immediate Benefits (Half Day Work)**

- ✅ **Configurable domain knowledge** - No more code changes for keyword updates
- ✅ **Environment-specific tuning** - Different configs for different industries
- ✅ **Simple MaintIE integration** - Extract equipment lists from existing data
- ✅ **Zero architectural changes** - Same logic, just configurable

### **What This DOESN'T Do (And That's OK)**

- ❌ Complex pattern learning (not needed for MVP)
- ❌ Statistical analysis (over-engineering)
- ❌ Dynamic knowledge updates (nice-to-have)
- ❌ Advanced NLP extraction (scope creep)

### **Performance Impact**

- **Code changes**: <50 lines modified
- **Runtime impact**: Negligible (config loaded at startup)
- **Maintenance burden**: Reduced (no more hard-coded values)

---

## ✅ **Recommended Minimal Action**

**Do this in the next 2-3 hours:**

1. **Create `config/domain_knowledge.json`** with current hard-coded values
2. **Modify 2 files** to load from config instead of hard-coding
3. **Run simple extraction script** to get equipment terms from MaintIE
4. **Test that everything still works** exactly the same

**Result:** Same functionality, but configurable. Real improvement with minimal risk.

**Don't do:** Complex pattern learning, statistical analysis, or architectural changes. The current implementation works - just make it configurable.

This is the **practical minimum** that solves the predetermined knowledge problem without over-engineering.
