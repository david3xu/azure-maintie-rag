# 🔍 **Hard-Coded Values Analysis: MaintIE-Enhanced RAG Implementation**

## Comprehensive Review & Configuration Recommendations

**Excellent question!** Yes, there are several hard-coded values throughout the implementation that should be made configurable for production use. Here's a systematic analysis:

---

## 📊 **Hard-Coded Values by Category**

### **🔴 Critical Hard-Coded Values (Should be configurable)**

| **File**                              | **Hard-Coded Value**  | **Current Code**                                       | **Impact**                |
| ------------------------------------- | --------------------- | ------------------------------------------------------ | ------------------------- |
| **src/knowledge/data_transformer.py** | File names            | `"gold_release.json"`, `"silver_release.json"`         | High - Different datasets |
| **src/knowledge/data_transformer.py** | Confidence thresholds | `confidence_base=0.9`, `0.7`                           | Medium - Model tuning     |
| **src/enhancement/query_analyzer.py** | Keyword lists         | `troubleshooting_keywords = ['failure', 'problem'...]` | High - Domain adaptation  |
| **src/generation/llm_interface.py**   | Prompt templates      | `"""You are helping with..."""`                        | High - Response quality   |
| **src/retrieval/vector_search.py**    | Batch size            | `batch_size=32`                                        | Medium - Performance      |
| **api/endpoints/query.py**            | Validation limits     | `min_length=3, max_length=500`                         | Medium - Use cases        |

### **🟡 Medium Impact Hard-Coded Values**

| **File**                              | **Hard-Coded Value** | **Current Code**                                 | **Reason to Configure**                   |
| ------------------------------------- | -------------------- | ------------------------------------------------ | ----------------------------------------- |
| **src/enhancement/query_analyzer.py** | Result limits        | `neighbors[:5]`, `related[:15]`                  | Different use cases need different limits |
| **src/retrieval/vector_search.py**    | Index type           | `IndexFlatIP`                                    | Different deployment needs                |
| **src/generation/llm_interface.py**   | Model parameters     | `top_p=0.9, frequency_penalty=0.1`               | Fine-tuning requirements                  |
| **src/pipeline/enhanced_rag.py**      | Fusion weights       | Referenced from settings but logic is hard-coded | Algorithm experimentation                 |

### **🟢 Acceptable Hard-Coded Values**

| **File**                              | **Hard-Coded Value** | **Reason Acceptable**     |
| ------------------------------------- | -------------------- | ------------------------- |
| **src/models/maintenance_models.py**  | Enum values          | Domain-specific constants |
| **api/main.py**                       | HTTP status codes    | Standard HTTP conventions |
| **src/knowledge/data_transformer.py** | JSON structure keys  | MaintIE schema specific   |

---

## 🛠️ **Recommended Configuration Improvements**

### **1. Enhanced Configuration File**

```python
# config/advanced_settings.py
"""
Extended configuration for MaintIE Enhanced RAG
Addresses all hard-coded values with environment overrides
"""

from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseSettings, Field


class AdvancedSettings(BaseSettings):
    """Extended settings covering all configurable parameters"""

    # Data Processing Settings
    gold_data_filename: str = Field(default="gold_release.json", env="GOLD_DATA_FILENAME")
    silver_data_filename: str = Field(default="silver_release.json", env="SILVER_DATA_FILENAME")
    gold_confidence_base: float = Field(default=0.9, env="GOLD_CONFIDENCE_BASE")
    silver_confidence_base: float = Field(default=0.7, env="SILVER_CONFIDENCE_BASE")

    # Query Analysis Settings
    max_related_entities: int = Field(default=15, env="MAX_RELATED_ENTITIES")
    max_neighbors: int = Field(default=5, env="MAX_NEIGHBORS")
    concept_expansion_limit: int = Field(default=10, env="CONCEPT_EXPANSION_LIMIT")

    # Retrieval Settings
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    faiss_index_type: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Generation Settings
    llm_top_p: float = Field(default=0.9, env="LLM_TOP_P")
    llm_frequency_penalty: float = Field(default=0.1, env="LLM_FREQUENCY_PENALTY")
    llm_presence_penalty: float = Field(default=0.1, env="LLM_PRESENCE_PENALTY")

    # API Validation Settings
    query_min_length: int = Field(default=3, env="QUERY_MIN_LENGTH")
    query_max_length: int = Field(default=500, env="QUERY_MAX_LENGTH")
    max_results_limit: int = Field(default=50, env="MAX_RESULTS_LIMIT")

    # Domain Knowledge Settings
    troubleshooting_keywords: List[str] = Field(
        default=[
            'failure', 'problem', 'issue', 'broken', 'not working',
            'troubleshoot', 'diagnose', 'fix', 'repair', 'malfunction'
        ],
        env="TROUBLESHOOTING_KEYWORDS"
    )

    procedural_keywords: List[str] = Field(
        default=[
            'how to', 'procedure', 'steps', 'process', 'method',
            'instructions', 'guide', 'manual', 'protocol'
        ],
        env="PROCEDURAL_KEYWORDS"
    )

    safety_keywords: List[str] = Field(
        default=[
            'safety', 'hazard', 'risk', 'dangerous', 'caution',
            'warning', 'lockout', 'ppe', 'procedure'
        ],
        env="SAFETY_KEYWORDS"
    )

    # Equipment Categories
    equipment_categories: Dict[str, List[str]] = Field(
        default={
            'rotating_equipment': ['pump', 'motor', 'compressor', 'turbine', 'fan'],
            'static_equipment': ['tank', 'vessel', 'pipe', 'valve'],
            'electrical': ['motor', 'generator', 'transformer', 'panel'],
            'hvac': ['fan', 'damper', 'coil', 'duct', 'filter'],
            'instrumentation': ['sensor', 'transmitter', 'gauge', 'indicator']
        }
    )

    # Abbreviation Expansions
    technical_abbreviations: Dict[str, str] = Field(
        default={
            'pm': 'preventive maintenance',
            'cm': 'corrective maintenance',
            'hvac': 'heating ventilation air conditioning',
            'loto': 'lockout tagout',
            'sop': 'standard operating procedure',
            'rca': 'root cause analysis'
        }
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global advanced settings instance
advanced_settings = AdvancedSettings()
```

### **2. Configurable Prompt Templates**

```python
# config/prompt_templates.py
"""
Configurable prompt templates for different query types
"""

from typing import Dict
from config.advanced_settings import advanced_settings


class ConfigurablePromptTemplates:
    """Manage prompt templates with configuration"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from configuration or files"""

        # Default templates (can be overridden by config files)
        default_templates = {
            "troubleshooting": """
You are helping with a maintenance troubleshooting issue. Please provide a comprehensive troubleshooting response for the following:

Query: {query}
Equipment Category: {equipment_category}
Identified Entities: {entities}
Related Concepts: {expanded_concepts}
Urgency Level: {urgency}

Relevant Documentation:
{context}

Safety Considerations:
{safety_considerations}

Please provide:
1. A systematic troubleshooting approach
2. Most likely causes ranked by probability
3. Step-by-step diagnostic procedures
4. Required tools and materials
5. Safety precautions specific to this equipment
6. When to escalate to specialized technicians

Format your response with clear headings and actionable steps.
""",

            "procedural": """
You are providing maintenance procedure guidance. Please create a detailed procedural response for:

Query: {query}
Equipment Category: {equipment_category}
Identified Entities: {entities}
Related Concepts: {expanded_concepts}

Relevant Documentation:
{context}

Safety Considerations:
{safety_considerations}

Please provide:
1. Step-by-step procedure with clear instructions
2. Required tools, parts, and materials
3. Safety precautions and PPE requirements
4. Quality checks and verification steps
5. Common pitfalls to avoid
6. Estimated time and skill level required

Use numbered steps and include safety reminders throughout.
"""
        }

        # TODO: Add ability to load templates from external files
        # templates_dir = Path("config/templates")
        # if templates_dir.exists():
        #     for template_file in templates_dir.glob("*.txt"):
        #         template_name = template_file.stem
        #         with open(template_file, 'r') as f:
        #             default_templates[template_name] = f.read()

        return default_templates

    def get_template(self, template_type: str) -> str:
        """Get template by type"""
        return self.templates.get(template_type, self.templates["general"])

    def update_template(self, template_type: str, template_content: str) -> None:
        """Update template dynamically"""
        self.templates[template_type] = template_content


# Global template manager
template_manager = ConfigurablePromptTemplates()
```

### **3. Updated Code with Configuration Integration**

```python
# src/enhancement/query_analyzer.py - Updated sections
class MaintenanceQueryAnalyzer:
    def __init__(self, transformer: Optional[MaintIEDataTransformer] = None):
        """Initialize analyzer with configurable parameters"""
        self.transformer = transformer
        self.config = advanced_settings  # Use advanced configuration

        # Load configurable keyword lists
        self.troubleshooting_keywords = self.config.troubleshooting_keywords
        self.procedural_keywords = self.config.procedural_keywords
        self.safety_keywords = self.config.safety_keywords

        # Load configurable equipment categories
        self.equipment_categories = self.config.equipment_categories

        # Load configurable abbreviations
        self.abbreviations = self.config.technical_abbreviations

        # Configurable limits
        self.max_related_entities = self.config.max_related_entities
        self.max_neighbors = self.config.max_neighbors

    def _normalize_query(self, query: str) -> str:
        """Normalize query text using configurable abbreviations"""
        normalized = query.lower().strip()

        # Use configurable abbreviations
        for abbr, expansion in self.abbreviations.items():
            normalized = re.sub(r'\b' + abbr + r'\b', expansion, normalized)

        return normalized

    def _find_related_entities(self, entities: List[str]) -> List[str]:
        """Find entities related using configurable limits"""
        related = set()

        if self.knowledge_graph:
            for entity in entities:
                entity_id = self._find_entity_id(entity)
                if entity_id:
                    try:
                        neighbors = nx.single_source_shortest_path_length(
                            self.knowledge_graph, entity_id, cutoff=2
                        )
                        for neighbor_id, distance in neighbors.items():
                            if distance > 0:
                                neighbor_text = self.knowledge_graph.nodes[neighbor_id].get('text', neighbor_id)
                                related.add(neighbor_text)
                    except:
                        continue

        # Use configurable limit instead of hard-coded [:15]
        return list(related)[:self.max_related_entities]
```

```python
# src/generation/llm_interface.py - Updated sections
class MaintenanceLLMInterface:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM interface with configurable parameters"""
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature

        # Use configurable advanced settings
        self.config = advanced_settings
        self.template_manager = template_manager

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with configurable parameters"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert maintenance engineer with 20+ years of experience in industrial equipment maintenance. Provide accurate, practical, and safety-focused maintenance guidance."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.config.llm_top_p,  # Now configurable
                frequency_penalty=self.config.llm_frequency_penalty,  # Now configurable
                presence_penalty=self.config.llm_presence_penalty  # Now configurable
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise

    def _build_maintenance_prompt(self, enhanced_query: EnhancedQuery, search_results: List[SearchResult]) -> str:
        """Build maintenance-specific prompt using configurable templates"""

        query_type = enhanced_query.analysis.query_type.value

        # Use configurable template manager
        template = self.template_manager.get_template(query_type)

        # Rest of the method remains the same...
```

---

## 📋 **Configuration File Examples**

### **production.env**

```bash
# Production Configuration for MaintIE Enhanced RAG

# Data Settings
GOLD_DATA_FILENAME=maintie_gold_v2.json
SILVER_DATA_FILENAME=maintie_silver_v2.json
GOLD_CONFIDENCE_BASE=0.95
SILVER_CONFIDENCE_BASE=0.75

# Performance Settings
EMBEDDING_BATCH_SIZE=64
MAX_RELATED_ENTITIES=20
CONCEPT_EXPANSION_LIMIT=15

# API Limits
QUERY_MAX_LENGTH=1000
MAX_RESULTS_LIMIT=100

# LLM Settings
LLM_TOP_P=0.85
LLM_FREQUENCY_PENALTY=0.2
LLM_PRESENCE_PENALTY=0.15

# Custom Keywords (JSON format)
TROUBLESHOOTING_KEYWORDS=["failure","malfunction","breakdown","defect","error","fault"]
```

### **development.env**

```bash
# Development Configuration

# Lower performance settings for development
EMBEDDING_BATCH_SIZE=16
MAX_RELATED_ENTITIES=10
CONCEPT_EXPANSION_LIMIT=8

# Relaxed validation for testing
QUERY_MIN_LENGTH=1
QUERY_MAX_LENGTH=1000

# More verbose LLM settings
LLM_TEMPERATURE=0.5
LLM_TOP_P=0.9
```

---

## 🚀 **Implementation Priority**

### **🔴 High Priority (Immediate)**

1. **Prompt templates** - Critical for response quality
2. **Keyword lists** - Essential for domain adaptation
3. **File paths** - Required for different datasets
4. **API validation limits** - Needed for different use cases

### **🟡 Medium Priority (Next Sprint)**

1. **Model parameters** - Important for fine-tuning
2. **Performance thresholds** - Optimization requirements
3. **Batch sizes** - Deployment-specific settings

### **🟢 Low Priority (Future Enhancement)**

1. **Logging formats** - Operational improvements
2. **Error messages** - Localization support
3. **Default responses** - Customization features

---

## ✅ **Recommended Action Plan**

1. **Create `config/advanced_settings.py`** with all configurable parameters
2. **Update core modules** to use configuration instead of hard-coded values
3. **Add environment-specific** `.env` files for different deployments
4. **Implement template management** system for prompt customization
5. **Add configuration validation** to prevent invalid settings
6. **Document all configuration options** for deployment teams

This approach will make the system much more flexible, maintainable, and suitable for different deployment environments and use cases.
