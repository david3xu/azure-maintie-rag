#!/usr/bin/env python3
"""
Adaptive Context Generator
Automatically generates context-aware prompts based on input data characteristics
"""

import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

@dataclass
class DataProfile:
    """Profile of input data characteristics"""
    domain: str
    entity_patterns: List[str]
    relationship_patterns: List[str]
    terminology: List[str]
    context_examples: List[str]
    quality_criteria: List[str]

class AdaptiveContextGenerator:
    """Generate context-aware prompts based on data analysis"""
    
    def __init__(self):
        self.known_domains = {
            "maintenance": self._get_maintenance_profile(),
            "medical": self._get_medical_profile(),
            "financial": self._get_financial_profile(),
            "legal": self._get_legal_profile(),
            "manufacturing": self._get_manufacturing_profile()
        }
    
    def analyze_data_characteristics(self, sample_texts: List[str], domain_hint: Optional[str] = None) -> DataProfile:
        """Analyze input data to determine domain and characteristics"""
        
        print(f"🔍 Analyzing {len(sample_texts)} sample texts for domain characteristics...")
        
        # Extract key terms and patterns
        all_text = " ".join(sample_texts).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = Counter(words)
        
        # Domain detection
        detected_domain = self._detect_domain(word_freq, domain_hint)
        
        # Extract patterns
        entity_patterns = self._extract_entity_patterns(sample_texts)
        relationship_patterns = self._extract_relationship_patterns(sample_texts)
        terminology = [word for word, freq in word_freq.most_common(50) if len(word) > 3]
        
        # Generate context examples from actual data
        context_examples = sample_texts[:5]  # Use first 5 as examples
        
        # Define quality criteria based on domain
        quality_criteria = self._get_quality_criteria(detected_domain)
        
        profile = DataProfile(
            domain=detected_domain,
            entity_patterns=entity_patterns,
            relationship_patterns=relationship_patterns,
            terminology=terminology,
            context_examples=context_examples,
            quality_criteria=quality_criteria
        )
        
        print(f"✅ Domain detected: {detected_domain}")
        print(f"📊 Key patterns: {len(entity_patterns)} entity types, {len(relationship_patterns)} relationship types")
        
        return profile
    
    def _detect_domain(self, word_freq: Counter, domain_hint: Optional[str]) -> str:
        """Detect domain based on word frequency analysis"""
        
        # Domain keywords
        domain_keywords = {
            "maintenance": ["repair", "replace", "service", "equipment", "component", "bearing", "hose", "valve", "engine", "pump"],
            "medical": ["patient", "diagnosis", "treatment", "symptom", "medication", "doctor", "hospital", "therapy"],
            "financial": ["account", "transaction", "payment", "balance", "interest", "loan", "credit", "investment"],
            "legal": ["contract", "agreement", "liability", "clause", "defendant", "plaintiff", "court", "evidence"],
            "manufacturing": ["production", "assembly", "quality", "defect", "batch", "process", "specification", "tolerance"]
        }
        
        # Score each domain
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(word_freq.get(keyword, 0) for keyword in keywords)
            domain_scores[domain] = score
        
        # Use hint if provided and reasonable
        if domain_hint and domain_hint in self.known_domains:
            detected_domain = domain_hint
        else:
            detected_domain = max(domain_scores, key=domain_scores.get)
        
        return detected_domain
    
    def _extract_entity_patterns(self, texts: List[str]) -> List[str]:
        """Extract common entity patterns from sample texts"""
        
        # Common entity patterns by type
        equipment_patterns = re.findall(r'\b(?:air conditioner|compressor|engine|pump|motor|generator)\b', " ".join(texts), re.IGNORECASE)
        component_patterns = re.findall(r'\b(?:thermostat|valve|hose|bearing|filter|sensor|switch)\b', " ".join(texts), re.IGNORECASE)
        problem_patterns = re.findall(r'\b(?:not working|unserviceable|broken|cracked|leaking|blown|seized)\b', " ".join(texts), re.IGNORECASE)
        
        patterns = []
        if equipment_patterns:
            patterns.append("primary_equipment")
        if component_patterns:
            patterns.append("components")
        if problem_patterns:
            patterns.append("problems")
        
        return patterns or ["generic_entities"]
    
    def _extract_relationship_patterns(self, texts: List[str]) -> List[str]:
        """Extract common relationship patterns from sample texts"""
        
        patterns = []
        all_text = " ".join(texts).lower()
        
        # Structural relationships
        if any(word in all_text for word in ["has", "contains", "part of", "component"]):
            patterns.append("structural_relationships")
        
        # Problem relationships  
        if any(word in all_text for word in ["problem", "issue", "fault", "failure"]):
            patterns.append("problem_relationships")
        
        # Action relationships
        if any(word in all_text for word in ["replace", "repair", "service", "fix"]):
            patterns.append("action_relationships")
        
        return patterns or ["generic_relationships"]
    
    def _get_quality_criteria(self, domain: str) -> List[str]:
        """Get quality criteria based on domain"""
        
        criteria_map = {
            "maintenance": [
                "Extract specific equipment and component names",
                "Identify clear problem descriptions", 
                "Capture maintenance actions required",
                "Preserve location and condition details"
            ],
            "medical": [
                "Extract specific medical conditions and symptoms",
                "Identify treatments and medications",
                "Capture patient demographics and history",
                "Preserve clinical context and severity"
            ],
            "financial": [
                "Extract account numbers and transaction details",
                "Identify financial products and services",
                "Capture monetary amounts and dates",
                "Preserve risk and compliance context"
            ],
            "legal": [
                "Extract legal entities and parties",
                "Identify contract terms and obligations",
                "Capture dates and jurisdictions",
                "Preserve legal precedents and citations"
            ],
            "manufacturing": [
                "Extract product specifications and tolerances",
                "Identify process steps and quality metrics",
                "Capture batch numbers and timestamps",
                "Preserve quality control context"
            ]
        }
        
        return criteria_map.get(domain, ["Extract meaningful entities with full context"])
    
    def generate_entity_extraction_prompt(self, profile: DataProfile) -> str:
        """Generate adaptive entity extraction prompt"""
        
        base_template = """# Adaptive Entity Extraction Template
# Domain: {domain}
# Auto-generated based on data analysis

system:
You are an expert {domain} analyst with deep understanding of {domain} terminology, processes, and relationships. Your expertise helps identify all meaningful entities that {domain} professionals need to track and understand.

user:
## Context: {domain} Knowledge Extraction

**Your Role**: You're helping build an intelligent {domain} knowledge system that will help professionals quickly find solutions and make informed decisions.

**Domain Knowledge**: Based on analysis of the provided data, you understand that {domain} records contain:
{domain_patterns}

**Quality Standards**: Extract entities that would help a {domain} professional understand:
{quality_criteria}

**Examples from Your Data**:
{context_examples}

## Records to Analyze:

{{{{ for text in texts }}}}
**Record {{{{ loop.index }}}}: {{{{ text }}}}
{{{{ endfor }}}}

## Required Output Format:

For each record, extract all meaningful entities with their context. Return a JSON array where each object represents an entity found in the text:

```json
[
  {{
    "entity_id": "entity_1",
    "text": "specific entity from text",
    "entity_type": "domain_appropriate_type",
    "confidence": 0.95,
    "context": "full context where entity appears",
    "source_record": 1,
    "semantic_role": "role in domain context",
    "domain_relevance": "why this entity matters in {domain}"
  }}
]
```

**Guidelines for Quality Extraction**:
- Extract entities exactly as they appear in the records
- Assign confidence scores based on how clearly the entity is mentioned
- Provide the full context where the entity appears
- Choose entity types that make sense for {domain} work
- Focus on entities that would be useful for finding similar problems or solutions
- Include both concrete objects and abstract concepts relevant to {domain}

**Entity Type Guidelines**:
- Use specific, descriptive types that reflect {domain} categories
- Create types that {domain} professionals would recognize
- Be consistent within similar entities

Extract all meaningful entities from each record - don't limit yourself to a fixed number. Quality and completeness are more important than brevity.
"""
        
        # Format domain-specific patterns
        domain_patterns = self._format_domain_patterns(profile)
        quality_criteria = self._format_quality_criteria(profile.quality_criteria)
        context_examples = self._format_context_examples(profile.context_examples)
        
        return base_template.format(
            domain=profile.domain,
            domain_patterns=domain_patterns,
            quality_criteria=quality_criteria,
            context_examples=context_examples
        )
    
    def generate_relationship_extraction_prompt(self, profile: DataProfile) -> str:
        """Generate adaptive relationship extraction prompt"""
        
        base_template = """# Adaptive Relationship Extraction Template  
# Domain: {domain}
# Auto-generated based on data analysis

system:
You are an expert {domain} analyst who understands how different entities relate to each other within {domain} contexts. Your knowledge helps build intelligent systems that can reason about {domain} relationships.

user:
## Context: {domain} Relationship Discovery

**Your Role**: You're analyzing {domain} records to understand how different entities relate to each other, helping build a knowledge graph that {domain} professionals can use to solve problems more effectively.

**{domain} Relationship Patterns**: Based on your expertise and data analysis, you know that {domain} records typically contain:

{relationship_patterns}

## Entities Found in Records:
{{{{ if entities }}}}
**Available Entities**: {{{{ entities|join(', ') }}}}
{{{{ endif }}}}

## Records Context:
{{{{ for text in texts }}}}
**Record {{{{ loop.index }}}}: {{{{ text }}}}
{{{{ endfor }}}}

## Required Output Format:

Extract all meaningful relationships between entities found in these records. Return a JSON array where each object represents a relationship:

```json
[
  {{
    "relation_id": "rel_1",
    "source_entity": "entity from available list",
    "target_entity": "entity from available list", 
    "relation_type": "domain_appropriate_relationship",
    "confidence": 0.95,
    "context": "text showing this relationship",
    "source_record": 1,
    "direction": "directed/undirected",
    "domain_relevance": "why this relationship matters in {domain}"
  }}
]
```

**Relationship Extraction Guidelines**:
- Focus on relationships that would help {domain} professionals understand:
  {quality_criteria}

- Extract relationships as they're naturally expressed in the {domain} language
- Use relationship types that {domain} professionals would recognize
- Include both explicit relationships (directly stated) and implicit ones (clearly implied)
- Assign confidence based on how clearly the relationship is expressed
- Provide the specific context where the relationship is mentioned

**Quality Focus**:
- Prioritize relationships that enable problem-solving reasoning
- Include hierarchical relationships for structural understanding
- Capture cause-effect relationships that help predict outcomes
- Extract action relationships that link problems to solutions

Extract all meaningful relationships you can identify - comprehensive coverage helps build a more useful knowledge graph for {domain} decision-making.
"""
        
        relationship_patterns = self._format_relationship_patterns(profile)
        quality_criteria = self._format_quality_criteria(profile.quality_criteria)
        
        return base_template.format(
            domain=profile.domain,
            relationship_patterns=relationship_patterns,
            quality_criteria=quality_criteria
        )
    
    def _format_domain_patterns(self, profile: DataProfile) -> str:
        """Format domain patterns for prompt"""
        
        pattern_descriptions = {
            "primary_equipment": f"- **Primary Systems**: Main {profile.domain} equipment and systems",
            "components": f"- **Components**: Parts and sub-components within {profile.domain} systems",
            "problems": f"- **Issues**: Problems and conditions that occur in {profile.domain}",
            "actions": f"- **Actions**: What needs to be done in {profile.domain} contexts",
            "locations": f"- **Locations**: Where {profile.domain} activities occur",
            "conditions": f"- **Conditions**: States and measurements in {profile.domain}"
        }
        
        patterns = []
        for pattern in profile.entity_patterns:
            if pattern in pattern_descriptions:
                patterns.append(pattern_descriptions[pattern])
        
        return "\n".join(patterns) if patterns else f"- **Entities**: Meaningful concepts in {profile.domain} context"
    
    def _format_relationship_patterns(self, profile: DataProfile) -> str:
        """Format relationship patterns for prompt"""
        
        pattern_descriptions = {
            "structural_relationships": f"""
**Structural Relationships**:
- `has_component`: Systems contain components
- `part_of`: Components belong to larger systems  
- `connected_to`: Physical or logical connections
- `located_at`: Positional relationships""",
            
            "problem_relationships": f"""
**Problem Relationships**:
- `has_problem`: Entity experiencing an issue
- `causes`: One problem leads to another
- `affects`: Problem impacts functionality""",
            
            "action_relationships": f"""
**Action Relationships**:
- `requires_action`: Problem needs specific action
- `involves_action`: Work to be done
- `performs_action`: Action done on entity"""
        }
        
        patterns = []
        for pattern in profile.relationship_patterns:
            if pattern in pattern_descriptions:
                patterns.append(pattern_descriptions[pattern])
        
        return "\n".join(patterns) if patterns else f"**General Relationships**: Meaningful connections in {profile.domain} context"
    
    def _format_quality_criteria(self, criteria: List[str]) -> str:
        """Format quality criteria as numbered list"""
        return "\n".join(f"  {i+1}. {criterion}" for i, criterion in enumerate(criteria))
    
    def _format_context_examples(self, examples: List[str]) -> str:
        """Format context examples"""
        formatted = []
        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
            formatted.append(f"   Example {i}: \"{example}\"")
        return "\n".join(formatted)
    
    # Domain profile definitions
    def _get_maintenance_profile(self) -> DataProfile:
        return DataProfile(
            domain="maintenance",
            entity_patterns=["primary_equipment", "components", "problems", "actions", "locations"],
            relationship_patterns=["structural_relationships", "problem_relationships", "action_relationships"],
            terminology=["repair", "replace", "service", "equipment", "component", "bearing", "hose", "valve"],
            context_examples=[],
            quality_criteria=[
                "What equipment is involved?",
                "What specific components are affected?", 
                "What problems are occurring?",
                "Where the problems are located?",
                "What actions are needed?"
            ]
        )
    
    def _get_medical_profile(self) -> DataProfile:
        return DataProfile(
            domain="medical",
            entity_patterns=["conditions", "symptoms", "treatments", "medications", "anatomical"],
            relationship_patterns=["diagnostic_relationships", "treatment_relationships", "causal_relationships"],
            terminology=["patient", "diagnosis", "treatment", "symptom", "medication", "therapy"],
            context_examples=[],
            quality_criteria=[
                "What medical conditions are present?",
                "What symptoms are reported?",
                "What treatments are prescribed?",
                "What medications are involved?",
                "What anatomical structures are affected?"
            ]
        )
    
    def _get_financial_profile(self) -> DataProfile:
        return DataProfile(
            domain="financial",
            entity_patterns=["accounts", "transactions", "products", "amounts", "dates"],
            relationship_patterns=["transactional_relationships", "ownership_relationships", "temporal_relationships"],
            terminology=["account", "transaction", "payment", "balance", "interest", "loan"],
            context_examples=[],
            quality_criteria=[
                "What financial accounts are involved?",
                "What transactions occurred?",
                "What monetary amounts are specified?",
                "What financial products are referenced?",
                "What dates and timeframes are relevant?"
            ]
        )
    
    def _get_legal_profile(self) -> DataProfile:
        return DataProfile(
            domain="legal", 
            entity_patterns=["parties", "documents", "obligations", "dates", "jurisdictions"],
            relationship_patterns=["contractual_relationships", "legal_relationships", "temporal_relationships"],
            terminology=["contract", "agreement", "liability", "clause", "defendant", "plaintiff"],
            context_examples=[],
            quality_criteria=[
                "What legal parties are involved?",
                "What legal documents are referenced?",
                "What obligations and rights are specified?",
                "What dates and deadlines are mentioned?",
                "What jurisdictions apply?"
            ]
        )
    
    def _get_manufacturing_profile(self) -> DataProfile:
        return DataProfile(
            domain="manufacturing",
            entity_patterns=["products", "processes", "specifications", "quality_metrics", "batches"],
            relationship_patterns=["process_relationships", "quality_relationships", "temporal_relationships"],
            terminology=["production", "assembly", "quality", "defect", "batch", "specification"],
            context_examples=[],
            quality_criteria=[
                "What products are being manufactured?",
                "What process steps are involved?",
                "What quality specifications apply?",
                "What batch or lot numbers are referenced?",
                "What quality metrics are measured?"
            ]
        )

def save_generated_prompts(entity_prompt: str, relation_prompt: str, profile: DataProfile) -> tuple[Path, Path]:
    """Save generated prompts to files"""
    
    output_dir = Path(__file__).parent.parent.parent / "prompt_flows" / "adaptive_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save entity extraction prompt
    entity_file = output_dir / f"{profile.domain}_entity_extraction.jinja2"
    with open(entity_file, 'w', encoding='utf-8') as f:
        f.write(entity_prompt)
    
    # Save relationship extraction prompt
    relation_file = output_dir / f"{profile.domain}_relation_extraction.jinja2"  
    with open(relation_file, 'w', encoding='utf-8') as f:
        f.write(relation_prompt)
    
    return entity_file, relation_file

# Example usage
if __name__ == "__main__":
    generator = AdaptiveContextGenerator()
    
    # Example: Analyze sample data and generate prompts
    sample_texts = [
        "patient complains of chest pain and shortness of breath",
        "diagnosed with pneumonia, prescribed antibiotics",
        "blood pressure elevated, recommend diet changes"
    ]
    
    profile = generator.analyze_data_characteristics(sample_texts, domain_hint="medical")
    
    entity_prompt = generator.generate_entity_extraction_prompt(profile)
    relation_prompt = generator.generate_relationship_extraction_prompt(profile)
    
    entity_file, relation_file = save_generated_prompts(entity_prompt, relation_prompt, profile)
    
    print(f"✅ Generated adaptive prompts:")
    print(f"   • Entity extraction: {entity_file}")
    print(f"   • Relationship extraction: {relation_file}")