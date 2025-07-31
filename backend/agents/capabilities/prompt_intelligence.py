"""
Prompt Service - Business logic for prompt generation and context management
Consolidated from core/prompt_generation/adaptive_context_generator.py
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
        logger.info("AdaptiveContextGenerator initialized with domain profiles")
    
    def analyze_data_characteristics(self, sample_texts: List[str], domain_hint: Optional[str] = None) -> DataProfile:
        """Analyze input data to determine domain and characteristics"""
        
        logger.info(f"Analyzing {len(sample_texts)} sample texts for domain characteristics...")
        
        # Combine all sample texts for analysis
        combined_text = " ".join(sample_texts).lower()
        
        # Detect domain if not provided
        detected_domain = domain_hint or self._detect_domain(combined_text)
        
        # Extract characteristics
        entity_patterns = self._extract_entity_patterns(combined_text, detected_domain)
        relationship_patterns = self._extract_relationship_patterns(combined_text, detected_domain)
        terminology = self._extract_terminology(combined_text, detected_domain)
        context_examples = self._extract_context_examples(sample_texts[:3])  # First 3 examples
        quality_criteria = self._generate_quality_criteria(detected_domain)
        
        profile = DataProfile(
            domain=detected_domain,
            entity_patterns=entity_patterns,
            relationship_patterns=relationship_patterns,
            terminology=terminology,
            context_examples=context_examples,
            quality_criteria=quality_criteria
        )
        
        logger.info(f"Generated data profile for domain: {detected_domain}")
        return profile
    
    def generate_context_aware_prompt(self, data_profile: DataProfile, task_type: str = "extraction") -> str:
        """Generate a context-aware prompt based on data profile"""
        
        base_prompts = {
            "extraction": self._get_extraction_prompt_template(),
            "relationship": self._get_relationship_prompt_template(),
            "summarization": self._get_summarization_prompt_template(),
            "classification": self._get_classification_prompt_template()
        }
        
        template = base_prompts.get(task_type, base_prompts["extraction"])
        
        # Customize prompt based on data profile
        customized_prompt = self._customize_prompt(template, data_profile)
        
        logger.info(f"Generated {task_type} prompt for {data_profile.domain} domain")
        return customized_prompt
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain from text content"""
        domain_keywords = {
            "maintenance": ["maintenance", "repair", "equipment", "failure", "service", "inspection"],
            "medical": ["patient", "diagnosis", "treatment", "medical", "health", "symptom"],
            "financial": ["investment", "revenue", "cost", "financial", "budget", "profit"],
            "legal": ["contract", "legal", "law", "compliance", "regulation", "agreement"],
            "manufacturing": ["production", "manufacturing", "assembly", "quality", "process", "factory"]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            domain_scores[domain] = score
        
        # Return domain with highest score, default to "general"
        if domain_scores:
            detected_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[detected_domain] > 0:
                return detected_domain
        
        return "general"
    
    def _extract_entity_patterns(self, text: str, domain: str) -> List[str]:
        """Extract entity patterns from text"""
        patterns = []
        
        # Common entity patterns
        patterns.extend([
            r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Person names
            r"\b\d{4}-\d{2}-\d{2}\b",         # Dates
            r"\b\$\d+(?:,\d{3})*(?:\.\d{2})?\b",  # Currency
            r"\b[A-Z]{2,}\b"                  # Acronyms
        ])
        
        # Domain-specific patterns
        if domain == "maintenance":
            patterns.extend([
                r"\b[A-Z]{2,}\d{3,}\b",       # Equipment IDs
                r"\b\d+\s*(hours?|days?|months?)\b",  # Time intervals
            ])
        elif domain == "medical":
            patterns.extend([
                r"\b[A-Z]\d{2}\.\d\b",        # ICD codes
                r"\b\d+\s*mg\b",              # Dosages
            ])
        
        return patterns
    
    def _extract_relationship_patterns(self, text: str, domain: str) -> List[str]:
        """Extract relationship patterns from text"""
        patterns = [
            "causes", "results in", "leads to", "associated with",
            "related to", "depends on", "affects", "influences"
        ]
        
        # Domain-specific relationship patterns
        if domain == "maintenance":
            patterns.extend([
                "requires", "maintains", "repairs", "replaces",
                "scheduled for", "due for", "operates on"
            ])
        elif domain == "medical":
            patterns.extend([
                "diagnoses", "treats", "prescribes", "symptoms of",
                "contraindicated with", "interacts with"
            ])
        
        return patterns
    
    def _extract_terminology(self, text: str, domain: str) -> List[str]:
        """Extract domain-specific terminology"""
        # Extract frequent capitalized terms and technical terms
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        technical_terms = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', text)
        
        # Count frequency and return top terms
        all_terms = words + technical_terms
        term_counts = Counter(all_terms)
        
        # Return top 20 most frequent terms
        top_terms = [term for term, count in term_counts.most_common(20)]
        
        return top_terms
    
    def _extract_context_examples(self, sample_texts: List[str]) -> List[str]:
        """Extract representative context examples"""
        # Return first few sentences from each sample
        examples = []
        for text in sample_texts:
            sentences = re.split(r'[.!?]+', text)
            if sentences:
                first_sentence = sentences[0].strip()
                if len(first_sentence) > 20:  # Minimum length
                    examples.append(first_sentence[:200])  # Limit length
        
        return examples[:5]  # Return up to 5 examples
    
    def _generate_quality_criteria(self, domain: str) -> List[str]:
        """Generate quality criteria for the domain"""
        base_criteria = [
            "Accuracy of extracted information",
            "Completeness of entity identification",
            "Consistency in relationship extraction",
            "Relevance to domain context"
        ]
        
        domain_criteria = {
            "maintenance": [
                "Equipment identification accuracy",
                "Maintenance procedure completeness",
                "Timeline accuracy"
            ],
            "medical": [
                "Medical terminology accuracy",
                "Patient safety considerations",
                "Clinical relevance"
            ],
            "financial": [
                "Financial accuracy",
                "Regulatory compliance",
                "Risk assessment completeness"
            ]
        }
        
        criteria = base_criteria + domain_criteria.get(domain, [])
        return criteria
    
    def _customize_prompt(self, template: str, profile: DataProfile) -> str:
        """Customize prompt template with profile information"""
        customizations = {
            "{DOMAIN}": profile.domain,
            "{ENTITY_PATTERNS}": ", ".join(profile.entity_patterns[:5]),
            "{RELATIONSHIP_PATTERNS}": ", ".join(profile.relationship_patterns[:5]),
            "{TERMINOLOGY}": ", ".join(profile.terminology[:10]),
            "{CONTEXT_EXAMPLES}": "\n".join([f"- {ex}" for ex in profile.context_examples[:3]]),
            "{QUALITY_CRITERIA}": "\n".join([f"- {qc}" for qc in profile.quality_criteria])
        }
        
        customized = template
        for placeholder, value in customizations.items():
            customized = customized.replace(placeholder, value)
        
        return customized
    
    # Domain profile methods
    def _get_maintenance_profile(self) -> DataProfile:
        """Get maintenance domain profile"""
        return DataProfile(
            domain="maintenance",
            entity_patterns=["equipment_id", "component", "technician", "date", "procedure"],
            relationship_patterns=["maintains", "repairs", "schedules", "requires", "replaces"],
            terminology=["maintenance", "repair", "inspection", "failure", "service"],
            context_examples=["Equipment ABC123 requires monthly inspection"],
            quality_criteria=["Equipment identification", "Procedure accuracy", "Timeline completeness"]
        )
    
    def _get_medical_profile(self) -> DataProfile:
        """Get medical domain profile"""
        return DataProfile(
            domain="medical", 
            entity_patterns=["patient", "diagnosis", "medication", "dosage", "doctor"],
            relationship_patterns=["diagnoses", "treats", "prescribes", "causes", "prevents"],
            terminology=["patient", "treatment", "diagnosis", "medication", "symptoms"],
            context_examples=["Patient John Smith diagnosed with hypertension"],
            quality_criteria=["Medical accuracy", "Patient safety", "Clinical relevance"]
        )
    
    def _get_financial_profile(self) -> DataProfile:
        """Get financial domain profile"""
        return DataProfile(
            domain="financial",
            entity_patterns=["amount", "account", "transaction", "date", "entity"],
            relationship_patterns=["transfers", "pays", "receives", "owes", "invests"],
            terminology=["revenue", "expense", "profit", "investment", "budget"],
            context_examples=["Company ABC transferred $10,000 to account XYZ"],
            quality_criteria=["Financial accuracy", "Compliance", "Risk assessment"]
        )
    
    def _get_legal_profile(self) -> DataProfile:
        """Get legal domain profile"""
        return DataProfile(
            domain="legal",
            entity_patterns=["party", "contract", "clause", "date", "obligation"],
            relationship_patterns=["agrees to", "binds", "requires", "prohibits", "permits"],
            terminology=["contract", "agreement", "obligation", "liability", "compliance"],
            context_examples=["Party A agrees to deliver services by December 31st"],
            quality_criteria=["Legal accuracy", "Compliance", "Obligation clarity"]
        )
    
    def _get_manufacturing_profile(self) -> DataProfile:
        """Get manufacturing domain profile"""
        return DataProfile(
            domain="manufacturing",
            entity_patterns=["product", "process", "machine", "operator", "quality_metric"],
            relationship_patterns=["produces", "assembles", "operates", "controls", "measures"],
            terminology=["production", "assembly", "quality", "process", "manufacturing"],
            context_examples=["Machine M1 produces 100 units per hour"],
            quality_criteria=["Production accuracy", "Quality standards", "Process efficiency"]
        )
    
    # Prompt templates
    def _get_extraction_prompt_template(self) -> str:
        """Get entity extraction prompt template"""
        return """
You are an expert knowledge extractor for the {DOMAIN} domain.

Your task is to extract entities and relationships from text data with the following characteristics:
- Domain: {DOMAIN}
- Common entity patterns: {ENTITY_PATTERNS}
- Common relationship patterns: {RELATIONSHIP_PATTERNS}
- Domain terminology: {TERMINOLOGY}

Context examples from this domain:
{CONTEXT_EXAMPLES}

Quality criteria for extraction:
{QUALITY_CRITERIA}

Please extract entities and relationships from the following text, ensuring accuracy and completeness according to the domain-specific criteria above.

Text to analyze:
"""
    
    def _get_relationship_prompt_template(self) -> str:
        """Get relationship extraction prompt template"""
        return """
You are an expert at identifying relationships in {DOMAIN} domain data.

Focus on extracting relationships that follow these patterns:
{RELATIONSHIP_PATTERNS}

Domain-specific terminology to consider:
{TERMINOLOGY}

Quality criteria:
{QUALITY_CRITERIA}

Context examples:
{CONTEXT_EXAMPLES}

Identify and extract relationships from the following text:
"""
    
    def _get_summarization_prompt_template(self) -> str:
        """Get summarization prompt template"""
        return """
You are an expert at summarizing {DOMAIN} domain content.

Key terminology to preserve:
{TERMINOLOGY}

Quality criteria for summaries:
{QUALITY_CRITERIA}

Context examples from this domain:
{CONTEXT_EXAMPLES}

Please provide a comprehensive summary of the following text:
"""
    
    def _get_classification_prompt_template(self) -> str:
        """Get classification prompt template"""
        return """
You are an expert classifier for {DOMAIN} domain content.

Domain characteristics:
- Common entity patterns: {ENTITY_PATTERNS}
- Key terminology: {TERMINOLOGY}

Quality criteria:
{QUALITY_CRITERIA}

Please classify the following text according to {DOMAIN} domain categories:
"""


class PromptService:
    """
    Unified Prompt Service
    Provides adaptive prompt generation and context management
    """
    
    def __init__(self):
        self.context_generator = AdaptiveContextGenerator()
        self.cached_profiles: Dict[str, DataProfile] = {}
        
        logger.info("PromptService initialized")
    
    def analyze_domain_data(self, sample_texts: List[str], domain_hint: Optional[str] = None) -> DataProfile:
        """Analyze sample texts to create domain profile"""
        cache_key = f"{domain_hint}_{hash(str(sample_texts[:3]))}"  # Cache based on first 3 samples
        
        if cache_key in self.cached_profiles:
            logger.info(f"Using cached domain profile for {domain_hint}")
            return self.cached_profiles[cache_key]
        
        profile = self.context_generator.analyze_data_characteristics(sample_texts, domain_hint)
        self.cached_profiles[cache_key] = profile
        
        return profile
    
    def generate_extraction_prompt(self, data_profile: DataProfile) -> str:
        """Generate entity extraction prompt"""
        return self.context_generator.generate_context_aware_prompt(data_profile, "extraction")
    
    def generate_relationship_prompt(self, data_profile: DataProfile) -> str:
        """Generate relationship extraction prompt"""
        return self.context_generator.generate_context_aware_prompt(data_profile, "relationship")
    
    def generate_summarization_prompt(self, data_profile: DataProfile) -> str:
        """Generate summarization prompt"""
        return self.context_generator.generate_context_aware_prompt(data_profile, "summarization")
    
    def generate_classification_prompt(self, data_profile: DataProfile) -> str:
        """Generate classification prompt"""
        return self.context_generator.generate_context_aware_prompt(data_profile, "classification")
    
    def get_domain_profile(self, domain: str) -> Optional[DataProfile]:
        """Get predefined domain profile"""
        if domain in self.context_generator.known_domains:
            return self.context_generator.known_domains[domain]
        return None
    
    def clear_cache(self):
        """Clear cached profiles"""
        self.cached_profiles.clear()
        logger.info("Prompt service cache cleared")