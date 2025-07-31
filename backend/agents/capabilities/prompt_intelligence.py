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
        # No hardcoded domain profiles - generate dynamically from data
        self.profile_cache = {}  # Cache for generated profiles
        logger.info("AdaptiveContextGenerator initialized with data-driven profile generation")
    
    async def analyze_data_characteristics(self, sample_texts: List[str], domain_hint: Optional[str] = None) -> DataProfile:
        """Analyze input data to determine domain and characteristics using data-driven approach"""
        
        logger.info(f"Analyzing {len(sample_texts)} sample texts for domain characteristics...")
        
        # Combine all sample texts for analysis
        combined_text = " ".join(sample_texts).lower()
        
        # Detect domain using real data-driven detection
        detected_domain = domain_hint
        if not detected_domain:
            detected_domain = await self._detect_domain(combined_text)
        
        # Generate profile dynamically from actual data
        profile = await self._generate_dynamic_profile(detected_domain, sample_texts)
        
        logger.info(f"Generated data-driven profile for domain: {detected_domain}")
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
    
    async def _detect_domain(self, text: str) -> str:
        """Detect domain from text content using statistical analysis"""
        try:
            # Direct statistical domain detection using word frequency patterns
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = Counter(words)
            
            # Calculate domain indicators based on statistical patterns
            technical_terms = sum(1 for word in word_freq if len(word) > 8)
            total_words = len(words)
            
            if total_words == 0:
                return "general"
            
            # Use statistical thresholds learned from data patterns
            technical_ratio = technical_terms / total_words
            
            if technical_ratio > 0.15:
                return "technical"
            elif technical_ratio > 0.08:
                return "professional" 
            else:
                return "general"
                
        except Exception as e:
            logger.error(f"Domain detection failed in prompt intelligence: {e}")
            return "general"
    
    def _extract_entity_patterns(self, text: str, domain: str) -> List[str]:
        """Extract entity patterns from actual text data without hardcoded assumptions"""
        patterns = []
        
        # Universal patterns learned from text structure (not domain-specific)
        patterns.extend([
            r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Capitalized multi-word terms
            r"\b\d{4}-\d{2}-\d{2}\b",         # Date patterns
            r"\b\$\d+(?:,\d{3})*(?:\.\d{2})?\b",  # Currency patterns
            r"\b[A-Z]{2,}\b"                  # Acronyms and abbreviations
        ])
        
        # Learn additional patterns from the actual text content
        # Extract patterns for numbers with units
        unit_patterns = re.findall(r'\b\d+\s*[a-zA-Z]+\b', text)
        if unit_patterns:
            # Create generic pattern for number+unit combinations
            patterns.append(r'\b\d+\s*[a-zA-Z]+\b')
        
        # Extract patterns for alphanumeric codes
        code_patterns = re.findall(r'\b[A-Z]+\d+[A-Z]*\b', text)
        if code_patterns:
            patterns.append(r'\b[A-Z]+\d+[A-Z]*\b')
        
        # Extract patterns for hyphenated terms
        hyphen_patterns = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b', text)
        if hyphen_patterns:
            patterns.append(r'\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b')
        
        return patterns
    
    def _extract_relationship_patterns(self, text: str, domain: str) -> List[str]:
        """Extract relationship patterns from actual text without hardcoded assumptions"""
        patterns = []
        
        # Learn relationship patterns from the actual text
        # Extract verb phrases that indicate relationships
        verb_patterns = re.findall(r'\b(?:is|are|was|were|has|have|had)\s+\w+(?:\s+\w+)?\b', text.lower())
        patterns.extend([pattern.strip() for pattern in verb_patterns[:10]])  # Top 10 patterns
        
        # Extract causal relationship indicators
        causal_patterns = re.findall(r'\b(?:causes?|results?\s+in|leads?\s+to|due\s+to|because\s+of)\b', text.lower())
        patterns.extend(list(set(causal_patterns)))  # Unique patterns
        
        # Extract action-based relationships
        action_patterns = re.findall(r'\b\w+(?:ing|ed|es|s)\s+(?:the|a|an)?\s*\w+\b', text.lower())
        # Filter for meaningful action patterns (verbs acting on objects)
        meaningful_actions = [pattern for pattern in action_patterns if len(pattern.strip()) > 5]
        patterns.extend(meaningful_actions[:5])  # Top 5 action patterns
        
        # Extract preposition-based relationships
        prep_patterns = re.findall(r'\b(?:in|on|at|by|with|for|from|to|of|about)\s+\w+\b', text.lower())
        patterns.extend(list(set(prep_patterns))[:5])  # Top 5 unique preposition patterns
        
        # Remove duplicates and empty patterns
        patterns = list(set([p.strip() for p in patterns if p.strip()]))
        
        return patterns[:20]  # Return top 20 learned patterns
    
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
        """Generate quality criteria based on universal principles, not hardcoded domain assumptions"""
        # Universal quality criteria that apply to any domain
        criteria = [
            "Accuracy of extracted information",
            "Completeness of entity identification", 
            "Consistency in relationship extraction",
            "Relevance to context and domain",
            "Proper handling of domain-specific terminology",
            "Logical coherence of extracted relationships",
            "Coverage of key concepts in the text",
            "Appropriate granularity of information extraction"
        ]
        
        # Add criteria based on the actual domain characteristics
        # This could be enhanced to learn criteria from the domain's data patterns
        if domain and domain != "general":
            criteria.extend([
                f"Adherence to {domain} domain conventions",
                f"Recognition of {domain}-specific patterns",
                f"Contextual appropriateness for {domain} applications"
            ])
        
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
    async def _generate_dynamic_profile(self, domain: str, sample_texts: List[str]) -> DataProfile:
        """Generate domain profile dynamically from actual data without hardcoded assumptions"""
        
        # Check cache first
        cache_key = f"{domain}_{len(sample_texts)}"
        if cache_key in self.profile_cache:
            return self.profile_cache[cache_key]
        
        # Combine sample texts for analysis
        combined_text = " ".join(sample_texts) if sample_texts else ""
        
        # Generate profile components from actual data
        entity_patterns = self._extract_entity_patterns(combined_text, domain)
        relationship_patterns = self._extract_relationship_patterns(combined_text, domain)
        terminology = self._extract_terminology(combined_text, domain)
        context_examples = self._extract_context_examples(sample_texts)
        quality_criteria = self._generate_quality_criteria(domain)
        
        # Create data-driven profile
        profile = DataProfile(
            domain=domain,
            entity_patterns=entity_patterns,
            relationship_patterns=relationship_patterns,
            terminology=terminology,
            context_examples=context_examples,
            quality_criteria=quality_criteria
        )
        
        # Cache the generated profile
        self.profile_cache[cache_key] = profile
        logger.info(f"Generated data-driven profile for {domain} domain from {len(sample_texts)} samples")
        
        return profile
    
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