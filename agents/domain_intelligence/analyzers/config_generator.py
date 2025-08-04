"""
Config Generator - Infrastructure and ML configuration generation

Generates domain-specific configurations:
- Azure infrastructure naming (Search, Storage, Cosmos, ML endpoints)
- ML model architecture scaling based on domain complexity
- Performance optimization configurations
"""

from dataclasses import dataclass
from typing import Dict, List

from .pattern_engine import ExtractedPatterns

# Import centralized configuration
from config.centralized_config import get_ml_hyperparameters_config, get_config_generator_config


@dataclass
class InfrastructureConfig:
    """Domain-specific Azure infrastructure configuration"""

    domain: str
    search_index: str
    storage_container: str
    cosmos_graph: str
    ml_endpoint: str
    confidence: float
    primary_concepts: List[str]


@dataclass
class MLModelConfig:
    """Domain-specific ML model configuration"""

    domain: str
    node_feature_dim: int
    hidden_dim: int
    num_layers: int
    learning_rate: float
    entity_types: List[str]
    relationship_types: List[str]
    confidence: float


@dataclass
class DomainConfig:
    """Complete domain configuration"""

    domain: str
    infrastructure: InfrastructureConfig
    ml_model: MLModelConfig
    patterns: ExtractedPatterns
    generation_confidence: float


class ConfigGenerator:
    """Generates infrastructure and ML configurations based on domain patterns"""

    def __init__(self):
        # Get centralized configuration
        self.config_gen_config = get_config_generator_config()
        
        # Base configurations for different domain complexities (from centralized configuration)
        ml_config = get_ml_hyperparameters_config()
        self.ml_config_templates = {
            "simple": ml_config.simple_gnn_config,
            "medium": ml_config.medium_gnn_config,
            "complex": ml_config.complex_gnn_config,
        }

    def generate_infrastructure_config(
        self, domain: str, patterns: ExtractedPatterns
    ) -> InfrastructureConfig:
        """Generate Azure resource configuration based on domain patterns"""

        # Extract primary concepts for naming
        primary_concepts = [
            p.pattern_text for p in patterns.entity_patterns if p.is_high_confidence()
        ][:self.config_gen_config.primary_concepts_fallback_limit]

        if not primary_concepts:
            # Fallback to most frequent patterns
            primary_concepts = [p.pattern_text for p in patterns.entity_patterns[:self.config_gen_config.primary_concepts_main_limit]]

        # Generate resource names based on learned concepts
        primary_concept = (
            primary_concepts[0].lower().replace(" ", "-")
            if primary_concepts
            else domain
        )
        secondary_concept = (
            primary_concepts[1].lower().replace(" ", "-")
            if len(primary_concepts) > 1
            else self.config_gen_config.fallback_secondary_concept
        )

        # Clean concept names for Azure resource naming (alphanumeric + hyphens only)
        primary_clean = self._clean_resource_name(primary_concept)
        secondary_clean = self._clean_resource_name(secondary_concept)

        return InfrastructureConfig(
            domain=domain,
            search_index=f"{domain}-docs-{primary_clean}",
            storage_container=f"{domain}-data-{secondary_clean}",
            cosmos_graph=f"{domain}-graph-{primary_clean}",
            ml_endpoint=f"gnn-{domain}-{primary_clean}",
            confidence=patterns.extraction_confidence,
            primary_concepts=primary_concepts,
        )

    def generate_ml_config(
        self, domain: str, patterns: ExtractedPatterns
    ) -> MLModelConfig:
        """Generate ML model configuration adapted to domain complexity"""

        # Assess domain complexity
        complexity = self._assess_domain_complexity(patterns)

        # Get base configuration
        base_config = self.ml_config_templates[complexity]

        # Extract entity and relationship types from patterns
        entity_types = [
            p.pattern_text for p in patterns.entity_patterns if p.confidence > self.config_gen_config.relationship_confidence_threshold
        ][:20]
        relationship_types = [
            p.pattern_text for p in patterns.relationship_patterns if p.confidence > self.config_gen_config.relationship_fallback_confidence
        ]

        # Learn relationship types from data if none found with sufficient confidence
        if not relationship_types:
            # Use pattern engine to discover relationship patterns from the same data
            all_patterns = patterns.entity_patterns + patterns.action_patterns
            inferred_relationships = self._infer_relationships_from_patterns(
                all_patterns
            )
            relationship_types = (
                inferred_relationships if inferred_relationships else [self.config_gen_config.relationship_type_connects]
            )

        # Adjust configuration based on pattern count and confidence
        entity_count = len(patterns.entity_patterns)
        avg_confidence = sum(p.confidence for p in patterns.entity_patterns) / max(
            1, len(patterns.entity_patterns)
        )

        # Scale dimensions based on complexity
        node_feature_dim = min(
            self.config_gen_config.node_feature_dim_max, 
            max(self.config_gen_config.node_feature_dim_min, entity_count * self.config_gen_config.entity_count_node_feature_multiplier)
        )
        hidden_dim = min(
            self.config_gen_config.hidden_dim_max, 
            max(self.config_gen_config.hidden_dim_min, entity_count * self.config_gen_config.entity_count_hidden_dim_multiplier)
        )
        num_layers = min(
            self.config_gen_config.num_layers_max, 
            max(self.config_gen_config.num_layers_min, len(entity_types) // self.config_gen_config.entity_types_layers_divisor + self.config_gen_config.entity_types_layers_base)
        )

        # Adjust learning rate based on confidence
        learning_rate = base_config["learning_rate"]
        if avg_confidence < self.config_gen_config.low_confidence_threshold:
            learning_rate *= self.config_gen_config.low_confidence_learning_rate_factor  # Lower learning rate for uncertain patterns

        return MLModelConfig(
            domain=domain,
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            entity_types=entity_types,
            relationship_types=relationship_types,
            confidence=patterns.extraction_confidence,
        )

    def generate_complete_config(
        self, domain: str, patterns: ExtractedPatterns
    ) -> DomainConfig:
        """Generate complete domain configuration"""

        infrastructure = self.generate_infrastructure_config(domain, patterns)
        ml_model = self.generate_ml_config(domain, patterns)

        # Overall confidence is minimum of infrastructure and ML confidence
        generation_confidence = min(infrastructure.confidence, ml_model.confidence)

        return DomainConfig(
            domain=domain,
            infrastructure=infrastructure,
            ml_model=ml_model,
            patterns=patterns,
            generation_confidence=generation_confidence,
        )

    def _assess_domain_complexity(self, patterns: ExtractedPatterns) -> str:
        """Assess domain complexity based on patterns"""

        entity_count = len(patterns.entity_patterns)
        high_confidence_count = len(
            [p for p in patterns.entity_patterns if p.is_high_confidence()]
        )
        relationship_count = len(patterns.relationship_patterns)

        # Simple heuristics for complexity assessment
        complexity_score = 0

        if entity_count > self.config_gen_config.entity_count_high_threshold:
            complexity_score += 2
        elif entity_count > self.config_gen_config.entity_count_medium_threshold:
            complexity_score += 1

        if high_confidence_count > self.config_gen_config.high_confidence_count_high_threshold:
            complexity_score += 2
        elif high_confidence_count > self.config_gen_config.high_confidence_count_medium_threshold:
            complexity_score += 1

        if relationship_count > self.config_gen_config.relationship_count_threshold:
            complexity_score += 1

        if patterns.extraction_confidence > self.config_gen_config.extraction_confidence_threshold:
            complexity_score += 1

        if complexity_score >= self.config_gen_config.complexity_score_simple_threshold:
            return "complex"
        elif complexity_score >= self.config_gen_config.complexity_score_medium_threshold:
            return "medium"
        else:
            return "simple"

    def _clean_resource_name(self, name: str) -> str:
        """Clean name for Azure resource naming requirements"""
        import re

        # Convert to lowercase, replace spaces and underscores with hyphens
        cleaned = name.lower().replace(" ", "-").replace("_", "-")

        # Remove non-alphanumeric characters except hyphens
        cleaned = re.sub(r"[^a-z0-9-]", "", cleaned)

        # Remove multiple consecutive hyphens
        cleaned = re.sub(r"-+", "-", cleaned)

        # Remove leading/trailing hyphens
        cleaned = cleaned.strip("-")

        # Ensure minimum length and maximum length for Azure resources
        if len(cleaned) < self.config_gen_config.min_resource_name_length:
            cleaned = f"{cleaned}-{self.config_gen_config.fallback_secondary_concept}"
        if len(cleaned) > self.config_gen_config.max_resource_name_length:
            cleaned = cleaned[:self.config_gen_config.max_resource_name_length].rstrip("-")

        return cleaned or self.config_gen_config.resource_name_fallback  # Fallback if everything gets stripped

    def _infer_relationships_from_patterns(self, patterns) -> List[str]:
        """Infer relationship types from entity and action patterns (data-driven approach)"""
        relationship_candidates = []

        # Analyze action patterns for relationship indicators
        for pattern in patterns:
            text = pattern.pattern_text.lower()

            # Look for verbs that indicate relationships
            if any(verb in text for verb in self.config_gen_config.relationship_verbs_connect):
                relationship_candidates.append(self.config_gen_config.relationship_type_connects)
            if any(verb in text for verb in self.config_gen_config.relationship_verbs_contain):
                relationship_candidates.append(self.config_gen_config.relationship_type_contains)
            if any(verb in text for verb in self.config_gen_config.relationship_verbs_use):
                relationship_candidates.append(self.config_gen_config.relationship_type_uses)
            if any(verb in text for verb in self.config_gen_config.relationship_verbs_create):
                relationship_candidates.append(self.config_gen_config.relationship_type_creates)
            if any(verb in text for verb in self.config_gen_config.relationship_verbs_part):
                relationship_candidates.append(self.config_gen_config.relationship_type_part_of)
            if any(verb in text for verb in self.config_gen_config.relationship_verbs_depend):
                relationship_candidates.append(self.config_gen_config.relationship_type_relates_to)

        # Remove duplicates and return most common inferred relationships
        from collections import Counter

        relationship_counts = Counter(relationship_candidates)

        # Return top most common inferred relationships
        return [rel for rel, count in relationship_counts.most_common(self.config_gen_config.top_relationships_limit)]
