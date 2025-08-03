"""
Config Generator - Infrastructure and ML configuration generation

Generates domain-specific configurations:
- Azure infrastructure naming (Search, Storage, Cosmos, ML endpoints)
- ML model architecture scaling based on domain complexity
- Performance optimization configurations
"""

from typing import Dict, List
from dataclasses import dataclass
from .pattern_engine import ExtractedPatterns


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
        # Base configurations for different domain complexities
        self.ml_config_templates = {
            "simple": {
                "node_feature_dim": 64,
                "hidden_dim": 128,
                "num_layers": 2,
                "learning_rate": 0.001
            },
            "medium": {
                "node_feature_dim": 128,
                "hidden_dim": 256,
                "num_layers": 3,
                "learning_rate": 0.001
            },
            "complex": {
                "node_feature_dim": 256,
                "hidden_dim": 512,
                "num_layers": 4,
                "learning_rate": 0.0005
            }
        }
    
    def generate_infrastructure_config(self, domain: str, patterns: ExtractedPatterns) -> InfrastructureConfig:
        """Generate Azure resource configuration based on domain patterns"""
        
        # Extract primary concepts for naming
        primary_concepts = [p.pattern_text for p in patterns.entity_patterns if p.is_high_confidence()][:3]
        
        if not primary_concepts:
            # Fallback to most frequent patterns
            primary_concepts = [p.pattern_text for p in patterns.entity_patterns[:2]]
        
        # Generate resource names based on learned concepts
        primary_concept = primary_concepts[0].lower().replace(" ", "-") if primary_concepts else domain
        secondary_concept = primary_concepts[1].lower().replace(" ", "-") if len(primary_concepts) > 1 else "data"
        
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
            primary_concepts=primary_concepts
        )
    
    def generate_ml_config(self, domain: str, patterns: ExtractedPatterns) -> MLModelConfig:
        """Generate ML model configuration adapted to domain complexity"""
        
        # Assess domain complexity
        complexity = self._assess_domain_complexity(patterns)
        
        # Get base configuration
        base_config = self.ml_config_templates[complexity]
        
        # Extract entity and relationship types from patterns
        entity_types = [p.pattern_text for p in patterns.entity_patterns if p.confidence > 0.6][:20]
        relationship_types = [p.pattern_text for p in patterns.relationship_patterns if p.confidence > 0.5]
        
        # Learn relationship types from data if none found with sufficient confidence
        if not relationship_types:
            # Use pattern engine to discover relationship patterns from the same data
            all_patterns = patterns.entity_patterns + patterns.action_patterns
            inferred_relationships = self._infer_relationships_from_patterns(all_patterns)
            relationship_types = inferred_relationships if inferred_relationships else ["connects"]
        
        # Adjust configuration based on pattern count and confidence
        entity_count = len(patterns.entity_patterns)
        avg_confidence = sum(p.confidence for p in patterns.entity_patterns) / max(1, len(patterns.entity_patterns))
        
        # Scale dimensions based on complexity
        node_feature_dim = min(256, max(64, entity_count * 4))
        hidden_dim = min(512, max(128, entity_count * 8))
        num_layers = min(4, max(2, len(entity_types) // 10 + 2))
        
        # Adjust learning rate based on confidence
        learning_rate = base_config["learning_rate"]
        if avg_confidence < 0.6:
            learning_rate *= 0.5  # Lower learning rate for uncertain patterns
        
        return MLModelConfig(
            domain=domain,
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            entity_types=entity_types,
            relationship_types=relationship_types,
            confidence=patterns.extraction_confidence
        )
    
    def generate_complete_config(self, domain: str, patterns: ExtractedPatterns) -> DomainConfig:
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
            generation_confidence=generation_confidence
        )
    
    def _assess_domain_complexity(self, patterns: ExtractedPatterns) -> str:
        """Assess domain complexity based on patterns"""
        
        entity_count = len(patterns.entity_patterns)
        high_confidence_count = len([p for p in patterns.entity_patterns if p.is_high_confidence()])
        relationship_count = len(patterns.relationship_patterns)
        
        # Simple heuristics for complexity assessment
        complexity_score = 0
        
        if entity_count > 100: complexity_score += 2
        elif entity_count > 50: complexity_score += 1
        
        if high_confidence_count > 20: complexity_score += 2
        elif high_confidence_count > 10: complexity_score += 1
        
        if relationship_count > 5: complexity_score += 1
        
        if patterns.extraction_confidence > 0.8: complexity_score += 1
        
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "simple"
    
    def _clean_resource_name(self, name: str) -> str:
        """Clean name for Azure resource naming requirements"""
        import re
        
        # Convert to lowercase, replace spaces and underscores with hyphens
        cleaned = name.lower().replace(" ", "-").replace("_", "-")
        
        # Remove non-alphanumeric characters except hyphens
        cleaned = re.sub(r'[^a-z0-9-]', '', cleaned)
        
        # Remove multiple consecutive hyphens
        cleaned = re.sub(r'-+', '-', cleaned)
        
        # Remove leading/trailing hyphens
        cleaned = cleaned.strip('-')
        
        # Ensure minimum length and maximum length for Azure resources
        if len(cleaned) < 3:
            cleaned = f"{cleaned}-data"
        if len(cleaned) > 50:
            cleaned = cleaned[:50].rstrip('-')
        
        return cleaned or "data"  # Fallback if everything gets stripped
    
    def _infer_relationships_from_patterns(self, patterns) -> List[str]:
        """Infer relationship types from entity and action patterns (data-driven approach)"""
        relationship_candidates = []
        
        # Analyze action patterns for relationship indicators
        for pattern in patterns:
            text = pattern.pattern_text.lower()
            
            # Look for verbs that indicate relationships
            if any(verb in text for verb in ['connect', 'link', 'join', 'bind']):
                relationship_candidates.append('connects')
            if any(verb in text for verb in ['contain', 'include', 'hold', 'have']):
                relationship_candidates.append('contains')
            if any(verb in text for verb in ['use', 'utiliz', 'employ', 'apply']):
                relationship_candidates.append('uses')
            if any(verb in text for verb in ['create', 'generat', 'produc', 'make']):
                relationship_candidates.append('creates')
            if any(verb in text for verb in ['part', 'component', 'element', 'piece']):
                relationship_candidates.append('part_of')
            if any(verb in text for verb in ['depend', 'rel', 'requir', 'need']):
                relationship_candidates.append('relates_to')
        
        # Remove duplicates and return most common inferred relationships
        from collections import Counter
        relationship_counts = Counter(relationship_candidates)
        
        # Return top 5 most common inferred relationships
        return [rel for rel, count in relationship_counts.most_common(5)]