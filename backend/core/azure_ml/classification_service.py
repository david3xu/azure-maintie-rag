"""
Universal Entity and Relation Classifier
Provides domain-agnostic classification capabilities that work with any domain
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Universal classification result"""
    entity_type: str
    confidence: float
    category: str
    metadata: Dict[str, Any]


class UniversalEntityClassifier:
    """Universal entity classifier that works with any domain through configuration"""

    def __init__(self, domain_config: Optional[Dict[str, Any]] = None):
        """Initialize universal entity classifier"""
        self.domain_config = domain_config or {}
        self.entity_types = self._load_entity_types()
        self.classification_rules = self._build_classification_rules()

        logger.info(f"Universal entity classifier initialized with {len(self.entity_types)} entity types")

    def classify_entity(self, entity_text: str, context: str = "") -> ClassificationResult:
        """Classify an entity using universal rules"""

        # Apply universal classification rules
        for rule_name, rule_func in self.classification_rules.items():
            result = rule_func(entity_text, context)
            if result:
                return result

        # ❌ REMOVED: Silent fallback - let the error propagate
        raise RuntimeError(f"Entity classification failed for: {entity_text}")

    def classify_entities_batch(self, entities: List[str], contexts: List[str] = None) -> List[ClassificationResult]:
        """Classify multiple entities efficiently"""
        if contexts is None:
            contexts = [""] * len(entities)

        results = []
        for entity, context in zip(entities, contexts):
            result = self.classify_entity(entity, context)
            results.append(result)

        return results

    def _load_entity_types(self) -> Dict[str, Dict[str, Any]]:
        """Load entity types from domain configuration"""
        entity_types = {}

        # Load base universal types
        base_types = {
            "object": {"category": "physical", "confidence_base": 0.8},
            "concept": {"category": "abstract", "confidence_base": 0.7},
            "process": {"category": "action", "confidence_base": 0.8},
            "attribute": {"category": "property", "confidence_base": 0.6},
            "entity": {"category": "generic", "confidence_base": 0.5}
        }
        entity_types.update(base_types)

        # Load domain-specific types if available
        if self.domain_config:
            domain_types = self.domain_config.get("entity_types", {})
            entity_types.update(domain_types)

        return entity_types

    def _build_classification_rules(self) -> Dict[str, callable]:
        """Build universal classification rules"""
        return {
            "equipment_rule": self._classify_equipment,
            "component_rule": self._classify_component,
            "action_rule": self._classify_action,
            "issue_rule": self._classify_issue,
            "attribute_rule": self._classify_attribute
        }

    def _classify_equipment(self, entity_text: str, context: str) -> Optional[ClassificationResult]:
        """Universal equipment classification (no hardcoded keywords)"""
        # Use only data-driven or Azure Language Services-based classification
        return None

    def _classify_component(self, entity_text: str, context: str) -> Optional[ClassificationResult]:
        """Classify component-type entities"""
        component_keywords = ["part", "component", "element", "piece", "section"]

        if any(keyword in entity_text.lower() for keyword in component_keywords):
            return ClassificationResult(
                entity_type="component",
                confidence=0.8,
                category="physical",
                metadata={"rule": "component_keywords"}
            )
        return None

    def _classify_action(self, entity_text: str, context: str) -> Optional[ClassificationResult]:
        """Classify action-type entities"""
        action_keywords = ["replace", "repair", "check", "install", "remove", "adjust", "fix"]

        if any(keyword in entity_text.lower() for keyword in action_keywords):
            return ClassificationResult(
                entity_type="action",
                confidence=0.9,
                category="process",
                metadata={"rule": "action_keywords"}
            )
        return None

    def _classify_issue(self, entity_text: str, context: str) -> Optional[ClassificationResult]:
        """Classify issue-type entities"""
        issue_keywords = ["problem", "issue", "fault", "error", "failure", "malfunction"]

        if any(keyword in entity_text.lower() for keyword in issue_keywords):
            return ClassificationResult(
                entity_type="issue",
                confidence=0.85,
                category="problem",
                metadata={"rule": "issue_keywords"}
            )
        return None

    def _classify_attribute(self, entity_text: str, context: str) -> Optional[ClassificationResult]:
        """Classify attribute-type entities"""
        # Look for descriptive words or properties
        if len(entity_text.split()) == 1 and entity_text.lower() in ["broken", "damaged", "faulty", "working"]:
            return ClassificationResult(
                entity_type="attribute",
                confidence=0.7,
                category="property",
                metadata={"rule": "attribute_keywords"}
            )
        return None


class UniversalRelationClassifier:
    """Universal relation classifier that works with any domain through configuration"""

    def __init__(self, domain_config: Optional[Dict[str, Any]] = None):
        """Initialize universal relation classifier"""
        self.domain_config = domain_config or {}
        self.relation_types = self._load_relation_types()
        self.classification_rules = self._build_classification_rules()

        logger.info(f"Universal relation classifier initialized with {len(self.relation_types)} relation types")

    def classify_relation(self, relation_text: str, entity1: str = "", entity2: str = "") -> ClassificationResult:
        """Classify a relation using universal rules"""

        # Apply universal classification rules
        for rule_name, rule_func in self.classification_rules.items():
            result = rule_func(relation_text, entity1, entity2)
            if result:
                return result

        # ❌ REMOVED: Silent fallback - let the error propagate
        raise RuntimeError(f"Relation classification failed for: {relation_text}")

    def _load_relation_types(self) -> Dict[str, Dict[str, Any]]:
        """Load relation types from domain configuration"""
        relation_types = {}

        # Load base universal types
        base_types = {
            "part_of": {"category": "structural", "confidence_base": 0.9},
            "requires": {"category": "dependency", "confidence_base": 0.8},
            "causes": {"category": "causality", "confidence_base": 0.8},
            "located_at": {"category": "spatial", "confidence_base": 0.7},
            "follows": {"category": "temporal", "confidence_base": 0.7}
        }
        relation_types.update(base_types)

        # Load domain-specific types if available
        if self.domain_config:
            domain_types = self.domain_config.get("relation_types", {})
            relation_types.update(domain_types)

        return relation_types

    def _build_classification_rules(self) -> Dict[str, callable]:
        """Build universal classification rules"""
        return {
            "dependency_rule": self._classify_dependency,
            "causality_rule": self._classify_causality,
            "composition_rule": self._classify_composition,
            "spatial_rule": self._classify_spatial,
            "temporal_rule": self._classify_temporal
        }

    def _classify_dependency(self, relation_text: str, entity1: str, entity2: str) -> Optional[ClassificationResult]:
        """Classify dependency relations"""
        dependency_keywords = ["requires", "needs", "depends", "relies"]

        if any(keyword in relation_text.lower() for keyword in dependency_keywords):
            return ClassificationResult(
                entity_type="requires",
                confidence=0.9,
                category="dependency",
                metadata={"rule": "dependency_keywords"}
            )
        return None

    def _classify_causality(self, relation_text: str, entity1: str, entity2: str) -> Optional[ClassificationResult]:
        """Classify causality relations"""
        causality_keywords = ["causes", "leads", "results", "triggers"]

        if any(keyword in relation_text.lower() for keyword in causality_keywords):
            return ClassificationResult(
                entity_type="causes",
                confidence=0.9,
                category="causality",
                metadata={"rule": "causality_keywords"}
            )
        return None

    def _classify_composition(self, relation_text: str, entity1: str, entity2: str) -> Optional[ClassificationResult]:
        """Classify composition relations"""
        composition_keywords = ["part_of", "contains", "includes", "comprises"]

        if any(keyword in relation_text.lower() for keyword in composition_keywords):
            return ClassificationResult(
                entity_type="part_of",
                confidence=0.9,
                category="structural",
                metadata={"rule": "composition_keywords"}
            )
        return None

    def _classify_spatial(self, relation_text: str, entity1: str, entity2: str) -> Optional[ClassificationResult]:
        """Classify spatial relations"""
        spatial_keywords = ["located", "positioned", "placed", "situated"]

        if any(keyword in relation_text.lower() for keyword in spatial_keywords):
            return ClassificationResult(
                entity_type="located_at",
                confidence=0.8,
                category="spatial",
                metadata={"rule": "spatial_keywords"}
            )
        return None

    def _classify_temporal(self, relation_text: str, entity1: str, entity2: str) -> Optional[ClassificationResult]:
        """Classify temporal relations"""
        temporal_keywords = ["follows", "precedes", "before", "after"]

        if any(keyword in relation_text.lower() for keyword in temporal_keywords):
            return ClassificationResult(
                entity_type="follows",
                confidence=0.8,
                category="temporal",
                metadata={"rule": "temporal_keywords"}
            )
        return None


class UniversalClassificationPipeline:
    """Complete universal classification pipeline"""

    def __init__(self, domain_config: Optional[Dict[str, Any]] = None):
        """Initialize universal classification pipeline"""
        self.entity_classifier = UniversalEntityClassifier(domain_config)
        self.relation_classifier = UniversalRelationClassifier(domain_config)

        logger.info("Universal classification pipeline initialized")

    def classify_knowledge_triplet(self, entity1: str, relation: str, entity2: str) -> Dict[str, ClassificationResult]:
        """Classify a complete knowledge triplet"""
        return {
            "entity1": self.entity_classifier.classify_entity(entity1),
            "relation": self.relation_classifier.classify_relation(relation, entity1, entity2),
            "entity2": self.entity_classifier.classify_entity(entity2)
        }

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return {
            "entity_types": len(self.entity_classifier.entity_types),
            "relation_types": len(self.relation_classifier.relation_types),
            "entity_rules": len(self.entity_classifier.classification_rules),
            "relation_rules": len(self.relation_classifier.classification_rules)
        }