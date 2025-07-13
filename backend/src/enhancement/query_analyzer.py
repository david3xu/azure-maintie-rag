"""
Query analysis and enhancement module
Understands maintenance queries and expands them using domain knowledge
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from collections import defaultdict
from pathlib import Path
import json

from src.models.maintenance_models import (
    QueryAnalysis, EnhancedQuery, QueryType, MaintenanceEntity
)
from src.knowledge.data_transformer import MaintIEDataTransformer
from config.settings import settings



logger = logging.getLogger(__name__)


class MaintenanceQueryAnalyzer:
    """Analyze and enhance maintenance queries using domain knowledge"""

    def __init__(self, transformer: Optional[MaintIEDataTransformer] = None):
        """Initialize analyzer with knowledge transformer"""
        self.transformer = transformer
        self.config = settings
        self.knowledge_graph: Optional[nx.Graph] = None
        self.entity_vocabulary: Dict[str, Any] = {}

        # Load domain knowledge from config file
        self.domain_knowledge = self._load_domain_knowledge()

        # Extract patterns from domain knowledge
        self.troubleshooting_keywords = self.domain_knowledge.get("query_classification", {}).get("troubleshooting", [])
        self.procedural_keywords = self.domain_knowledge.get("query_classification", {}).get("procedural", [])
        self.preventive_keywords = self.domain_knowledge.get("query_classification", {}).get("preventive", [])
        self.safety_keywords = self.domain_knowledge.get("query_classification", {}).get("safety", [])

        self.equipment_categories = self.domain_knowledge.get("equipment_categories", {})
        self.abbreviations = self.domain_knowledge.get("technical_abbreviations", {})

        # Use configurable patterns from domain knowledge
        self.equipment_patterns = self.domain_knowledge.get("equipment_patterns", {})
        self.failure_patterns = self.domain_knowledge.get("failure_patterns", {})
        self.procedure_patterns = self.domain_knowledge.get("procedure_patterns", {})
        self.component_patterns = self.domain_knowledge.get("component_patterns", {})

        # Load extracted knowledge from MaintIE
        self.maintie_equipment = self.domain_knowledge.get("maintie_equipment", [])
        self.extracted_abbreviations = self.domain_knowledge.get("extracted_abbreviations", {})

        # Load knowledge if transformer provided
        if self.transformer:
            self._load_knowledge()

        logger.info("MaintenanceQueryAnalyzer initialized with domain knowledge")

    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain knowledge from configuration file"""
        config_path = Path("config/domain_knowledge.json")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Domain knowledge config not found: {config_path}. Using minimal defaults.")
            return self._get_minimal_defaults()
        except Exception as e:
            logger.error(f"Error loading domain knowledge: {e}")
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
            "technical_abbreviations": {},
            "equipment_patterns": {},
            "failure_patterns": {},
            "procedure_patterns": {},
            "component_patterns": {},
            "maintie_equipment": [],
            "extracted_abbreviations": {}
        }

    def _load_knowledge(self) -> None:
        """Load knowledge graph and vocabulary"""
        try:
            if hasattr(self.transformer, 'knowledge_graph'):
                self.knowledge_graph = self.transformer.knowledge_graph

            # Load entity vocabulary if available
            vocab_path = settings.processed_data_dir / "entity_vocabulary.json"
            if vocab_path.exists():
                import json
                with open(vocab_path, 'r') as f:
                    self.entity_vocabulary = json.load(f)

            logger.info("Knowledge loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load knowledge: {e}")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze maintenance query and extract key information"""
        logger.info(f"Analyzing query: {query}")

        # Clean and normalize query
        normalized_query = self._normalize_query(query)

        # Extract entities
        entities = self._extract_entities(normalized_query)

        # Classify query type
        query_type = self._classify_query_type(normalized_query)

        # Detect intent
        intent = self._detect_intent(normalized_query, query_type)

        # Assess complexity
        complexity = self._assess_complexity(normalized_query, entities)

        # Determine urgency
        urgency = self._determine_urgency(normalized_query)

        # Identify equipment category
        equipment_category = self._identify_equipment_category(entities)

        analysis = QueryAnalysis(
            original_query=query,
            query_type=query_type,
            entities=entities,
            intent=intent,
            complexity=complexity,
            urgency=urgency,
            equipment_category=equipment_category,
            confidence=0.85  # Base confidence
        )

        logger.info(f"Query analysis complete: {analysis.to_dict()}")
        return analysis

    def enhance_query(self, analysis: QueryAnalysis) -> EnhancedQuery:
        """Enhance query with expanded concepts and domain knowledge"""
        logger.info("Enhancing query with domain knowledge")

        # Expand concepts using knowledge graph
        expanded_concepts = self._expand_concepts(analysis.entities)

        # Find related entities
        related_entities = self._find_related_entities(analysis.entities)

        # Add domain context
        domain_context = self._add_domain_context(analysis)

        # Build structured search query
        structured_search = self._build_structured_search(
            analysis.entities, expanded_concepts
        )

        # Identify safety considerations
        safety_considerations = self._identify_safety_considerations(
            analysis.entities, expanded_concepts
        )

        enhanced = EnhancedQuery(
            analysis=analysis,
            expanded_concepts=expanded_concepts,
            related_entities=related_entities,
            domain_context=domain_context,
            structured_search=structured_search,
            safety_considerations=safety_considerations
        )

        logger.info(f"Query enhancement complete: {len(expanded_concepts)} concepts expanded")
        return enhanced

    def _normalize_query(self, query: str) -> str:
        """Normalize query text using configurable abbreviations"""
        # Convert to lowercase
        normalized = query.lower().strip()

        # Use domain knowledge abbreviations
        for abbr, expansion in self.abbreviations.items():
            normalized = re.sub(r'\b' + abbr + r'\b', expansion, normalized)

        # Also use extracted abbreviations from MaintIE
        for abbr, expansion in self.extracted_abbreviations.items():
            normalized = re.sub(r'\b' + abbr + r'\b', expansion, normalized)

        return normalized

    def _extract_entities(self, query: str) -> List[str]:
        """Extract maintenance entities from query"""
        entities = []

        # Use entity vocabulary if available
        if self.entity_vocabulary:
            entity_to_type = self.entity_vocabulary.get("entity_to_type", {})
            for entity_text in entity_to_type.keys():
                if entity_text.lower() in query:
                    entities.append(entity_text)

        # Pattern-based extraction as fallback
        equipment_entities = self._extract_equipment_entities(query)
        failure_entities = self._extract_failure_entities(query)
        component_entities = self._extract_component_entities(query)

        entities.extend(equipment_entities)
        entities.extend(failure_entities)
        entities.extend(component_entities)

        # Remove duplicates and return
        return list(set(entities))

    def _extract_equipment_entities(self, query: str) -> List[str]:
        """Extract equipment-related entities"""
        equipment = []
        for pattern, entity in self.equipment_patterns.items():
            if re.search(pattern, query):
                equipment.append(entity)
        return equipment

    def _extract_failure_entities(self, query: str) -> List[str]:
        """Extract failure-related entities"""
        failures = []
        for pattern, entity in self.failure_patterns.items():
            if re.search(pattern, query):
                failures.append(entity)
        return failures

    def _extract_component_entities(self, query: str) -> List[str]:
        """Extract component-related entities using configurable patterns"""
        components = []

        for pattern, component in self.config.component_patterns.items():
            if re.search(pattern, query):
                components.append(component)

        return components

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of maintenance query using domain knowledge keywords"""

        # Use domain knowledge keyword lists
        if any(keyword in query for keyword in self.troubleshooting_keywords):
            return QueryType.TROUBLESHOOTING
        elif any(keyword in query for keyword in self.procedural_keywords):
            return QueryType.PROCEDURAL
        elif any(keyword in query for keyword in self.preventive_keywords):
            return QueryType.PREVENTIVE
        elif any(keyword in query for keyword in self.safety_keywords):
            return QueryType.SAFETY
        else:
            return QueryType.INFORMATIONAL

    def _detect_intent(self, query: str, query_type: QueryType) -> str:
        """Detect specific intent within query type"""

        intent_patterns = {
            QueryType.TROUBLESHOOTING: {
                'failure_analysis': ['analysis', 'cause', 'root cause', 'why'],
                'quick_fix': ['fix', 'solve', 'resolve', 'repair'],
                'diagnosis': ['diagnose', 'identify', 'determine', 'check']
            },
            QueryType.PROCEDURAL: {
                'step_by_step': ['steps', 'procedure', 'how to'],
                'best_practice': ['best', 'proper', 'correct', 'standard'],
                'requirements': ['require', 'need', 'necessary', 'must']
            },
            QueryType.PREVENTIVE: {
                'scheduling': ['schedule', 'when', 'frequency', 'interval'],
                'inspection': ['inspect', 'check', 'examine', 'monitor'],
                'replacement': ['replace', 'change', 'renew', 'substitute']
            }
        }

        if query_type in intent_patterns:
            for intent, keywords in intent_patterns[query_type].items():
                if any(keyword in query for keyword in keywords):
                    return intent

        return 'general'

    def _assess_complexity(self, query: str, entities: List[str]) -> str:
        """Assess query complexity"""
        complexity_score = 0

        # Length factor
        if len(query.split()) > 10:
            complexity_score += 1

        # Entity count factor
        if len(entities) > 3:
            complexity_score += 1

        # Technical terms factor
        technical_terms = ['analysis', 'diagnosis', 'troubleshoot', 'maintenance']
        if any(term in query for term in technical_terms):
            complexity_score += 1

        # Multiple systems factor
        if len([e for e in entities if 'system' in e.lower()]) > 1:
            complexity_score += 1

        if complexity_score >= 3:
            return 'high'
        elif complexity_score >= 1:
            return 'medium'
        else:
            return 'low'

    def _determine_urgency(self, query: str) -> str:
        """Determine query urgency"""
        high_urgency = ['emergency', 'urgent', 'critical', 'immediate', 'failure']
        medium_urgency = ['problem', 'issue', 'repair', 'fix']

        if any(word in query for word in high_urgency):
            return 'high'
        elif any(word in query for word in medium_urgency):
            return 'medium'
        else:
            return 'low'

    def _identify_equipment_category(self, entities: List[str]) -> Optional[str]:
        """Identify equipment category from entities using domain knowledge categories"""
        for category, equipment_list in self.equipment_categories.items():
            if any(equipment in ' '.join(entities).lower() for equipment in equipment_list):
                return category

        return None

    def _expand_concepts(self, entities: List[str]) -> List[str]:
        """Expand concepts using knowledge graph"""
        expanded = set(entities)  # Start with original entities

        if self.knowledge_graph:
            for entity in entities:
                # Find entity in knowledge graph
                entity_id = self._find_entity_id(entity)
                if entity_id and entity_id in self.knowledge_graph:
                    # Get neighbors
                    neighbors = list(self.knowledge_graph.neighbors(entity_id))
                    for neighbor in neighbors[:5]:  # Limit expansion
                        neighbor_text = self.knowledge_graph.nodes[neighbor].get('text', neighbor)
                        expanded.add(neighbor_text)

        # Add rule-based expansions
        rule_expansions = self._rule_based_expansion(entities)
        expanded.update(rule_expansions)

        return list(expanded)

    def _find_entity_id(self, entity_text: str) -> Optional[str]:
        """Find entity ID for given text"""
        if self.transformer and hasattr(self.transformer, 'entities'):
            for entity_id, entity in self.transformer.entities.items():
                if entity.text.lower() == entity_text.lower():
                    return entity_id
        return None

    def _rule_based_expansion(self, entities: List[str]) -> List[str]:
        """Rule-based concept expansion using domain knowledge rules"""
        expansions = []
        expansion_rules = self.domain_knowledge.get("expansion_rules", {})

        for entity in entities:
            entity_lower = entity.lower()
            for base_entity, related_terms in expansion_rules.items():
                if base_entity in entity_lower:
                    expansions.extend(related_terms)

        return expansions

    def _find_related_entities(self, entities: List[str]) -> List[str]:
        """Find entities related to the query entities using configurable limits"""
        related = set()

        # Use knowledge graph relationships
        if self.knowledge_graph:
            for entity in entities:
                entity_id = self._find_entity_id(entity)
                if entity_id:
                    # Get entities within 2 hops
                    try:
                        neighbors = nx.single_source_shortest_path_length(
                            self.knowledge_graph, entity_id, cutoff=2
                        )
                        for neighbor_id, distance in neighbors.items():
                            if distance > 0:  # Exclude self
                                neighbor_text = self.knowledge_graph.nodes[neighbor_id].get('text', neighbor_id)
                                related.add(neighbor_text)
                    except:
                        continue

        # Use configurable limit instead of hard-coded [:15]
        return list(related)[:self.config.max_related_entities]

    def _add_domain_context(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Add maintenance domain context"""
        context = {
            'equipment_type': analysis.equipment_category,
            'maintenance_type': analysis.query_type.value,
            'complexity_level': analysis.complexity,
            'urgency_level': analysis.urgency,
            'typical_procedures': self._get_typical_procedures(analysis),
            'common_tools': self._get_common_tools(analysis.entities),
            'safety_requirements': self._get_safety_requirements(analysis.entities)
        }

        return context

    def _get_typical_procedures(self, analysis: QueryAnalysis) -> List[str]:
        """Get typical procedures for the query type using domain knowledge"""
        typical_procedures = self.domain_knowledge.get("typical_procedures", {})
        return typical_procedures.get(analysis.query_type.value, ['general procedure'])

    def _get_common_tools(self, entities: List[str]) -> List[str]:
        """Get common tools for the entities using domain knowledge mappings"""
        tools = set()
        tool_mappings = self.domain_knowledge.get("tool_mappings", {})

        for entity in entities:
            entity_lower = entity.lower()
            for equipment, equipment_tools in tool_mappings.items():
                if equipment in entity_lower:
                    tools.update(equipment_tools)

        return list(tools)

    def _get_safety_requirements(self, entities: List[str]) -> List[str]:
        """Get safety requirements for the entities using domain knowledge mappings"""
        safety_reqs = set(['general safety procedures'])
        safety_mappings = self.domain_knowledge.get("safety_mappings", {})

        for entity in entities:
            entity_lower = entity.lower()
            if any(term in entity_lower for term in ['motor', 'electrical', 'power']):
                safety_reqs.update(safety_mappings.get('electrical', []))
            if any(term in entity_lower for term in ['pump', 'pressure', 'hydraulic']):
                safety_reqs.update(safety_mappings.get('pressure', []))
            if any(term in entity_lower for term in ['rotating', 'motor', 'pump']):
                safety_reqs.update(safety_mappings.get('rotating', []))

        return list(safety_reqs)

    def _build_structured_search(self, entities: List[str], expanded_concepts: List[str]) -> str:
        """Build structured search query"""
        all_terms = entities + expanded_concepts

        # Group related terms
        grouped_terms = []
        if entities:
            entity_group = " OR ".join(f'"{term}"' for term in entities)
            grouped_terms.append(f"({entity_group})")

        if expanded_concepts:
            concept_group = " OR ".join(f'"{term}"' for term in expanded_concepts[:10])
            grouped_terms.append(f"({concept_group})")

        return " AND ".join(grouped_terms) if grouped_terms else " OR ".join(all_terms)

    def _identify_safety_considerations(self, entities: List[str], expanded_concepts: List[str]) -> List[str]:
        """Identify safety considerations"""
        all_terms = entities + expanded_concepts
        safety_considerations = []

        # High-risk equipment/activities
        if any(term in ' '.join(all_terms).lower() for term in ['electrical', 'high voltage', 'power']):
            safety_considerations.append('Electrical safety - lockout/tagout required')

        if any(term in ' '.join(all_terms).lower() for term in ['pressure', 'hydraulic', 'pneumatic']):
            safety_considerations.append('Pressure system safety - proper isolation required')

        if any(term in ' '.join(all_terms).lower() for term in ['rotating', 'motor', 'pump']):
            safety_considerations.append('Rotating equipment safety - ensure complete stop')

        if any(term in ' '.join(all_terms).lower() for term in ['chemical', 'fluid', 'oil']):
            safety_considerations.append('Chemical safety - review MSDS and use proper PPE')

        return safety_considerations

def create_analyzer(transformer: Optional[MaintIEDataTransformer] = None) -> MaintenanceQueryAnalyzer:
    """Factory function to create query analyzer"""
    return MaintenanceQueryAnalyzer(transformer)


if __name__ == "__main__":
    # Example usage
    analyzer = MaintenanceQueryAnalyzer()

    test_query = "How to troubleshoot centrifugal pump seal failure?"
    analysis = analyzer.analyze_query(test_query)
    enhanced = analyzer.enhance_query(analysis)

    print("Analysis:", analysis.to_dict())
    print("Enhanced:", enhanced.to_dict())
