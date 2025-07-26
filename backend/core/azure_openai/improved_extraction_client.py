"""
Improved Knowledge Extractor with Real-time Comparison Output
Fixes critical issues: entity-relation linking, context preservation, rich extraction
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from openai import AzureOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class ImprovedKnowledgeExtractor:
    """
    Fixed knowledge extractor that preserves context and creates proper entity-relation links
    """

    def __init__(self, domain_name: str = "maintenance"):
        self.domain_name = domain_name

        # Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            azure_endpoint=settings.openai_api_base
        )
        self.deployment_name = settings.openai_deployment_name

        # Output directory for comparisons
        self.output_dir = Path(settings.BASE_DIR) / "data" / "extraction_comparisons"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ImprovedKnowledgeExtractor initialized for {domain_name}")

    def extract_with_comparison(self, texts: List[str], sample_size: int = 10) -> Dict[str, Any]:
        """
        Extract knowledge and create detailed comparison with raw text
        """
        # Take sample for detailed analysis
        sample_texts = texts[:sample_size]

        results = {
            "extraction_timestamp": datetime.now().isoformat(),
            "domain": self.domain_name,
            "sample_size": len(sample_texts),
            "total_available": len(texts),
            "comparisons": []
        }

        logger.info(f"Processing {len(sample_texts)} texts for detailed extraction comparison")

        for i, raw_text in enumerate(sample_texts):
            logger.info(f"Processing text {i+1}/{len(sample_texts)}")

            # Extract from single text for detailed comparison
            extraction_result = self._extract_from_single_text(raw_text)

            comparison = {
                "text_id": i,
                "raw_text": raw_text,
                "extracted_entities": extraction_result["entities"],
                "extracted_relations": extraction_result["relations"],
                "extraction_quality": self._assess_extraction_quality(raw_text, extraction_result),
                "issues_identified": self._identify_issues(raw_text, extraction_result)
            }

            results["comparisons"].append(comparison)

            # Real-time output
            self._print_comparison(i, raw_text, extraction_result)

        # Save detailed results
        output_file = self.output_dir / f"extraction_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Detailed comparison saved to: {output_file}")

        # Summary statistics
        results["summary"] = self._generate_summary_stats(results["comparisons"])

        return results

    def _extract_from_single_text(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relations from a single text with proper linking
        """
        # Step 1: Extract entities with full context
        entities = self._extract_entities_with_context(text)

        # Step 2: Extract relations linking the actual entities
        relations = self._extract_relations_with_linking(text, entities)

        return {
            "entities": entities,
            "relations": relations,
            "raw_text": text
        }

    def _extract_entities_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities preserving full context and semantic meaning
        """
        prompt = f"""
        Analyze this maintenance text and extract specific entities with their full context.

        Text: "{text}"

        Extract entities as JSON objects with:
        - "text": exact entity phrase from the text
        - "entity_type": semantic category (equipment, component, issue, action, location, etc.)
        - "context": surrounding words that give meaning
        - "semantic_role": what role this entity plays (subject, object, component, etc.)

        Focus on:
        1. Specific equipment/components mentioned
        2. Issues or problems described
        3. Actions or procedures
        4. Locations or positions
        5. States or conditions

        Return JSON array of entity objects:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2
            )

            entities_text = response.choices[0].message.content.strip()
            entities = self._parse_json_response(entities_text, "entities")

            # Add unique IDs
            for i, entity in enumerate(entities):
                entity["entity_id"] = f"entity_{i}"
                if "context" not in entity:
                    entity["context"] = text  # Fallback to full text

            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def _extract_relations_with_linking(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations that properly link to extracted entities
        """
        if len(entities) < 2:
            return []

        # Create entity reference for the LLM
        entity_references = {}
        entity_list = []
        for entity in entities:
            ref = f"{entity['entity_id']}: {entity['text']}"
            entity_references[entity['entity_id']] = entity['text']
            entity_list.append(ref)

        prompt = f"""
        Given this maintenance text and the extracted entities, identify relationships between them.

        Text: "{text}"

        Extracted Entities:
        {chr(10).join(entity_list)}

        Extract relationships as JSON objects with:
        - "source_entity_id": ID of source entity (e.g., "entity_0")
        - "target_entity_id": ID of target entity (e.g., "entity_1")
        - "relation_type": relationship name (e.g., "has_issue", "requires", "part_of")
        - "context": text snippet showing this relationship
        - "confidence": confidence score 0-1

        Only extract relationships that are clearly expressed in the text.
        Both source and target must be from the entity list above.

        Return JSON array of relation objects:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.2
            )

            relations_text = response.choices[0].message.content.strip()
            relations = self._parse_json_response(relations_text, "relations")

            # Validate entity IDs exist
            valid_relations = []
            valid_entity_ids = {entity['entity_id'] for entity in entities}

            for i, relation in enumerate(relations):
                relation["relation_id"] = f"relation_{i}"

                # Validate entity IDs
                source_id = relation.get("source_entity_id", "")
                target_id = relation.get("target_entity_id", "")

                if source_id in valid_entity_ids and target_id in valid_entity_ids:
                    valid_relations.append(relation)
                else:
                    logger.warning(f"Invalid entity IDs in relation: {source_id} -> {target_id}")

            return valid_relations

        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []

    def _parse_json_response(self, response_text: str, expected_type: str) -> List[Dict[str, Any]]:
        """
        Parse JSON response with fallback handling
        """
        try:
            # Try direct JSON parsing
            if response_text.strip().startswith('['):
                return json.loads(response_text)

            # Look for JSON array in the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # If no JSON found, return empty list
            logger.warning(f"No valid JSON found in {expected_type} response")
            return []

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for {expected_type}: {e}")
            return []

    def _assess_extraction_quality(self, raw_text: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of extraction compared to raw text
        """
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        # Calculate metrics
        text_length = len(raw_text)
        entities_count = len(entities)
        relations_count = len(relations)

        # Context preservation score
        total_context_chars = sum(len(e.get("context", "")) for e in entities)
        context_preservation = min(total_context_chars / text_length, 1.0) if text_length > 0 else 0

        # Entity coverage (how much of the original text is represented)
        entity_chars = sum(len(e.get("text", "")) for e in entities)
        entity_coverage = entity_chars / text_length if text_length > 0 else 0

        # Relationship connectivity
        connected_entities = set()
        for rel in relations:
            connected_entities.add(rel.get("source_entity_id", ""))
            connected_entities.add(rel.get("target_entity_id", ""))

        connectivity_score = len(connected_entities) / entities_count if entities_count > 0 else 0

        return {
            "entities_extracted": entities_count,
            "relations_extracted": relations_count,
            "context_preservation_score": round(context_preservation, 3),
            "entity_coverage_score": round(entity_coverage, 3),
            "connectivity_score": round(connectivity_score, 3),
            "overall_quality": round((context_preservation + entity_coverage + connectivity_score) / 3, 3)
        }

    def _identify_issues(self, raw_text: str, extraction: Dict[str, Any]) -> List[str]:
        """
        Identify issues with the extraction
        """
        issues = []

        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        # Check for common issues
        if not entities:
            issues.append("No entities extracted")

        if not relations:
            issues.append("No relations extracted")

        # Check for disconnected relations
        disconnected_relations = [
            rel for rel in relations
            if not rel.get("source_entity_id") or not rel.get("target_entity_id")
        ]
        if disconnected_relations:
            issues.append(f"{len(disconnected_relations)} relations have missing entity IDs")

        # Check for empty contexts
        empty_context_entities = [e for e in entities if not e.get("context", "").strip()]
        if empty_context_entities:
            issues.append(f"{len(empty_context_entities)} entities have no context")

        # Check for very short entities (likely generic types instead of specific instances)
        short_entities = [e for e in entities if len(e.get("text", "")) < 3]
        if short_entities:
            issues.append(f"{len(short_entities)} entities are very short/generic")

        return issues

    def _print_comparison(self, text_id: int, raw_text: str, extraction: Dict[str, Any]):
        """
        Print real-time comparison to console
        """
        print(f"\n{'='*80}")
        print(f"TEXT {text_id + 1}: EXTRACTION COMPARISON")
        print(f"{'='*80}")

        print(f"\n📝 RAW TEXT:")
        print(f"   \"{raw_text}\"")

        print(f"\n🔍 EXTRACTED ENTITIES:")
        entities = extraction.get("entities", [])
        if entities:
            for entity in entities:
                print(f"   • {entity.get('text', 'N/A')} [{entity.get('entity_type', 'unknown')}]")
                print(f"     Context: \"{entity.get('context', 'N/A')}\"")
        else:
            print("   ❌ No entities extracted")

        print(f"\n🔗 EXTRACTED RELATIONS:")
        relations = extraction.get("relations", [])
        if relations:
            for relation in relations:
                source_id = relation.get("source_entity_id", "")
                target_id = relation.get("target_entity_id", "")
                relation_type = relation.get("relation_type", "unknown")

                # Find actual entity texts
                source_text = next((e['text'] for e in entities if e['entity_id'] == source_id), source_id)
                target_text = next((e['text'] for e in entities if e['entity_id'] == target_id), target_id)

                print(f"   • {source_text} --[{relation_type}]--> {target_text}")
                print(f"     Context: \"{relation.get('context', 'N/A')}\"")
        else:
            print("   ❌ No relations extracted")

        # Quality assessment
        quality = self._assess_extraction_quality(raw_text, extraction)
        print(f"\n📊 QUALITY METRICS:")
        print(f"   • Entities: {quality['entities_extracted']}")
        print(f"   • Relations: {quality['relations_extracted']}")
        print(f"   • Context Preservation: {quality['context_preservation_score']}")
        print(f"   • Entity Coverage: {quality['entity_coverage_score']}")
        print(f"   • Connectivity: {quality['connectivity_score']}")
        print(f"   • Overall Quality: {quality['overall_quality']}")

        # Issues
        issues = self._identify_issues(raw_text, extraction)
        if issues:
            print(f"\n⚠️  ISSUES IDENTIFIED:")
            for issue in issues:
                print(f"   • {issue}")

    def _generate_summary_stats(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics across all comparisons
        """
        if not comparisons:
            return {}

        # Aggregate quality metrics
        quality_scores = [comp["extraction_quality"]["overall_quality"] for comp in comparisons]
        context_scores = [comp["extraction_quality"]["context_preservation_score"] for comp in comparisons]
        coverage_scores = [comp["extraction_quality"]["entity_coverage_score"] for comp in comparisons]
        connectivity_scores = [comp["extraction_quality"]["connectivity_score"] for comp in comparisons]

        # Count extraction results
        total_entities = sum(comp["extraction_quality"]["entities_extracted"] for comp in comparisons)
        total_relations = sum(comp["extraction_quality"]["relations_extracted"] for comp in comparisons)

        # Issue analysis
        all_issues = []
        for comp in comparisons:
            all_issues.extend(comp["issues_identified"])

        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        return {
            "total_texts_processed": len(comparisons),
            "total_entities_extracted": total_entities,
            "total_relations_extracted": total_relations,
            "avg_entities_per_text": round(total_entities / len(comparisons), 2),
            "avg_relations_per_text": round(total_relations / len(comparisons), 2),
            "quality_metrics": {
                "avg_overall_quality": round(sum(quality_scores) / len(quality_scores), 3),
                "avg_context_preservation": round(sum(context_scores) / len(context_scores), 3),
                "avg_entity_coverage": round(sum(coverage_scores) / len(coverage_scores), 3),
                "avg_connectivity": round(sum(connectivity_scores) / len(connectivity_scores), 3)
            },
            "common_issues": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }


if __name__ == "__main__":
    # Test the improved extractor
    extractor = ImprovedKnowledgeExtractor("maintenance")

    sample_texts = [
        "air conditioner thermostat not working",
        "air receiver safety valves to be replaced",
        "analyse failed driveline component",
        "auxiliary Cat engine lube service",
        "axle temperature sensor fault"
    ]

    results = extractor.extract_with_comparison(sample_texts)
    print(f"\n🎯 EXTRACTION COMPLETE")
    print(f"Summary: {results['summary']}")
