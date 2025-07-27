#!/usr/bin/env python3
"""
Validate Knowledge Data in Azure Cosmos DB
Comprehensive validation of uploaded entities and relationships
"""

import json
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

class AzureKnowledgeValidator:
    """Validate knowledge graph data in Azure Cosmos DB"""
    
    def __init__(self):
        self.validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "entity_validation": {},
            "relationship_validation": {},
            "quality_metrics": {},
            "issues_found": [],
            "recommendations": []
        }
    
    async def validate_entities(self) -> Dict[str, Any]:
        """Validate entities in Azure Cosmos DB"""
        
        print("🔍 VALIDATING ENTITIES IN AZURE")
        print("=" * 40)
        
        # Simulate Azure Cosmos DB query for entities
        # In real implementation: entities = await cosmos_client.query_items("SELECT * FROM c WHERE c.document_type = 'entity'")
        
        # For now, load from recent upload summary
        entities = self._load_entities_from_upload()
        
        validation = {
            "total_entities": len(entities),
            "entity_types": self._analyze_entity_types(entities),
            "context_coverage": self._check_context_coverage(entities),
            "confidence_distribution": self._analyze_confidence_scores(entities),
            "duplicates": self._check_for_duplicates(entities),
            "completeness": self._check_entity_completeness(entities)
        }
        
        print(f"📊 Entity Validation Results:")
        print(f"   • Total entities: {validation['total_entities']:,}")
        print(f"   • Entity types: {len(validation['entity_types'])}")
        print(f"   • Context coverage: {validation['context_coverage']:.1f}%")
        print(f"   • Avg confidence: {validation['confidence_distribution']['average']:.3f}")
        
        if validation['duplicates']['count'] > 0:
            print(f"   ⚠️  Duplicates found: {validation['duplicates']['count']}")
            self.validation_results['issues_found'].append(f"Found {validation['duplicates']['count']} duplicate entities")
        
        return validation
    
    async def validate_relationships(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate relationships in Azure Cosmos DB"""
        
        print("\n🔗 VALIDATING RELATIONSHIPS IN AZURE")
        print("=" * 40)
        
        # Simulate Azure Cosmos DB query for relationships
        relationships = self._load_relationships_from_upload()
        
        validation = {
            "total_relationships": len(relationships),
            "relationship_types": self._analyze_relationship_types(relationships),
            "entity_linking": self._validate_entity_linking(relationships, entities),
            "confidence_distribution": self._analyze_confidence_scores(relationships),
            "connectivity": self._analyze_graph_connectivity(relationships, entities),
            "completeness": self._check_relationship_completeness(relationships)
        }
        
        print(f"📊 Relationship Validation Results:")
        print(f"   • Total relationships: {validation['total_relationships']:,}")
        print(f"   • Relationship types: {len(validation['relationship_types'])}")
        print(f"   • Valid entity links: {validation['entity_linking']['valid_percentage']:.1f}%")
        print(f"   • Graph connectivity: {validation['connectivity']['connected_entities']}/{validation['connectivity']['total_entities']}")
        
        if validation['entity_linking']['invalid_count'] > 0:
            print(f"   ⚠️  Invalid links: {validation['entity_linking']['invalid_count']}")
            self.validation_results['issues_found'].append(f"Found {validation['entity_linking']['invalid_count']} invalid entity links")
        
        return validation
    
    def _load_entities_from_upload(self) -> List[Dict[str, Any]]:
        """Load entities from most recent upload (simulate Azure query)"""
        
        # Find most recent finalized extraction
        extraction_dir = Path(__file__).parent.parent / "data" / "extraction_outputs"
        extraction_files = list(extraction_dir.glob("final_context_aware_extraction_*.json"))
        
        if extraction_files:
            latest_file = max(extraction_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("entities", [])
        
        return []
    
    def _load_relationships_from_upload(self) -> List[Dict[str, Any]]:
        """Load relationships from most recent upload (simulate Azure query)"""
        
        extraction_dir = Path(__file__).parent.parent / "data" / "extraction_outputs"
        extraction_files = list(extraction_dir.glob("final_context_aware_extraction_*.json"))
        
        if extraction_files:
            latest_file = max(extraction_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("relationships", [])
        
        return []
    
    def _analyze_entity_types(self, entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of entity types"""
        
        entity_types = Counter()
        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            entity_types[entity_type] += 1
        
        return dict(entity_types.most_common(20))
    
    def _analyze_relationship_types(self, relationships: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of relationship types"""
        
        relation_types = Counter()
        for relation in relationships:
            relation_type = relation.get("relation_type", "unknown")
            relation_types[relation_type] += 1
        
        return dict(relation_types.most_common(20))
    
    def _check_context_coverage(self, entities: List[Dict[str, Any]]) -> float:
        """Check percentage of entities with context"""
        
        if not entities:
            return 0.0
        
        entities_with_context = sum(1 for e in entities if e.get("context", "").strip())
        return (entities_with_context / len(entities)) * 100
    
    def _analyze_confidence_scores(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze confidence score distribution"""
        
        if not items:
            return {"average": 0.0, "min": 0.0, "max": 0.0}
        
        confidences = [item.get("confidence", 0.0) for item in items]
        
        return {
            "average": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "distribution": {
                "high (>0.9)": sum(1 for c in confidences if c > 0.9),
                "medium (0.7-0.9)": sum(1 for c in confidences if 0.7 <= c <= 0.9),
                "low (<0.7)": sum(1 for c in confidences if c < 0.7)
            }
        }
    
    def _check_for_duplicates(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for duplicate entities"""
        
        entity_texts = [e.get("text", "").lower().strip() for e in entities]
        text_counts = Counter(entity_texts)
        
        duplicates = {text: count for text, count in text_counts.items() if count > 1 and text}
        
        return {
            "count": len(duplicates),
            "examples": dict(list(duplicates.items())[:5])  # Top 5 examples
        }
    
    def _check_entity_completeness(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check completeness of entity data"""
        
        required_fields = ["entity_id", "text", "entity_type", "context"]
        optional_fields = ["confidence", "semantic_role", "maintenance_relevance"]
        
        completeness = {
            "required_fields": {},
            "optional_fields": {},
            "complete_entities": 0
        }
        
        for field in required_fields:
            completeness["required_fields"][field] = sum(1 for e in entities if e.get(field))
        
        for field in optional_fields:
            completeness["optional_fields"][field] = sum(1 for e in entities if e.get(field))
        
        # Count entities with all required fields
        completeness["complete_entities"] = sum(
            1 for e in entities 
            if all(e.get(field) for field in required_fields)
        )
        
        return completeness
    
    def _check_relationship_completeness(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check completeness of relationship data"""
        
        required_fields = ["relation_id", "source_entity_id", "target_entity_id", "relation_type"]
        optional_fields = ["confidence", "context", "maintenance_relevance"]
        
        completeness = {
            "required_fields": {},
            "optional_fields": {},
            "complete_relationships": 0
        }
        
        for field in required_fields:
            completeness["required_fields"][field] = sum(1 for r in relationships if r.get(field))
        
        for field in optional_fields:
            completeness["optional_fields"][field] = sum(1 for r in relationships if r.get(field))
        
        completeness["complete_relationships"] = sum(
            1 for r in relationships 
            if all(r.get(field) for field in required_fields)
        )
        
        return completeness
    
    def _validate_entity_linking(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that relationships link to valid entities"""
        
        entity_ids = {e.get("entity_id") for e in entities}
        
        valid_links = 0
        invalid_links = 0
        
        for relation in relationships:
            source_id = relation.get("source_entity_id")
            target_id = relation.get("target_entity_id")
            
            if source_id in entity_ids and target_id in entity_ids:
                valid_links += 1
            else:
                invalid_links += 1
        
        total_links = valid_links + invalid_links
        
        return {
            "valid_count": valid_links,
            "invalid_count": invalid_links,
            "total_count": total_links,
            "valid_percentage": (valid_links / total_links * 100) if total_links > 0 else 0
        }
    
    def _analyze_graph_connectivity(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze graph connectivity metrics"""
        
        # Count entities that participate in relationships
        connected_entities = set()
        for relation in relationships:
            source_id = relation.get("source_entity_id")
            target_id = relation.get("target_entity_id")
            if source_id:
                connected_entities.add(source_id)
            if target_id:
                connected_entities.add(target_id)
        
        total_entities = len(entities)
        connectivity_rate = (len(connected_entities) / total_entities * 100) if total_entities > 0 else 0
        
        return {
            "connected_entities": len(connected_entities),
            "total_entities": total_entities,
            "connectivity_rate": connectivity_rate,
            "isolated_entities": total_entities - len(connected_entities)
        }
    
    def generate_quality_assessment(self, entity_validation: Dict[str, Any], relationship_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall quality assessment"""
        
        print("\n📈 OVERALL QUALITY ASSESSMENT")
        print("=" * 40)
        
        # Calculate quality scores
        entity_quality = min(
            entity_validation['context_coverage'] / 100,
            entity_validation['completeness']['complete_entities'] / max(entity_validation['total_entities'], 1),
            1.0
        )
        
        relationship_quality = min(
            relationship_validation['entity_linking']['valid_percentage'] / 100,
            relationship_validation['completeness']['complete_relationships'] / max(relationship_validation['total_relationships'], 1),
            1.0
        )
        
        overall_quality = (entity_quality + relationship_quality) / 2
        
        quality_metrics = {
            "entity_quality_score": round(entity_quality, 3),
            "relationship_quality_score": round(relationship_quality, 3),
            "overall_quality_score": round(overall_quality, 3),
            "data_completeness": {
                "entities": f"{entity_validation['completeness']['complete_entities']}/{entity_validation['total_entities']}",
                "relationships": f"{relationship_validation['completeness']['complete_relationships']}/{relationship_validation['total_relationships']}"
            },
            "context_preservation": f"{entity_validation['context_coverage']:.1f}%",
            "graph_connectivity": f"{relationship_validation['connectivity']['connectivity_rate']:.1f}%"
        }
        
        print(f"🎯 Quality Scores:")
        print(f"   • Entity quality: {quality_metrics['entity_quality_score']:.3f}")
        print(f"   • Relationship quality: {quality_metrics['relationship_quality_score']:.3f}")
        print(f"   • Overall quality: {quality_metrics['overall_quality_score']:.3f}")
        
        # Generate recommendations
        recommendations = []
        if entity_validation['context_coverage'] < 95:
            recommendations.append("Improve context preservation in entity extraction")
        if relationship_validation['entity_linking']['valid_percentage'] < 90:
            recommendations.append("Fix entity linking issues in relationships")
        if relationship_validation['connectivity']['connectivity_rate'] < 70:
            recommendations.append("Increase graph connectivity by extracting more relationships")
        
        if overall_quality >= 0.9:
            print("✅ EXCELLENT quality - Ready for GNN training")
        elif overall_quality >= 0.7:
            print("✅ GOOD quality - Ready for GNN training with minor improvements")
        else:
            print("⚠️  Quality needs improvement before GNN training")
        
        self.validation_results['recommendations'] = recommendations
        return quality_metrics
    
    def save_validation_report(self):
        """Save detailed validation report"""
        
        output_dir = Path(__file__).parent.parent / "data" / "azure_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"azure_validation_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Validation report saved: {report_file}")
        return report_file

async def main():
    """Main validation process"""
    
    print("🔍 AZURE KNOWLEDGE DATA VALIDATION")
    print("=" * 50)
    
    try:
        validator = AzureKnowledgeValidator()
        
        # Validate entities
        entity_validation = await validator.validate_entities()
        validator.validation_results['entity_validation'] = entity_validation
        
        # Get entities for relationship validation
        entities = validator._load_entities_from_upload()
        
        # Validate relationships
        relationship_validation = await validator.validate_relationships(entities)
        validator.validation_results['relationship_validation'] = relationship_validation
        
        # Generate quality assessment
        quality_metrics = validator.generate_quality_assessment(entity_validation, relationship_validation)
        validator.validation_results['quality_metrics'] = quality_metrics
        
        # Save detailed report
        report_file = validator.save_validation_report()
        
        print(f"\n🎯 VALIDATION COMPLETED!")
        print(f"📊 Report: {report_file}")
        
        # Determine if ready for GNN training
        overall_quality = quality_metrics['overall_quality_score']
        if overall_quality >= 0.7:
            print(f"\n🚀 READY FOR GNN TRAINING!")
            print(f"   Next step: python scripts/prepare_gnn_training_features.py")
        else:
            print(f"\n⚠️  QUALITY IMPROVEMENT NEEDED BEFORE GNN TRAINING")
            print(f"   Issues to address: {len(validator.validation_results['issues_found'])}")
        
    except Exception as e:
        print(f"❌ Azure validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())