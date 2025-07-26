"""
Universal Quality Assessment for Azure Prompt Flow
Evaluates extraction quality without domain-specific assumptions
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


def assess_extraction_quality(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]], 
    original_texts: List[str]
) -> Dict[str, Any]:
    """
    Universal quality assessment for knowledge extraction
    
    Args:
        entities: List of extracted entity dictionaries
        relations: List of extracted relation dictionaries
        original_texts: Original input texts for analysis
        
    Returns:
        Quality assessment metrics and recommendations
    """
    try:
        # Basic extraction metrics
        total_entities = len(entities)
        total_relations = len(relations)
        total_texts = len(original_texts)
        
        # Entity quality analysis
        entity_confidence_scores = [e.get("confidence", 0.0) for e in entities]
        avg_entity_confidence = sum(entity_confidence_scores) / len(entity_confidence_scores) if entity_confidence_scores else 0.0
        
        entity_types = [e.get("entity_type", "") for e in entities]
        unique_entity_types = len(set(entity_types))
        entity_type_distribution = dict(Counter(entity_types))
        
        # Relation quality analysis  
        relation_confidence_scores = [r.get("confidence", 0.0) for r in relations]
        avg_relation_confidence = sum(relation_confidence_scores) / len(relation_confidence_scores) if relation_confidence_scores else 0.0
        
        relation_types = [r.get("relation_type", "") for r in relations]
        unique_relation_types = len(set(relation_types))
        relation_type_distribution = dict(Counter(relation_types))
        
        # Text analysis metrics
        total_chars = sum(len(text) for text in original_texts)
        avg_text_length = total_chars / total_texts if total_texts > 0 else 0
        
        # Universal quality indicators (no domain assumptions)
        quality_indicators = []
        overall_score = 1.0
        
        # Check extraction density
        entities_per_text = total_entities / total_texts if total_texts > 0 else 0
        if entities_per_text < 1:
            quality_indicators.append("Low entity extraction density")
            overall_score *= 0.8
        elif entities_per_text > 20:
            quality_indicators.append("Very high entity extraction - possible over-extraction")
            overall_score *= 0.9
            
        # Check relation coverage
        relations_per_entity = total_relations / total_entities if total_entities > 0 else 0
        if relations_per_entity < 0.3:
            quality_indicators.append("Few relationships identified - knowledge graph may be sparse")
            overall_score *= 0.7
            
        # Check confidence levels
        if avg_entity_confidence < 0.6:
            quality_indicators.append("Low average entity confidence")
            overall_score *= 0.8
        if avg_relation_confidence < 0.6:
            quality_indicators.append("Low average relation confidence")  
            overall_score *= 0.8
            
        # Check type diversity
        if unique_entity_types < total_entities * 0.3:
            quality_indicators.append("Limited entity type diversity - many duplicate types")
            overall_score *= 0.9
        if unique_relation_types < 3:
            quality_indicators.append("Very limited relation type diversity")
            overall_score *= 0.8
            
        # Generate quality tier
        if overall_score >= 0.9:
            quality_tier = "excellent"
        elif overall_score >= 0.75:
            quality_tier = "good"
        elif overall_score >= 0.6:
            quality_tier = "acceptable"
        else:
            quality_tier = "needs_improvement"
            
        # Universal recommendations (no domain bias)
        recommendations = []
        if entities_per_text < 1:
            recommendations.append("Consider providing longer or more detailed text samples")
        if relations_per_entity < 0.3:
            recommendations.append("Text may benefit from more explicit relationship descriptions")
        if avg_entity_confidence < 0.7:
            recommendations.append("Review text quality and clarity for better entity extraction")
        if unique_relation_types < 5:
            recommendations.append("Provide text with more diverse relationship patterns")
            
        # Compile quality assessment
        quality_assessment = {
            "overall_score": round(overall_score, 3),
            "quality_tier": quality_tier,
            "extraction_metrics": {
                "total_entities": total_entities,
                "total_relations": total_relations,
                "unique_entity_types": unique_entity_types,
                "unique_relation_types": unique_relation_types,
                "entities_per_text": round(entities_per_text, 2),
                "relations_per_entity": round(relations_per_entity, 2)
            },
            "confidence_metrics": {
                "avg_entity_confidence": round(avg_entity_confidence, 3),
                "avg_relation_confidence": round(avg_relation_confidence, 3),
                "high_confidence_entities": len([e for e in entities if e.get("confidence", 0) > 0.8]),
                "high_confidence_relations": len([r for r in relations if r.get("confidence", 0) > 0.8])
            },
            "text_analysis": {
                "total_texts": total_texts,
                "total_characters": total_chars,
                "avg_text_length": round(avg_text_length, 1)
            },
            "quality_indicators": quality_indicators,
            "recommendations": recommendations,
            "entity_type_distribution": entity_type_distribution,
            "relation_type_distribution": relation_type_distribution,
            "assessment_timestamp": datetime.now().isoformat(),
            "assessment_method": "universal_prompt_flow"
        }
        
        logger.info(f"Quality assessment completed: {quality_tier} ({overall_score:.3f})")
        
        return quality_assessment
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}", exc_info=True)
        return {
            "overall_score": 0.0,
            "quality_tier": "assessment_failed",
            "error": str(e),
            "assessment_timestamp": datetime.now().isoformat()
        }


# Main entry point for Azure Prompt Flow
def main(entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], original_texts: List[str]) -> Dict[str, Any]:
    """Main function called by Azure Prompt Flow"""
    return assess_extraction_quality(entities, relations, original_texts)


if __name__ == "__main__":
    # Test with sample data
    sample_entities = [
        {"entity_id": "e1", "text": "valve", "entity_type": "valve", "confidence": 0.8},
        {"entity_id": "e2", "text": "bearing", "entity_type": "bearing", "confidence": 0.9}
    ]
    sample_relations = [
        {"relation_id": "r1", "relation_type": "connected_to", "confidence": 0.8}
    ]
    sample_texts = ["The valve is connected to the bearing through a hydraulic system."]
    
    result = main(sample_entities, sample_relations, sample_texts)
    print(json.dumps(result, indent=2))