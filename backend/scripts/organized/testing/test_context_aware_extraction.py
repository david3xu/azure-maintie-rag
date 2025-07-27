#!/usr/bin/env python3
"""
Test Context-Aware Knowledge Extraction
Uses the new context engineering approach instead of constraining prompt engineering
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_maintenance_texts() -> List[str]:
    """Load a sample of real maintenance texts for testing"""
    
    raw_file = Path("data/raw/maintenance_all_texts.md")
    if not raw_file.exists():
        logger.warning(f"Raw file not found: {raw_file}")
        # Use hardcoded sample
        return [
            "air conditioner thermostat not working",
            "bearing on air conditioner compressor unserviceable", 
            "blown o-ring off steering hose",
            "brake system pressure low",
            "coolant temperature sensor malfunction",
            "diesel engine fuel filter clogged",
            "hydraulic pump pressure relief valve stuck"
        ]
    
    # Load first 10 real maintenance texts
    texts = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('<id>'):
                text_content = line.replace('<id>', '').strip()
                if text_content:
                    texts.append(text_content)
                    if len(texts) >= 10:  # Sample size for testing
                        break
    
    logger.info(f"Loaded {len(texts)} maintenance texts for testing")
    return texts

async def test_context_aware_entity_extraction(texts: List[str]) -> Dict[str, Any]:
    """Test the new context-aware entity extraction"""
    
    try:
        # Import the Azure OpenAI service
        from core.azure_openai.completion_service import AzureOpenAICompletionService
        from jinja2 import Environment, FileSystemLoader
        
        # Initialize Azure OpenAI service
        openai_service = AzureOpenAICompletionService()
        
        # Load the context-aware template
        template_dir = Path("prompt_flows/universal_knowledge_extraction")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("context_aware_entity_extraction.jinja2")
        
        # Render the prompt with context
        prompt = template.render(texts=texts)
        
        logger.info("Testing context-aware entity extraction...")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Call Azure OpenAI
        response = await openai_service.complete_chat(
            messages=[
                {"role": "system", "content": "You are an expert maintenance engineer."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4",
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse the response
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text
        
        try:
            entities = json.loads(json_text)
            logger.info(f"Successfully extracted {len(entities)} entities")
            
            return {
                "success": True,
                "entities": entities,
                "entity_count": len(entities),
                "sample_entities": entities[:5],  # First 5 for preview
                "raw_response": response_text
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                "success": False,
                "error": f"JSON parsing failed: {e}",
                "raw_response": response_text
            }
            
    except Exception as e:
        logger.error(f"Context-aware extraction failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def test_context_aware_relation_extraction(texts: List[str], entities: List[str]) -> Dict[str, Any]:
    """Test the new context-aware relationship extraction"""
    
    try:
        from core.azure_openai.completion_service import AzureOpenAICompletionService
        from jinja2 import Environment, FileSystemLoader
        
        openai_service = AzureOpenAICompletionService()
        
        # Load the context-aware template
        template_dir = Path("prompt_flows/universal_knowledge_extraction")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("context_aware_relation_extraction.jinja2")
        
        # Render the prompt with context
        prompt = template.render(texts=texts, entities=entities)
        
        logger.info("Testing context-aware relationship extraction...")
        
        # Call Azure OpenAI
        response = await openai_service.complete_chat(
            messages=[
                {"role": "system", "content": "You are an expert maintenance engineer."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4",
            temperature=0.3,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text
        
        try:
            relationships = json.loads(json_text)
            logger.info(f"Successfully extracted {len(relationships)} relationships")
            
            return {
                "success": True,
                "relationships": relationships,
                "relationship_count": len(relationships),
                "sample_relationships": relationships[:5],
                "raw_response": response_text
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                "success": False,
                "error": f"JSON parsing failed: {e}",
                "raw_response": response_text
            }
            
    except Exception as e:
        logger.error(f"Context-aware relation extraction failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def run_complete_context_aware_test():
    """Run complete test of context-aware extraction approach"""
    
    print("🧪 Testing Context-Aware Knowledge Extraction")
    print("=" * 60)
    
    # Load sample texts
    texts = load_sample_maintenance_texts()
    print(f"📝 Testing with {len(texts)} maintenance texts:")
    for i, text in enumerate(texts[:3], 1):
        print(f"   {i}. {text}")
    print("   ...")
    
    # Test entity extraction
    print("\n🔍 Testing Context-Aware Entity Extraction...")
    entity_results = await test_context_aware_entity_extraction(texts)
    
    if entity_results["success"]:
        print(f"✅ Entity Extraction Success: {entity_results['entity_count']} entities")
        print("📊 Sample Entities:")
        for entity in entity_results["sample_entities"]:
            print(f"   • {entity.get('text', 'N/A')} ({entity.get('entity_type', 'N/A')}) - confidence: {entity.get('confidence', 'N/A')}")
    else:
        print(f"❌ Entity Extraction Failed: {entity_results['error']}")
        print("Raw response:", entity_results.get('raw_response', 'N/A')[:200])
        return
    
    # Extract entity names for relationship extraction
    entity_names = [e.get('text', '') for e in entity_results['entities']]
    
    # Test relationship extraction
    print("\n🔗 Testing Context-Aware Relationship Extraction...")
    relation_results = await test_context_aware_relation_extraction(texts, entity_names)
    
    if relation_results["success"]:
        print(f"✅ Relationship Extraction Success: {relation_results['relationship_count']} relationships")
        print("📊 Sample Relationships:")
        for rel in relation_results["sample_relationships"]:
            print(f"   • {rel.get('source_entity', 'N/A')} --{rel.get('relation_type', 'N/A')}--> {rel.get('target_entity', 'N/A')} (conf: {rel.get('confidence', 'N/A')})")
    else:
        print(f"❌ Relationship Extraction Failed: {relation_results['error']}")
        print("Raw response:", relation_results.get('raw_response', 'N/A')[:200])
        return
    
    # Save results for analysis
    output_file = Path("data/extraction_outputs/context_aware_test_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    combined_results = {
        "test_info": {
            "approach": "context_aware_extraction",
            "test_date": "2025-07-26",
            "texts_processed": len(texts),
            "source_texts": texts
        },
        "entity_extraction": entity_results,
        "relationship_extraction": relation_results,
        "quality_assessment": {
            "entities_per_text": entity_results['entity_count'] / len(texts),
            "relationships_per_text": relation_results['relationship_count'] / len(texts),
            "extraction_success": entity_results['success'] and relation_results['success']
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n📈 Quality Assessment:")
    print(f"   • Entities per text: {combined_results['quality_assessment']['entities_per_text']:.1f}")
    print(f"   • Relationships per text: {combined_results['quality_assessment']['relationships_per_text']:.1f}")
    print(f"   • Overall success: {combined_results['quality_assessment']['extraction_success']}")
    
    # Compare with old approach
    print("\n📊 Comparison with Old Approach:")
    print("   Old (Constraining):")
    print("     • 50 total entities (not per text)")
    print("     • 30 total relationships (not per text)")  
    print("     • Generic types only, no instances")
    print("     • No context preservation")
    print("   New (Context-Aware):")
    print(f"     • {entity_results['entity_count']} entities with full context")
    print(f"     • {relation_results['relationship_count']} relationships with confidence")
    print("     • Specific instances from each text")
    print("     • Rich context and semantic roles")

if __name__ == "__main__":
    asyncio.run(run_complete_context_aware_test())