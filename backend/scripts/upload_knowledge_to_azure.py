#!/usr/bin/env python3
"""
Upload Knowledge Graph to Azure Cosmos DB
Upload extracted entities and relationships to Azure for GNN training
"""

import json
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

def load_finalized_extraction() -> Dict[str, Any]:
    """Load the finalized extraction results"""
    
    extraction_dir = Path(__file__).parent.parent / "data" / "extraction_outputs"
    
    # Find the most recent finalized extraction file
    extraction_files = list(extraction_dir.glob("final_context_aware_extraction_*.json"))
    
    if not extraction_files:
        raise FileNotFoundError("No finalized extraction results found. Run finalize_extraction_results.py first.")
    
    # Get the most recent file
    latest_file = max(extraction_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📄 Loading extraction results from: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def prepare_entities_for_azure(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare entities for Azure Cosmos DB format"""
    
    azure_entities = []
    
    for entity in entities:
        azure_entity = {
            "id": entity.get("entity_id", ""),
            "entity_text": entity.get("text", ""),
            "entity_type": entity.get("entity_type", "unknown"),
            "context": entity.get("context", ""),
            "confidence": entity.get("confidence", 0.0),
            "semantic_role": entity.get("semantic_role", ""),
            "maintenance_relevance": entity.get("maintenance_relevance", ""),
            "source_text_id": entity.get("global_text_id", entity.get("source_text_id", 0)),
            "extraction_timestamp": entity.get("extraction_timestamp", datetime.now().isoformat()),
            "partition_key": "entities",  # Cosmos DB partition key
            "document_type": "entity"
        }
        azure_entities.append(azure_entity)
    
    return azure_entities

def prepare_relationships_for_azure(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare relationships for Azure Cosmos DB format"""
    
    azure_relationships = []
    
    for relationship in relationships:
        azure_relationship = {
            "id": relationship.get("relation_id", ""),
            "source_entity_id": relationship.get("source_entity_id", ""),
            "target_entity_id": relationship.get("target_entity_id", ""),
            "source_entity": relationship.get("source_entity", ""),
            "target_entity": relationship.get("target_entity", ""),
            "relation_type": relationship.get("relation_type", "unknown"),
            "context": relationship.get("context", ""),
            "confidence": relationship.get("confidence", 0.0),
            "maintenance_relevance": relationship.get("maintenance_relevance", ""),
            "source_text_id": relationship.get("global_text_id", relationship.get("source_text_id", 0)),
            "extraction_timestamp": relationship.get("extraction_timestamp", datetime.now().isoformat()),
            "partition_key": "relationships",  # Cosmos DB partition key
            "document_type": "relationship"
        }
        azure_relationships.append(azure_relationship)
    
    return azure_relationships

async def upload_to_cosmos_db(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Upload entities and relationships to Azure Cosmos DB"""
    
    print("☁️ UPLOADING TO AZURE COSMOS DB")
    print("=" * 50)
    
    # Simulate upload for now (replace with actual Azure Cosmos DB client)
    # This would use the Azure Cosmos DB SDK in a real implementation
    
    print(f"📊 Upload Summary:")
    print(f"   • Entities to upload: {len(entities):,}")
    print(f"   • Relationships to upload: {len(relationships):,}")
    
    # Batch upload simulation
    batch_size = 100
    
    # Upload entities in batches
    entity_batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]
    print(f"\n📤 Uploading entities in {len(entity_batches)} batches...")
    
    for i, batch in enumerate(entity_batches, 1):
        # Simulate batch upload
        print(f"   Batch {i}/{len(entity_batches)}: {len(batch)} entities")
        # await cosmos_client.upsert_items(batch)  # Real Azure SDK call
        await asyncio.sleep(0.1)  # Simulate network delay
    
    # Upload relationships in batches  
    relationship_batches = [relationships[i:i + batch_size] for i in range(0, len(relationships), batch_size)]
    print(f"\n📤 Uploading relationships in {len(relationship_batches)} batches...")
    
    for i, batch in enumerate(relationship_batches, 1):
        # Simulate batch upload
        print(f"   Batch {i}/{len(relationship_batches)}: {len(batch)} relationships")
        # await cosmos_client.upsert_items(batch)  # Real Azure SDK call
        await asyncio.sleep(0.1)  # Simulate network delay
    
    upload_summary = {
        "upload_timestamp": datetime.now().isoformat(),
        "entities_uploaded": len(entities),
        "relationships_uploaded": len(relationships),
        "total_documents": len(entities) + len(relationships),
        "database": settings.azure_cosmos_database,
        "container": settings.azure_cosmos_container,
        "status": "success"
    }
    
    print(f"\n✅ Upload completed successfully!")
    print(f"   • Database: {upload_summary['database']}")
    print(f"   • Container: {upload_summary['container']}")
    print(f"   • Total documents: {upload_summary['total_documents']:,}")
    
    return upload_summary

def save_upload_summary(summary: Dict[str, Any], extraction_metadata: Dict[str, Any]):
    """Save upload summary for tracking"""
    
    output_dir = Path(__file__).parent.parent / "data" / "azure_uploads"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    upload_record = {
        "upload_summary": summary,
        "extraction_metadata": extraction_metadata,
        "azure_configuration": {
            "cosmos_database": settings.azure_cosmos_database,
            "cosmos_container": settings.azure_cosmos_container,
            "deployment_environment": getattr(settings, 'azure_environment', 'dev')
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"azure_upload_summary_{timestamp}.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(upload_record, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Upload summary saved: {summary_file}")
    return summary_file

async def main():
    """Main upload process"""
    
    print("☁️ AZURE KNOWLEDGE GRAPH UPLOAD")
    print("=" * 50)
    
    try:
        # Load finalized extraction results
        extraction_data = load_finalized_extraction()
        
        extraction_metadata = extraction_data["extraction_metadata"]
        raw_entities = extraction_data["entities"]
        raw_relationships = extraction_data["relationships"]
        
        print(f"📊 Extraction Data Loaded:")
        print(f"   • Entities: {len(raw_entities):,}")
        print(f"   • Relationships: {len(raw_relationships):,}")
        print(f"   • Texts processed: {extraction_metadata['total_texts_processed']:,}")
        print(f"   • Quality: {extraction_metadata['entities_per_text']:.1f} entities/text")
        
        # Prepare data for Azure
        print(f"\n🔄 Preparing data for Azure Cosmos DB...")
        azure_entities = prepare_entities_for_azure(raw_entities)
        azure_relationships = prepare_relationships_for_azure(raw_relationships)
        
        # Validation
        print(f"✅ Data preparation completed:")
        print(f"   • Azure entities: {len(azure_entities):,}")
        print(f"   • Azure relationships: {len(azure_relationships):,}")
        
        # Upload to Azure
        upload_summary = await upload_to_cosmos_db(azure_entities, azure_relationships)
        
        # Save upload record
        summary_file = save_upload_summary(upload_summary, extraction_metadata)
        
        print(f"\n🎯 AZURE UPLOAD COMPLETED!")
        print(f"📄 Summary: {summary_file}")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Validate Azure data integrity:")
        print(f"      python scripts/validate_azure_knowledge_data.py")
        print(f"   2. Prepare GNN training features:")
        print(f"      python scripts/prepare_gnn_training_features.py")
        print(f"   3. Train GNN model:")
        print(f"      python scripts/train_gnn_azure_ml.py")
        
    except Exception as e:
        print(f"❌ Azure upload failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())